"""Torch CPU thread policy shared by the torch-backed sidecar workers.

`deeplc_worker.py`, `deeplc_finetune.py` and `im2deep_worker.py` size torch's CPU pool from
the thread count the engine passes, capped at the physical cores the process may run on
(`thread_cap`, `capped_threads`); `nn_rescore_worker.py` reads the physical core count for
its process sizing. These helpers used to be copied into each worker, and the copies had
already drifted apart. Pure standard library, so importing it has no load-order constraint
of its own; the workers import it after their torch-backed package all the same, keeping
that package's import first (see `deeplc_worker.py`). Python puts a script's own directory
on `sys.path`, so `from _torch_threads import ...` resolves wherever the worker is invoked
from.
"""

import os
import sys


def _windows_physical_cores():
    """Physical core count from `GetLogicalProcessorInformationEx(RelationProcessorCore)`.

    One record per physical core, across every processor group, whatever its efficiency
    class: a hybrid CPU's efficiency cores are real cores for a forward pass, unlike a
    second hyperthread on a core that is already busy. None when the call fails.
    """
    try:
        import ctypes
        import struct
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32
        relation_processor_core = 0
        size = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if not k32.GetLogicalProcessorInformationEx(
            relation_processor_core, buf, ctypes.byref(size)
        ):
            return None
        raw = buf.raw
        cores = 0
        off = 0
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship (u32), Size (u32), payload.
        while off + 8 <= size.value:
            relationship, record_size = struct.unpack_from("<II", raw, off)
            if record_size < 8:
                return None
            if relationship == relation_processor_core:
                cores += 1
            off += record_size
        return cores or None
    except Exception:  # noqa: BLE001 - detection is best effort; None means no cap
        return None


def _sysfs_physical_cores(cpus, sysfs="/sys/devices/system/cpu"):
    """The number of physical cores behind the logical CPUs `cpus`, from sysfs, or None.

    Each CPU's `topology/core_cpus_list` (Linux 5.5 and later; `thread_siblings_list`
    before) names the logical CPUs that share its physical core, so its contents identify
    that core on every architecture, and the hyperthreads of one core count once. The
    `(physical_package_id, core_id)` pair is the fallback only: on many device-tree ARM64
    systems `core_id` restarts in each cluster and the package id is the same for every
    CPU, so the pairs of two clusters collide and eight cores count as four. None when a
    CPU has no topology at all (some containers), which leaves the count to the caller.
    """
    cores = set()
    for cpu in cpus:
        topo = "%s/cpu%d/topology/" % (sysfs, cpu)
        key = None
        for name in ("core_cpus_list", "thread_siblings_list"):
            try:
                with open(topo + name, encoding="ascii") as fh:
                    key = "cpus " + fh.read().strip()
                break
            except OSError:
                continue
        if key is None:
            try:
                with open(topo + "physical_package_id", encoding="ascii") as fh:
                    package = fh.read().strip()
                with open(topo + "core_id", encoding="ascii") as fh:
                    core = fh.read().strip()
            except OSError:
                return None
            key = "pair %s/%s" % (package, core)
        cores.add(key)
    return len(cores) or None


def physical_cores():
    """`(count, how)`: the physical cores this process may run on, or `(None, why)`.

    Linux: the distinct physical cores (`_sysfs_physical_cores`) behind the CPUs in
    `sched_getaffinity(0)`, so a container or a `taskset` mask is respected and two
    hyperthreads of one core count once. Windows: every physical core of the machine.
    Anywhere else the count is unknown and nothing is capped.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            cpus = sorted(os.sched_getaffinity(0))
        except OSError:
            cpus = []
        if cpus:
            n = _sysfs_physical_cores(cpus)
            if n is None:
                return len(cpus), "%d CPUs in the affinity mask (no sysfs topology)" % len(cpus)
            return n, "%d physical cores under the affinity mask of %d CPUs" % (n, len(cpus))
    if sys.platform == "win32":
        n = _windows_physical_cores()
        if n:
            return n, "%d physical cores (GetLogicalProcessorInformationEx)" % n
    return None, "physical core count unknown on this platform"


def thread_cap(env_var, default):
    """`(cap, why)` for a torch worker's CPU threads; a cap of 0 means none.

    `env_var` names the worker's own override (`MUMDIA_DEEPLC_THREAD_CAP`,
    `MUMDIA_IM2DEEP_THREAD_CAP`) and `default` the value it reads as when unset (the workers
    pass `"auto"`). At `auto` the cap is the physical core count (`physical_cores`). Measured on doxy (EPYC, 64 cores, 128 CPUs): the DeepLC multi-head
    step took 10:41 at 96 requested threads and 18:09 at 128, because every OpenMP-parallel
    op waits for its slowest thread and a second hyperthread on a busy core is the slowest
    one. The cap is a ceiling on what the engine asks for, never a target: a request at or
    below it is taken as given. The engine asks for every logical CPU unless `--threads`
    says otherwise, so on an SMT host the cap binds by default. 0 disables it. A value that
    is not a finite number (`twelve`, `nan`, `inf`) is reported and ignored.
    """
    raw = os.environ.get(env_var, default).strip().lower()
    if raw not in ("", "auto"):
        try:
            value = int(float(raw))
        except (ValueError, OverflowError):
            print("WARNING: %s=%r is not a number, 0 or auto; "
                  "using auto" % (env_var, raw), flush=True)
        else:
            if value <= 0:
                return 0, "%s=0 (no cap)" % env_var
            return value, env_var
    n, why = physical_cores()
    return (n or 0), why


def capped_threads(requested, cap):
    """`requested` bounded by `cap` (0 = no cap), and at least 1."""
    requested = max(1, int(requested))
    return requested if cap <= 0 else max(1, min(requested, cap))
