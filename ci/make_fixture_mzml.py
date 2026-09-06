#!/usr/bin/env python
"""Generate the deterministic mzML fixture the end-to-end smoke test runs on.

Why generate instead of committing a data file: a real DIA run is tens of
gigabytes, a usable slice is still megabytes of binary in git, and any excerpt of
a public raw file carries a licence question. Nothing was checked in as a result,
so CI never ran the engine end to end and `convert` (the one stage with an
external parser) had no test at all.

Why generate FROM the engine's own library: the planted fragment peaks have to sit
at the m/z the engine expects, to within its tolerance. Recomputing peptide and
fragment masses here would duplicate the mass model and drift from it. Instead
this script reads `fragment_library_precursors.parquet` and
`fragment_library_fragments.parquet` that `mumdia predict-frag` just produced, and
plants exactly those m/z values. The fixture therefore cannot disagree with the
engine about masses; if the mass model changes, the fixture changes with it.

What it writes: a small DIA run. Each cycle is one MS1 scan followed by one MS2
scan per isolation window. A chosen set of TARGET precursors is planted with a
Gaussian elution profile across neighbouring cycles, in the MS2 windows that
contain their m/z, using their library fragments and predicted intensities.
Decoys are never planted, so target-decoy separation is real and FDR has
something to estimate. Seeded noise peaks are added so matching is not trivially
easy.

Retention time is planted as an affine function of the library's `predicted_irt`
plus bounded jitter, so `rt-im-train` has a genuine relationship to calibrate
rather than a constant.

Usage:
    python ci/make_fixture_mzml.py \\
        --precursors work/lib_prec.parquet \\
        --fragments  work/lib_frag.parquet \\
        --out test_data/fixture.mzML \\
        --manifest work/fixture_planted.json

The manifest lists what was planted; `ci/check_smoke.py` asserts the run recovers
it. Output is byte-deterministic for a given input library and options.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import sys
from pathlib import Path

import pyarrow.parquet as pq

# mzML controlled-vocabulary accessions used below.
CV = {
    "ms_level": ("MS:1000511", "ms level"),
    "centroid": ("MS:1000127", "centroid spectrum"),
    "ms1_spectrum": ("MS:1000579", "MS1 spectrum"),
    "msn_spectrum": ("MS:1000580", "MSn spectrum"),
    "positive": ("MS:1000130", "positive scan"),
    "scan_start": ("MS:1000016", "scan start time"),
    "f64": ("MS:1000523", "64-bit float"),
    "f32": ("MS:1000521", "32-bit float"),
    "no_compression": ("MS:1000576", "no compression"),
    "mz_array": ("MS:1000514", "m/z array"),
    "intensity_array": ("MS:1000515", "intensity array"),
    "iso_target": ("MS:1000827", "isolation window target m/z"),
    "iso_lower": ("MS:1000828", "isolation window lower offset"),
    "iso_upper": ("MS:1000829", "isolation window upper offset"),
    "selected_ion_mz": ("MS:1000744", "selected ion m/z"),
    "charge_state": ("MS:1000041", "charge state"),
    "cid": ("MS:1000133", "collision-induced dissociation"),
}


def splitmix64(state: int) -> tuple[int, int]:
    """One step of splitmix64. A named, portable generator rather than
    `random`, so the fixture is identical on every Python version and platform.
    """
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


class Rng:
    """Deterministic uniform floats in [0, 1)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_float(self) -> float:
        self.state, value = splitmix64(self.state)
        return (value >> 11) / float(1 << 53)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.next_float()


def encode_f64(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}d", *values)).decode("ascii")


def encode_f32(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")


def cvparam(key: str, value: str = "", extra: str = "") -> str:
    acc, name = CV[key]
    return f'<cvParam cvRef="MS" accession="{acc}" name="{name}" value="{value}"{extra}/>'


def binary_array(values: list[float], kind: str) -> str:
    """One `binaryDataArray`. m/z is 64-bit and intensity 32-bit, which is what
    real files do and what keeps the fixture small.
    """
    if kind == "mz":
        encoded = encode_f64(values)
        dtype, array_cv = "f64", "mz_array"
    else:
        encoded = encode_f32(values)
        dtype, array_cv = "f32", "intensity_array"
    return (
        f'      <binaryDataArray encodedLength="{len(encoded)}">\n'
        f"        {cvparam(dtype)}\n"
        f"        {cvparam('no_compression')}\n"
        f"        {cvparam(array_cv)}\n"
        f"        <binary>{encoded}</binary>\n"
        f"      </binaryDataArray>\n"
    )


def spectrum_xml(
    index: int,
    scan_number: int,
    ms_level: int,
    rt_minutes: float,
    mzs: list[float],
    intensities: list[float],
    window: tuple[float, float, float] | None = None,
) -> str:
    """One `<spectrum>`.

    Spectra are declared centroid so `convert` does not run its local-maxima
    centroiding over them. That matters: centroiding would move the planted m/z
    by a fraction of the sample spacing and the fixture's whole point is that the
    peaks sit exactly where the library says.

    Scan start time is written in minutes with an explicit unit, because
    `convert` multiplies `start_time()` by 60 to reach seconds.
    """
    unit = ' unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"'
    level_cv = "ms1_spectrum" if ms_level == 1 else "msn_spectrum"
    parts = [
        f'    <spectrum index="{index}" id="scan={scan_number}" '
        f'defaultArrayLength="{len(mzs)}">\n',
        f"      {cvparam(level_cv)}\n",
        f"      {cvparam('ms_level', str(ms_level))}\n",
        f"      {cvparam('centroid')}\n",
        f"      {cvparam('positive')}\n",
        "      <scanList count=\"1\">\n",
        '        <cvParam cvRef="MS" accession="MS:1000795" name="no combination" value=""/>\n',
        "        <scan>\n",
        f"          {cvparam('scan_start', f'{rt_minutes:.6f}', unit)}\n",
        "        </scan>\n",
        "      </scanList>\n",
    ]
    if window is not None:
        target, lower, upper = window
        parts += [
            '      <precursorList count="1">\n',
            "        <precursor>\n",
            "          <isolationWindow>\n",
            f"            {cvparam('iso_target', f'{target:.4f}')}\n",
            f"            {cvparam('iso_lower', f'{lower:.4f}')}\n",
            f"            {cvparam('iso_upper', f'{upper:.4f}')}\n",
            "          </isolationWindow>\n",
            # A DIA window isolates a range, not one ion. Real DIA files still
            # carry a selectedIon at the window centre, and `convert` reads it
            # into `precursor_mz`, so the fixture writes one too.
            '          <selectedIonList count="1">\n',
            "            <selectedIon>\n",
            f"              {cvparam('selected_ion_mz', f'{target:.4f}')}\n",
            "            </selectedIon>\n",
            "          </selectedIonList>\n",
            "          <activation>\n",
            f"            {cvparam('cid')}\n",
            "          </activation>\n",
            "        </precursor>\n",
            "      </precursorList>\n",
        ]
    parts.append(f'      <binaryDataArrayList count="2">\n')
    parts.append(binary_array(mzs, "mz"))
    parts.append(binary_array(intensities, "intensity"))
    parts.append("      </binaryDataArrayList>\n")
    parts.append("    </spectrum>\n")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precursors", required=True, help="fragment_library_precursors.parquet")
    ap.add_argument("--fragments", required=True, help="fragment_library_fragments.parquet")
    ap.add_argument("--out", required=True, help="mzML to write")
    ap.add_argument("--manifest", default=None, help="JSON listing what was planted")
    ap.add_argument("--n-planted", type=int, default=48, help="target precursors to plant")
    ap.add_argument("--cycles", type=int, default=60, help="MS1/MS2 cycles")
    ap.add_argument("--windows", type=int, default=4, help="MS2 isolation windows per cycle")
    ap.add_argument("--cycle-seconds", type=float, default=2.0)
    ap.add_argument("--peak-sigma-seconds", type=float, default=4.0)
    ap.add_argument("--noise-peaks", type=int, default=12, help="random noise peaks per MS2 scan")
    ap.add_argument(
        "--library-noise-peaks",
        type=int,
        default=10,
        help="noise peaks per MS2 scan drawn from the library's own fragment m/z pool",
    )
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="print the summary but not the per-precursor listing. Exists so a caller "
             "never has to truncate this output with `head`: closing the pipe early "
             "raises OSError [Errno 22] on a Windows console at interpreter shutdown "
             "(not EPIPE), which exits 120 and reads as a mystery CI failure.",
    )
    a = ap.parse_args()

    prec = pq.read_table(
        a.precursors,
        columns=["candidate_id", "peptidoform", "charge", "precursor_mz", "predicted_irt", "label"],
    ).to_pydict()
    frag = pq.read_table(
        a.fragments, columns=["candidate_id", "mz", "predicted_intensity", "name"]
    ).to_pydict()

    targets = [
        i for i, label in enumerate(prec["label"]) if label == "target"
    ]
    if not targets:
        print("error: the library contains no target precursors", file=sys.stderr)
        return 2

    # Deterministic selection: sort by (charge, m/z) and take an even stride, so
    # the planted set spans the m/z range and both charge states instead of
    # clustering at one end.
    targets.sort(key=lambda i: (prec["charge"][i], prec["precursor_mz"][i]))
    stride = max(1, len(targets) // a.n_planted)
    chosen = targets[::stride][: a.n_planted]

    # Fragment m/z pool over the WHOLE library, targets and decoys alike. Noise
    # drawn from it is what gives decoys the chance evidence a real run has: real
    # chemical noise lands on library masses sometimes, which is exactly why
    # target-decoy competition is needed. Without it every accepted candidate here
    # was a target, rescore refused the run for having no decoys, and the fixture
    # could not reach FDR, quant or report at all.
    mz_pool: list[float] = sorted({round(float(m), 6) for m in frag["mz"]})

    frag_by_cand: dict[int, list[tuple[float, float, str]]] = {}
    for cid, mz, inten, name in zip(
        frag["candidate_id"], frag["mz"], frag["predicted_intensity"], frag["name"]
    ):
        frag_by_cand.setdefault(int(cid), []).append((float(mz), float(inten), name))

    # Isolation windows spanning the planted precursors, with a small margin so
    # no planted precursor lands exactly on a boundary.
    mz_values = [prec["precursor_mz"][i] for i in chosen]
    lo, hi = min(mz_values) - 5.0, max(mz_values) + 5.0
    width = (hi - lo) / a.windows
    windows = [(lo + k * width, lo + (k + 1) * width) for k in range(a.windows)]

    # Plant RT as an affine function of predicted iRT plus bounded jitter, so RT
    # calibration has a real relationship to fit. Keep every apex inside the
    # middle of the gradient so its full peak is sampled.
    total_seconds = a.cycles * a.cycle_seconds
    rt_lo, rt_hi = 0.20 * total_seconds, 0.80 * total_seconds
    irts = [float(prec["predicted_irt"][i]) for i in chosen]
    irt_lo, irt_hi = min(irts), max(irts)
    span = (irt_hi - irt_lo) or 1.0
    rng = Rng(a.seed)
    planted = []
    for i, irt in zip(chosen, irts):
        frac = (irt - irt_lo) / span
        apex = rt_lo + frac * (rt_hi - rt_lo) + rng.uniform(-1.5, 1.5)
        fragments = sorted(frag_by_cand.get(int(prec["candidate_id"][i]), []), key=lambda f: -f[1])
        planted.append(
            {
                "candidate_id": int(prec["candidate_id"][i]),
                "peptidoform": prec["peptidoform"][i],
                "charge": int(prec["charge"][i]),
                "precursor_mz": float(prec["precursor_mz"][i]),
                "predicted_irt": irt,
                "apex_seconds": apex,
                "base_intensity": rng.uniform(2.0e5, 2.0e6),
                "fragments": [
                    {"mz": m, "intensity": it, "name": n} for m, it, n in fragments[:6]
                ],
            }
        )
    if not all(p["fragments"] for p in planted):
        print("error: a chosen precursor has no library fragments", file=sys.stderr)
        return 2

    def profile(apex: float, rt: float) -> float:
        z = (rt - apex) / a.peak_sigma_seconds
        return math.exp(-0.5 * z * z)

    spectra: list[str] = []
    index = 0
    scan_number = 1
    for cycle in range(a.cycles):
        cycle_start = cycle * a.cycle_seconds

        # MS1: the monoisotopic precursor peak of every planted precursor, on the
        # same elution profile, plus noise.
        ms1: dict[float, float] = {}
        for p in planted:
            amp = p["base_intensity"] * profile(p["apex_seconds"], cycle_start)
            if amp > 1.0:
                ms1[p["precursor_mz"]] = ms1.get(p["precursor_mz"], 0.0) + amp
        for _ in range(a.noise_peaks):
            ms1[rng.uniform(lo, hi)] = rng.uniform(1.0e3, 2.0e4)
        mzs = sorted(ms1)
        spectra.append(
            spectrum_xml(index, scan_number, 1, cycle_start / 60.0, mzs, [ms1[m] for m in mzs])
        )
        index += 1
        scan_number += 1

        # MS2: one scan per window, offset within the cycle so retention times
        # are strictly increasing as a real acquisition's are.
        for w, (w_lo, w_hi) in enumerate(windows):
            rt = cycle_start + (w + 1) * a.cycle_seconds / (a.windows + 1)
            peaks: dict[float, float] = {}
            for p in planted:
                if not (w_lo <= p["precursor_mz"] < w_hi):
                    continue
                amp = p["base_intensity"] * profile(p["apex_seconds"], rt)
                if amp <= 1.0:
                    continue
                for f in p["fragments"]:
                    peaks[f["mz"]] = peaks.get(f["mz"], 0.0) + amp * f["intensity"]
            for _ in range(a.noise_peaks):
                peaks[rng.uniform(150.0, 1500.0)] = rng.uniform(1.0e3, 2.0e4)
            # Library-coincident noise, well below the planted amplitudes so it
            # cannot outscore a real peptide, and unstructured in RT so it does
            # not form a credible elution profile.
            for _ in range(a.library_noise_peaks):
                mz = mz_pool[int(rng.next_float() * len(mz_pool)) % len(mz_pool)]
                peaks[mz] = peaks.get(mz, 0.0) + rng.uniform(2.0e3, 3.0e4)
            mzs = sorted(peaks)
            centre = 0.5 * (w_lo + w_hi)
            half = 0.5 * (w_hi - w_lo)
            spectra.append(
                spectrum_xml(
                    index,
                    scan_number,
                    2,
                    rt / 60.0,
                    mzs,
                    [peaks[m] for m in mzs],
                    window=(centre, half, half),
                )
            )
            index += 1
            scan_number += 1

    header = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<mzML xmlns="http://psi.hupo.org/ms/mzml" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://psi.hupo.org/ms/mzml '
        'http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd" version="1.1.0" id="mumdia_fixture">\n'
        '  <cvList count="2">\n'
        '    <cv id="MS" fullName="PSI-MS controlled vocabulary" '
        'URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/>\n'
        '    <cv id="UO" fullName="Unit Ontology" '
        'URI="https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo"/>\n'
        "  </cvList>\n"
        "  <fileDescription>\n"
        "    <fileContent>\n"
        f"      {cvparam('ms1_spectrum')}\n"
        f"      {cvparam('msn_spectrum')}\n"
        "    </fileContent>\n"
        "  </fileDescription>\n"
        '  <softwareList count="1">\n'
        '    <software id="mumdia_fixture" version="1">\n'
        '      <cvParam cvRef="MS" accession="MS:1000799" name="custom unreleased software tool" value="mumdia fixture generator"/>\n'
        "    </software>\n"
        "  </softwareList>\n"
        '  <instrumentConfigurationList count="1">\n'
        '    <instrumentConfiguration id="IC1">\n'
        '      <cvParam cvRef="MS" accession="MS:1000031" name="instrument model" value="synthetic"/>\n'
        "    </instrumentConfiguration>\n"
        "  </instrumentConfigurationList>\n"
        '  <dataProcessingList count="1">\n'
        '    <dataProcessing id="DP1">\n'
        '      <processingMethod order="1" softwareRef="mumdia_fixture">\n'
        '        <cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML" value=""/>\n'
        "      </processingMethod>\n"
        "    </dataProcessing>\n"
        "  </dataProcessingList>\n"
        '  <run id="fixture_run" defaultInstrumentConfigurationRef="IC1">\n'
        f'    <spectrumList count="{len(spectra)}" defaultDataProcessingRef="DP1">\n'
    )
    footer = "    </spectrumList>\n  </run>\n</mzML>\n"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        for s in spectra:
            fh.write(s)
        fh.write(footer)

    n_ms1 = a.cycles
    n_ms2 = a.cycles * a.windows
    print(
        f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB): {len(spectra)} spectra "
        f"({n_ms1} MS1, {n_ms2} MS2), {a.windows} windows over "
        f"{lo:.1f}-{hi:.1f} m/z, {total_seconds:.0f} s gradient, "
        f"{len(planted)} planted target precursors"
    )
    if not a.quiet:
        for p in planted:
            print(
                f"  planted {p['peptidoform']}/{p['charge']} at {p['precursor_mz']:.4f} m/z, "
                f"apex {p['apex_seconds']:.1f} s, {len(p['fragments'])} fragments"
            )

    if a.manifest:
        Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
        with open(a.manifest, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(
                {
                    "options": vars(a),
                    "isolation_windows": [{"lower": w[0], "upper": w[1]} for w in windows],
                    "gradient_seconds": total_seconds,
                    "n_spectra": len(spectra),
                    "planted": planted,
                },
                fh,
                indent=2,
                sort_keys=True,
            )
        print(f"wrote {a.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
