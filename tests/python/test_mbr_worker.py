"""Contract tests for `scripts/mbr_worker.py` (Stage D3, rescuable tier).

This worker needs only numpy and pyarrow, so unlike the ML sidecars it always
runs here rather than skipping. A nonzero exit aborts `mumdia mbr` outright
(`docs/13_sidecars.md`, "Failure behavior": MBR has no strict gate and no
fallback), and a silent contract break wastes an experiment-wide run.

The headline case is `test_m5_lowers_every_psm_q_column`. Lowering only
`q_value` left 34,280 of 34,664 accepted transfers unquantified on the HYE
pooled run, because quant gates on `run_psm_q` rather than `q_value`
(`mbr_worker.py:269-274`). Every PSM-level q column that `quant.q_filter` can
select must be lowered on the accepted row, or the transfers are accepted and
then thrown away.

Dataset layout (built once per session):

* 600 "good" transfer candidates and 30 "bad" ones, ids 0..629, all targets.
* Four runs. Candidates are confident (`q_value = 0.001 <= --q-anchor`) in runs
  1 and 2, sub-threshold in run 0 where they were still extracted, so run 0 is
  the only run with transfer candidates.
* Good candidates sit within 0.07 s of their cross-run predicted RT; bad ones
  are ~500 s away, far outside anything the permuted-RT null supports.
* Run 3 is the control for row keying: the same `candidate_id` values carry
  high-q scored rows there, but run 3's psms table does not contain them, so
  they are never transfer candidates. A worker keying the augmentation on
  `candidate_id` alone instead of `(candidate_id, source)` would lower these.
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from conftest import (
    read_columns,
    run_worker,
    run_worker_ok,
    write_psms_table,
    write_scored_table,
)

N_GOOD = 600
N_BAD = 30
N_RUNS = 4
# Present only in run 3's psms table and absent from the scored table, so run 3
# has an extracted-candidate map that shares nothing with the transfer set.
DUMMY_CID = 1_000_000

GOOD_IDS = list(range(N_GOOD))
BAD_IDS = list(range(N_GOOD, N_GOOD + N_BAD))
ALL_IDS = GOOD_IDS + BAD_IDS

# Source-0 q values, all above --q-anchor so the candidate is "rescuable".
Q_SUB = 0.5
RUN_Q_GOOD = 0.6
EXP_Q_GOOD = 0.7
# The bad candidates carry a pre-existing zero in the two secondary q columns:
# the augmentation is a minimum, not an assignment, so these must never rise.
RUN_Q_BAD = 0.0
EXP_Q_BAD = 0.0
# Source-3 q values (the control rows that must stay untouched).
Q_OTHER = 0.9
RUN_Q_OTHER = 0.91
EXP_Q_OTHER = 0.92


def _base_rt(cid):
    """Reference retention time of a candidate, 5 s apart so a permuted-RT
    decoy transfer can never land within the good candidates' RT residual."""
    if cid < N_GOOD:
        return 200.0 + 5.0 * cid
    return 3300.0 + 5.0 * (cid - N_GOOD)


def _observed_rt_run0(cid):
    if cid < N_GOOD:
        return _base_rt(cid) + 0.01 + 0.0001 * cid
    return _base_rt(cid) + 500.0 + 0.1 * (cid - N_GOOD)


def _build_scored_rows():
    cols = {k: [] for k in ("candidate_id", "source", "label", "q_value",
                            "peptidoform", "charge", "protein_group")}
    run_q = []
    exp_q = []
    for cid in ALL_IDS:
        for src in range(N_RUNS):
            cols["candidate_id"].append(cid)
            cols["source"].append(src)
            cols["label"].append("target")
            cols["peptidoform"].append("PEP{}K".format(cid))
            cols["charge"].append(2 + (cid % 2))
            cols["protein_group"].append("PG{}".format(cid % 11))
            if src == 0:
                cols["q_value"].append(Q_SUB)
                run_q.append(RUN_Q_GOOD if cid < N_GOOD else RUN_Q_BAD)
                exp_q.append(EXP_Q_GOOD if cid < N_GOOD else EXP_Q_BAD)
            elif src == 3:
                cols["q_value"].append(Q_OTHER)
                run_q.append(RUN_Q_OTHER)
                exp_q.append(EXP_Q_OTHER)
            else:
                cols["q_value"].append(0.001)
                run_q.append(0.002)
                exp_q.append(0.003)
    return cols, {"run_psm_q": run_q, "experiment_psm_q": exp_q}


@pytest.fixture(scope="session")
def mbr_dataset(tmp_path_factory):
    """Write the scored table and the four per-run psms tables once."""
    d = tmp_path_factory.mktemp("mbr_dataset")
    rows, extra_q = _build_scored_rows()
    scored = write_scored_table(d / "scored_combined.parquet", rows, extra_q=extra_q)
    psms = []
    # run 0: everything extracted, at the observed RT under test.
    psms.append(write_psms_table(
        d / "psms_0.parquet", ALL_IDS, [_observed_rt_run0(c) for c in ALL_IDS]))
    # runs 1 and 2: the confident anchors, straddling the reference RT so their
    # median is exactly the reference.
    psms.append(write_psms_table(
        d / "psms_1.parquet", ALL_IDS, [_base_rt(c) + 0.02 for c in ALL_IDS]))
    psms.append(write_psms_table(
        d / "psms_2.parquet", ALL_IDS, [_base_rt(c) - 0.02 for c in ALL_IDS]))
    # run 3: an extracted set that contains none of the transfer candidates.
    psms.append(write_psms_table(d / "psms_3.parquet", [DUMMY_CID], [100.0]))
    return {
        "dir": d,
        "scored": scored,
        "psms_csv": ",".join(str(p) for p in psms),
        "n_scored_rows": len(rows["candidate_id"]),
        "scored_in": read_columns(scored),
    }


def _run_mbr(dataset, out_dir, q_transfer, seed=7, out_scored=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    transferred = out_dir / "transferred.parquet"
    scored_out = out_dir / "scored_transferred.parquet"
    argv = [
        dataset["scored"],
        dataset["psms_csv"],
        transferred,
        "--q-anchor", 0.01,
        "--min-anchor-runs", 2,
        "--q-transfer", q_transfer,
        "--seed", seed,
    ]
    if out_scored:
        argv += ["--out-scored", scored_out]
    stdout, _ = run_worker_ok("mbr_worker.py", *argv)
    return {
        "transferred": transferred,
        "scored_out": scored_out if out_scored else None,
        "stdout": stdout,
    }


@pytest.fixture(scope="session")
def mbr_result(mbr_dataset, tmp_path_factory):
    """One worker invocation at `--q-transfer 0.05`, shared by several tests."""
    out_dir = tmp_path_factory.mktemp("mbr_out_005")
    return _run_mbr(mbr_dataset, out_dir, 0.05)


def _accepted_pairs(transferred_path):
    t = read_columns(transferred_path)
    return set(zip((int(c) for c in t["candidate_id"]), (int(s) for s in t["source"])))


# ---------------------------------------------------------------------------
# the permuted-RT decoy null and the transfer q threshold
# ---------------------------------------------------------------------------


def test_accepted_transfers_respect_q_transfer(mbr_result):
    """Every accepted transfer must carry `transfer_q <= --q-transfer`.

    If it does not, MBR is reporting transfers at an FDR the caller never asked
    for: the accepted set is what `--out-scored` promotes into quant, so a
    threshold that is not enforced here inflates the identification list with
    false transfers that no downstream stage re-checks.
    """
    t = read_columns(mbr_result["transferred"])
    q = np.asarray(t["transfer_q"], dtype=float)
    assert q.size > 0, "no transfer was accepted; the null rejected everything"
    assert np.isfinite(q).all()
    assert (q <= 0.05).all(), "accepted a transfer above --q-transfer"


def test_permuted_rt_null_separates_concordant_from_discordant_transfers(mbr_result):
    """The RT-concordant set is accepted and the ~500 s discordant set is not.

    The permuted-RT decoy null is the only thing standing between MBR and
    transferring an identification onto whatever happened to be extracted at
    the wrong retention time. If discordant candidates pass, MBR turns into an
    unvalidated ID amplifier.
    """
    accepted = _accepted_pairs(mbr_result["transferred"])
    assert accepted == {(c, 0) for c in GOOD_IDS}, (
        "the accepted set is not exactly the RT-concordant candidates in run 0"
    )


def test_transferred_row_count_matches_accepted_transfers(mbr_result):
    """`transferred.parquet` holds exactly one row per accepted transfer.

    The row count is what the caller reports and what `--out-scored` flags. If
    the two disagree, the transfer count in the log and the rows quant can
    actually see diverge, which is how an MBR regression hides.
    """
    n_rows = pq.read_table(str(mbr_result["transferred"])).num_rows
    flags = read_columns(mbr_result["scored_out"])["is_transferred"]
    assert n_rows == len(GOOD_IDS)
    assert n_rows == int(np.asarray(flags, dtype=bool).sum())


def test_transferred_table_carries_the_documented_columns(mbr_result):
    """The output schema the Rust caller and the report stage read back.

    A renamed or dropped column is not caught by an exit code; the readback
    keys on `candidate_id` (`docs/13_sidecars.md`, "Argv contract") and a
    missing RT column silently removes the only evidence a transfer is real.
    """
    t = read_columns(mbr_result["transferred"])
    for col in ("candidate_id", "source", "peptidoform", "charge", "protein_group",
                "label", "expected_rt", "observed_rt", "rt_delta", "transfer_q"):
        assert col in t, "transferred.parquet is missing {}".format(col)
    exp = np.asarray(t["expected_rt"], dtype=float)
    obs = np.asarray(t["observed_rt"], dtype=float)
    delta = np.asarray(t["rt_delta"], dtype=float)
    assert np.allclose(delta, np.abs(obs - exp), atol=1e-9)
    assert (delta < 1.0).all(), "an accepted transfer is not RT-concordant"
    assert set(t["label"]) == {"target"}


def test_worker_is_deterministic_for_a_fixed_seed(mbr_dataset, tmp_path):
    """Two runs at the same `--seed` must produce the same transfers.

    `--seed` receives the engine-wide `rng_seed` (`main.rs:709`) and drives the
    permuted-RT null. If the null is not reproducible, neither is the accepted
    transfer set, so an MBR-enabled experiment cannot be re-run to the same
    numbers and no A/B comparison against it means anything.
    """
    a = _run_mbr(mbr_dataset, tmp_path / "a", 0.05, seed=11, out_scored=False)
    b = _run_mbr(mbr_dataset, tmp_path / "b", 0.05, seed=11, out_scored=False)
    ta = read_columns(a["transferred"])
    tb = read_columns(b["transferred"])
    assert list(ta) == list(tb)
    for col in ta:
        if isinstance(ta[col], list):
            assert ta[col] == tb[col]
        else:
            assert np.array_equal(np.asarray(ta[col]), np.asarray(tb[col]))


def test_looser_q_transfer_only_grows_the_accepted_set(mbr_dataset, tmp_path):
    """The accepted set must be monotone in `--q-transfer`.

    q is a running minimum from the tail (`mbr_worker.py:198`), so a looser
    threshold can only add transfers. A non-monotone accepted set means the q
    mapping back to row order (`mbr_worker.py:199`) is misaligned, which would
    attach one candidate's q to another candidate's row.
    """
    tight = _run_mbr(mbr_dataset, tmp_path / "tight", 0.001, seed=7, out_scored=False)
    loose = _run_mbr(mbr_dataset, tmp_path / "loose", 0.5, seed=7, out_scored=False)
    tight_set = _accepted_pairs(tight["transferred"])
    loose_set = _accepted_pairs(loose["transferred"])
    assert tight_set <= loose_set
    assert len(loose_set) > len(GOOD_IDS), (
        "loosening --q-transfer to 0.5 admitted no additional transfer"
    )


# ---------------------------------------------------------------------------
# M5: the augmented scored table (the regression this suite exists for)
# ---------------------------------------------------------------------------


def test_m5_lowers_every_psm_q_column(mbr_dataset, mbr_result):
    """`q_value`, `run_psm_q` and `experiment_psm_q` must all be lowered.

    This is the 2026-08-26 HYE regression: with only `q_value` lowered, quant
    gated on `run_psm_q` and 34,280 of 34,664 accepted transfers produced no
    quantity at all. Every PSM-level q column `quant.q_filter` can select has
    to move, or MBR accepts transfers that the pipeline then discards.
    """
    before = mbr_dataset["scored_in"]
    after = read_columns(mbr_result["scored_out"])
    t = read_columns(mbr_result["transferred"])

    tq = {(int(c), int(s)): float(q) for c, s, q in
          zip(t["candidate_id"], t["source"], t["transfer_q"])}
    key = list(zip((int(c) for c in before["candidate_id"]),
                   (int(s) for s in before["source"])))
    accepted = np.array([k in tq for k in key])
    assert accepted.sum() == len(GOOD_IDS)

    lowered_to = np.array([tq.get(k, np.inf) for k in key], dtype=float)
    for col in ("q_value", "run_psm_q", "experiment_psm_q"):
        assert col in after, "the augmented table dropped {}".format(col)
        expect = np.minimum(np.asarray(before[col], dtype=float), lowered_to)
        got = np.asarray(after[col], dtype=float)
        assert np.allclose(got, expect, rtol=0, atol=1e-12), (
            "{} is not min(original, transfer_q) on every row".format(col)
        )

    # The accepted rows started at 0.5 / 0.6 / 0.7, so all three must have
    # strictly dropped; equality would mean the column was never touched.
    for col, original in (("q_value", Q_SUB),
                          ("run_psm_q", RUN_Q_GOOD),
                          ("experiment_psm_q", EXP_Q_GOOD)):
        got = np.asarray(after[col], dtype=float)[accepted]
        assert (got < original).all(), (
            "{} was not lowered on the accepted transfer rows".format(col)
        )


def test_m5_sets_is_transferred_on_exactly_the_accepted_rows(mbr_dataset, mbr_result):
    """`is_transferred` must flag the accepted rows and nothing else.

    It is the only marker distinguishing a transferred identification from a
    directly scored one. Over-flagging inflates the reported transfer count;
    under-flagging makes the transfers invisible to any downstream filter.
    """
    before = mbr_dataset["scored_in"]
    after = read_columns(mbr_result["scored_out"])
    accepted = _accepted_pairs(mbr_result["transferred"])
    key = list(zip((int(c) for c in before["candidate_id"]),
                   (int(s) for s in before["source"])))
    expect = np.array([k in accepted for k in key])
    got = np.asarray(after["is_transferred"], dtype=bool)
    assert np.array_equal(got, expect)


def test_m5_touches_only_the_matching_candidate_id_and_source(mbr_dataset, mbr_result):
    """Only the `(candidate_id, source)` row of an accepted transfer may change.

    `candidate_id` is the library index and repeats across runs. Keying the
    augmentation on `candidate_id` alone would promote the same precursor in
    every run of the experiment, including runs where it was never extracted,
    which manufactures identifications out of nothing. Run 3 here holds exactly
    those rows: same ids, high q, not extracted.
    """
    before = mbr_dataset["scored_in"]
    after = read_columns(mbr_result["scored_out"])
    accepted = _accepted_pairs(mbr_result["transferred"])
    key = list(zip((int(c) for c in before["candidate_id"]),
                   (int(s) for s in before["source"])))
    untouched = np.array([k not in accepted for k in key])
    assert untouched.sum() == mbr_dataset["n_scored_rows"] - len(GOOD_IDS)

    for col in ("q_value", "run_psm_q", "experiment_psm_q"):
        assert np.allclose(
            np.asarray(after[col], dtype=float)[untouched],
            np.asarray(before[col], dtype=float)[untouched],
            rtol=0, atol=1e-12,
        ), "{} changed on a row that was not an accepted transfer".format(col)
    assert not np.asarray(after["is_transferred"], dtype=bool)[untouched].any()

    # The run-3 control rows specifically: same candidate_id, different source.
    src = np.asarray(before["source"], dtype=int)
    run3 = src == 3
    assert run3.sum() == len(ALL_IDS)
    assert np.allclose(np.asarray(after["q_value"], dtype=float)[run3], Q_OTHER)
    assert np.allclose(np.asarray(after["run_psm_q"], dtype=float)[run3], RUN_Q_OTHER)
    assert np.allclose(
        np.asarray(after["experiment_psm_q"], dtype=float)[run3], EXP_Q_OTHER
    )


def test_m5_never_raises_a_q_that_is_already_lower(mbr_dataset, tmp_path):
    """The augmentation is a minimum, never an assignment.

    Run at `--q-transfer 0.5` so the RT-discordant candidates are accepted too;
    their `run_psm_q` / `experiment_psm_q` start at 0.0. Writing `transfer_q`
    unconditionally would raise those to ~0.28 and drop rows that a
    `run_psm_q`-gated quant had legitimately accepted, i.e. MBR would lose
    identifications it was supposed to add.
    """
    result = _run_mbr(mbr_dataset, tmp_path / "loose_out", 0.5, seed=7)
    before = mbr_dataset["scored_in"]
    after = read_columns(result["scored_out"])
    t = read_columns(result["transferred"])

    tq = {(int(c), int(s)): float(q) for c, s, q in
          zip(t["candidate_id"], t["source"], t["transfer_q"])}
    bad_rows = [(c, 0) for c in BAD_IDS]
    assert all(k in tq for k in bad_rows), (
        "the RT-discordant candidates were not accepted at --q-transfer 0.5, "
        "so this test cannot exercise the never-raise branch"
    )
    bad_tq = np.array([tq[k] for k in bad_rows])
    assert (bad_tq > 0.0).all(), "transfer_q is zero; assignment and minimum agree"

    key = list(zip((int(c) for c in before["candidate_id"]),
                   (int(s) for s in before["source"])))
    idx = {k: i for i, k in enumerate(key)}
    for k in bad_rows:
        i = idx[k]
        assert float(after["run_psm_q"][i]) == pytest.approx(RUN_Q_BAD, abs=1e-12)
        assert float(after["experiment_psm_q"][i]) == pytest.approx(EXP_Q_BAD, abs=1e-12)
        # ... while q_value, which started above transfer_q, did move.
        assert float(after["q_value"][i]) == pytest.approx(tq[k], abs=1e-12)
        assert bool(after["is_transferred"][i])


@pytest.mark.parametrize("present", [("q_value",), ("q_value", "run_psm_q")])
def test_m5_works_on_a_table_missing_a_q_column(mbr_dataset, tmp_path, present):
    """A scored table without every q column must still be augmented.

    Artifacts written by older engine versions, and the split tables of an
    experiment-wide rescore, do not all carry `run_psm_q` and
    `experiment_psm_q`. The worker lowers only the columns that exist; if it
    instead assumed all three, `mumdia mbr` would abort on a table it should
    have handled, and it must not invent the absent column either.
    """
    rows, extra_q = _build_scored_rows()
    keep = {k: v for k, v in extra_q.items() if k in present}
    scored = write_scored_table(tmp_path / "scored_subset.parquet", rows, extra_q=keep)
    dataset = dict(mbr_dataset)
    dataset["scored"] = scored
    result = _run_mbr(dataset, tmp_path / "out", 0.05)

    after = read_columns(result["scored_out"])
    before = read_columns(scored)
    t = read_columns(result["transferred"])
    tq = {(int(c), int(s)): float(q) for c, s, q in
          zip(t["candidate_id"], t["source"], t["transfer_q"])}
    key = list(zip((int(c) for c in before["candidate_id"]),
                   (int(s) for s in before["source"])))
    lowered_to = np.array([tq.get(k, np.inf) for k in key], dtype=float)

    for col in ("q_value",) + tuple(present):
        expect = np.minimum(np.asarray(before[col], dtype=float), lowered_to)
        assert np.allclose(np.asarray(after[col], dtype=float), expect,
                           rtol=0, atol=1e-12)
    for col in ("run_psm_q", "experiment_psm_q"):
        if col not in present:
            assert col not in after, (
                "the worker invented a {} column that the input did not have".format(col)
            )
    assert int(np.asarray(after["is_transferred"], dtype=bool).sum()) == len(GOOD_IDS)


# ---------------------------------------------------------------------------
# cross-run RT calibration and the empty case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_anchors,calibrated", [(250, True), (150, False)])
def test_binned_map_removes_a_systematic_inter_run_rt_offset(
    tmp_path, n_anchors, calibrated
):
    """`binned_map` must align other runs onto the reference run's RT axis.

    Runs 1 and 2 here are shifted +300 s from run 0. With enough shared anchors
    the predicted RT lands on the observed apex; below the 200-anchor floor
    (`mbr_worker.py:104,108`) the map silently falls back to the identity and
    every prediction is off by the full inter-run offset. That fallback is the
    production trap: a small experiment gets no calibration and MBR then
    rejects every real transfer while looking like it simply found nothing.
    """
    offset = 300.0
    anchors = list(range(n_anchors))
    transfers = list(range(n_anchors, n_anchors + 40))
    base = {c: 100.0 + 10.0 * c for c in anchors}
    base.update({c: 300.0 + 50.0 * (c - n_anchors) for c in transfers})

    cols = {k: [] for k in ("candidate_id", "source", "label", "q_value",
                            "peptidoform", "charge", "protein_group")}
    for cid in anchors + transfers:
        for src in range(3):
            confident = cid in anchors or src != 0
            cols["candidate_id"].append(cid)
            cols["source"].append(src)
            cols["label"].append("target")
            cols["q_value"].append(0.001 if confident else 0.5)
            cols["peptidoform"].append("PEP{}K".format(cid))
            cols["charge"].append(2)
            cols["protein_group"].append("PG0")
    scored = write_scored_table(tmp_path / "scored.parquet", cols)

    every = anchors + transfers
    psms = [
        write_psms_table(tmp_path / "p0.parquet", every, [base[c] for c in every]),
        write_psms_table(tmp_path / "p1.parquet", every,
                         [base[c] + offset for c in every]),
        write_psms_table(tmp_path / "p2.parquet", every,
                         [base[c] + offset for c in every]),
    ]
    out = tmp_path / "transferred.parquet"
    run_worker_ok(
        "mbr_worker.py", scored, ",".join(str(p) for p in psms), out,
        "--q-anchor", 0.01, "--min-anchor-runs", 2, "--q-transfer", 1.0, "--seed", 3,
    )

    t = read_columns(out)
    assert set(int(c) for c in t["candidate_id"]) == set(transfers)
    residual = np.abs(np.asarray(t["expected_rt"], dtype=float)
                      - np.asarray(t["observed_rt"], dtype=float))
    if calibrated:
        assert residual.max() < 1.0, (
            "with {} shared anchors the binned-median map should remove the "
            "{} s offset".format(n_anchors, offset)
        )
    else:
        assert np.allclose(residual, offset, atol=1.0), (
            "below the 200-anchor floor the map must be the identity, leaving "
            "the full inter-run offset in the prediction"
        )


def test_no_transfer_candidates_writes_an_empty_table_and_no_scored_table(tmp_path):
    """With nothing to transfer the worker exits 0, and `--out-scored` is skipped.

    `mbr_worker.py:187-188` returns before the M5 block, so a caller that
    passed `--out-scored` gets no file. A downstream quant pointed at that path
    fails on a missing input rather than on a nonzero MBR exit, so the true
    cause is not in the MBR log; pinning the behaviour keeps that documented.
    """
    ids = list(range(20))
    cols = {k: [] for k in ("candidate_id", "source", "label", "q_value",
                            "peptidoform", "charge", "protein_group")}
    for cid in ids:
        for src in range(2):
            cols["candidate_id"].append(cid)
            cols["source"].append(src)
            cols["label"].append("target")
            cols["q_value"].append(0.001)
            cols["peptidoform"].append("PEP{}K".format(cid))
            cols["charge"].append(2)
            cols["protein_group"].append("PG0")
    scored = write_scored_table(tmp_path / "scored.parquet", cols)
    psms = [
        write_psms_table(tmp_path / "p0.parquet", ids, [100.0 + c for c in ids]),
        write_psms_table(tmp_path / "p1.parquet", ids, [100.0 + c for c in ids]),
    ]
    out = tmp_path / "transferred.parquet"
    scored_out = tmp_path / "scored_out.parquet"
    stdout, _ = run_worker_ok(
        "mbr_worker.py", scored, ",".join(str(p) for p in psms), out,
        "--q-anchor", 0.01, "--min-anchor-runs", 2, "--q-transfer", 0.01, "--seed", 0,
        "--out-scored", scored_out,
    )
    assert "no transfer candidates" in stdout
    assert pq.read_table(str(out)).num_rows == 0
    assert not scored_out.exists()


def test_missing_psms_path_fails_loudly(mbr_dataset, tmp_path):
    """A bad `<psms_csv>` entry must exit nonzero, not produce a partial result.

    MBR has no fallback: `run_mbr(...)?` propagates and aborts the command
    (`main.rs:697-710`). A worker that swallowed a missing per-run psms table
    would emit a transfer set built from a subset of the runs, which silently
    changes the anchor support behind `--min-anchor-runs`.
    """
    bad = ",".join([str(tmp_path / "does_not_exist.parquet")]
                   + mbr_dataset["psms_csv"].split(",")[1:])
    rc, _, err = run_worker(
        "mbr_worker.py", mbr_dataset["scored"], bad, tmp_path / "out.parquet",
        "--q-anchor", 0.01, "--min-anchor-runs", 2, "--q-transfer", 0.05, "--seed", 0,
    )
    assert rc != 0
    assert "does_not_exist" in err or "No such file" in err or "FileNotFound" in err
