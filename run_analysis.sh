#!/usr/bin/env bash
# Full MuMDIA pipeline + analysis for the ecoli splitmult1p5_calibonly run.
#
# Output layout:
#   runs/<RUN_NAME>/
#     pipeline/     MuMDIA pipeline outputs (result_dir)
#     debug/        Debug TSVs, partition PNGs, pickles (via symlink debug/)
#     comparison/
#       vs_diann/                    DIA-NN compare tables
#       split_stage_buckets/         RT-window coverage analysis
#       mirror_xics/                 MuMDIA vs DIA-NN mirror XIC plots
#       suspicious_ms2_mirrors/      MS2 mirror plots for gap peptidoforms
#       margin_ms2pip_q10/           Margin+MS2PIP correlation (q<=10%)
#       margin_ms2pip_q25/           Margin+MS2PIP correlation (q<=25%)
#       similarity_cutoff/           Target count vs similarity cutoff plots
#       qfiltered_q01/               q<=1%  target/decoy export
#       qfiltered_q10/               q<=10% target/decoy export
#       qfiltered_q25/               q<=25% target/decoy export
#       audit_peptidoforms/          Full peptidoform audit bundles
#
# Usage:
#   bash run_analysis.sh            # run everything from scratch
#   SKIP_PIPELINE=1 bash run_analysis.sh   # skip pipeline, only re-run analysis

set -euo pipefail

PYTHON="/home/robbin/anaconda3/envs/py312/bin/python"
CONFIG="configs/config_ecoli_splitmult1p5_calibonly.json"

RUN_NAME="ecoli_splitmult1p5_calibonly"
RUN_DIR="runs/${RUN_NAME}"
PIPELINE_DIR="${RUN_DIR}/pipeline"
DEBUG_DIR="${RUN_DIR}/debug"
COMPARISON_DIR="${RUN_DIR}/comparison"

DIANN_REPORT_FULL="diann_results/report_full.tsv"
DIANN_REPORT_XIC="diann_results/report_with_xic.tsv"
DIANN_XIC_PARQUET="diann_results/report_with_xic_xic/LFQ_Orbitrap_AIF_Ecoli_01.xic.parquet"

RT_MARGIN_DUMP="${PIPELINE_DIR}/rt_margin_input_dump"
STAGE3_DUMP="${PIPELINE_DIR}/stage3_feature_input_dump"
REPLAYED_MARGINS_DIR="${RT_MARGIN_DUMP}/replayed_margins"

# Ensure debug/ symlink always points at current run debug dir.
mkdir -p "${DEBUG_DIR}" "${COMPARISON_DIR}"
ln -sfn "${RUN_DIR}/debug" debug

log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo; echo "════════════════════════════════════════"; echo " $*"; echo "════════════════════════════════════════"; }

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1–3: Main MuMDIA pipeline
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${SKIP_PIPELINE:-0}" != "1" ]]; then
    step "Running MuMDIA pipeline"
    ${PYTHON} run.py "${CONFIG}"
    log "Pipeline complete."
else
    log "SKIP_PIPELINE=1 — skipping pipeline, running analysis only."
fi

# ─────────────────────────────────────────────────────────────────────────────
# XGBoost rescoring
# ─────────────────────────────────────────────────────────────────────────────
step "XGBoost rescoring"
${PYTHON} scripts/run_mokapot_xgboost.py \
    --pin-file "${PIPELINE_DIR}/outfile.pin" \
    --output-dir "${PIPELINE_DIR}" \
    --file-root "xgboost" \
    --write-decoys
log "XGBoost done."

# Use XGBoost mokapot outputs as the canonical mokapot files.
cp "${PIPELINE_DIR}/xgboost.mokapot.psms.txt"          "${PIPELINE_DIR}/mokapot.psms.txt"
cp "${PIPELINE_DIR}/xgboost.mokapot.peptides.txt"      "${PIPELINE_DIR}/mokapot.peptides.txt"
cp "${PIPELINE_DIR}/xgboost.mokapot.decoy.psms.txt"    "${PIPELINE_DIR}/mokapot.decoy.psms.txt"   2>/dev/null || true
cp "${PIPELINE_DIR}/xgboost.mokapot.decoy.peptides.txt" "${PIPELINE_DIR}/mokapot.decoy.peptides.txt" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Replay RT margins (needed for MS2PIP correlation analysis)
# ─────────────────────────────────────────────────────────────────────────────
step "Replaying RT margins"
${PYTHON} scripts/replay_rt_margins.py \
    --dump-dir "${RT_MARGIN_DUMP}" \
    --output-dir "${REPLAYED_MARGINS_DIR}" \
    --use-global-ms2-rt-grid
log "RT margin replay done."

# ─────────────────────────────────────────────────────────────────────────────
# MuMDIA vs DIA-NN comparison
# ─────────────────────────────────────────────────────────────────────────────
step "MuMDIA vs DIA-NN comparison"
${PYTHON} compare_mumdia_diann.py \
    --mumdia-dir "${PIPELINE_DIR}" \
    --diann-report "${DIANN_REPORT_FULL}" \
    --output-dir "${COMPARISON_DIR}/vs_diann"
log "Comparison done."

# ─────────────────────────────────────────────────────────────────────────────
# Split-stage RT-window coverage buckets
# ─────────────────────────────────────────────────────────────────────────────
step "Split-stage missing bucket analysis"
${PYTHON} scripts/analyze_split_stage_missing_buckets.py \
    --config "${CONFIG}" \
    --split-psms "${PIPELINE_DIR}/df_psms.tsv" \
    --diann-report "${DIANN_REPORT_FULL}" \
    --output-dir "${COMPARISON_DIR}/split_stage_buckets"
log "Bucket analysis done."

# ─────────────────────────────────────────────────────────────────────────────
# Mirror XIC plots (MuMDIA vs DIA-NN)
# ─────────────────────────────────────────────────────────────────────────────
step "Mirror XIC plots"
${PYTHON} scripts/plot_mirror_xics.py \
    --merged-features "${COMPARISON_DIR}/vs_diann/merged_features.tsv" \
    --gap-peptides    "${COMPARISON_DIR}/vs_diann/gap_peptides.tsv" \
    --df-fragment     "${PIPELINE_DIR}/df_fragment.tsv" \
    --diann-report    "${DIANN_REPORT_XIC}" \
    --diann-xic       "${DIANN_XIC_PARQUET}" \
    --output-dir      "${COMPARISON_DIR}/mirror_xics"
log "Mirror XIC plots done."

SUSPICIOUS_TSV="${COMPARISON_DIR}/mirror_xics/mirror_candidates_suspicious.tsv"

# ─────────────────────────────────────────────────────────────────────────────
# Suspicious MS2 mirror plots
# ─────────────────────────────────────────────────────────────────────────────
step "Suspicious MS2 mirror plots"
${PYTHON} scripts/plot_suspicious_ms2_mirrors.py \
    --suspicious-tsv  "${SUSPICIOUS_TSV}" \
    --dump-dir        "${STAGE3_DUMP}" \
    --output-dir      "${COMPARISON_DIR}/suspicious_ms2_mirrors"
log "Suspicious MS2 mirrors done."

# ─────────────────────────────────────────────────────────────────────────────
# Export q-filtered targets + decoys at three thresholds
# ─────────────────────────────────────────────────────────────────────────────
step "Exporting q-filtered targets/decoys"
for Q in 0.01 0.10 0.25; do
    QSUFFIX=$(echo "$Q" | tr -d '.')
    OUT_PATH="${COMPARISON_DIR}/qfiltered_q${QSUFFIX}/qfiltered_q${QSUFFIX}.tsv"
    mkdir -p "$(dirname "${OUT_PATH}")"
    ${PYTHON} scripts/export_qfiltered_targets_decoys.py \
        --target-psms   "${PIPELINE_DIR}/xgboost.mokapot.psms.txt" \
        --decoy-psms    "${PIPELINE_DIR}/xgboost.mokapot.decoy.psms.txt" \
        --pin-file      "${PIPELINE_DIR}/outfile.pin" \
        --q-threshold   "${Q}" \
        --output-path   "${OUT_PATH}"
    log "  q<=${Q} export done."
done

# ─────────────────────────────────────────────────────────────────────────────
# Margin + MS2PIP correlation analysis at q10 and q25
# ─────────────────────────────────────────────────────────────────────────────
REPLAYED_FRAGMENT_TABLE="${REPLAYED_MARGINS_DIR}/df_fragment_with_replayed_margins.tsv"
CONTEXT_PICKLE="${STAGE3_DUMP}/theoretical_fragment_context.pkl"

step "Margin + MS2PIP correlation analysis (q<=10%)"
${PYTHON} scripts/analyze_margin_summed_ms2pip_correlations.py \
    --fragment-table  "${REPLAYED_FRAGMENT_TABLE}" \
    --context-pickle  "${CONTEXT_PICKLE}" \
    --suspicious-tsv  "${SUSPICIOUS_TSV}" \
    --candidate-tsv   "${COMPARISON_DIR}/qfiltered_q010/qfiltered_q010.tsv" \
    --output-dir      "${COMPARISON_DIR}/margin_ms2pip_q10"
log "q<=10% margin analysis done."

step "Margin + MS2PIP correlation analysis (q<=25%)"
${PYTHON} scripts/analyze_margin_summed_ms2pip_correlations.py \
    --fragment-table  "${REPLAYED_FRAGMENT_TABLE}" \
    --context-pickle  "${CONTEXT_PICKLE}" \
    --suspicious-tsv  "${SUSPICIOUS_TSV}" \
    --candidate-tsv   "${COMPARISON_DIR}/qfiltered_q025/qfiltered_q025.tsv" \
    --output-dir      "${COMPARISON_DIR}/margin_ms2pip_q25"
log "q<=25% margin analysis done."

# ─────────────────────────────────────────────────────────────────────────────
# Similarity cutoff comparison plot
# ─────────────────────────────────────────────────────────────────────────────
step "Similarity cutoff comparison plot"
${PYTHON} scripts/plot_target_count_cutoff_comparison.py \
    --q10-table  "${COMPARISON_DIR}/margin_ms2pip_q10/margin_summed_ms2pip_correlations.tsv" \
    --q25-table  "${COMPARISON_DIR}/margin_ms2pip_q25/margin_summed_ms2pip_correlations.tsv" \
    --output-dir "${COMPARISON_DIR}/similarity_cutoff"
log "Similarity cutoff plot done."

# ─────────────────────────────────────────────────────────────────────────────
# Full peptidoform audit bundles
# ─────────────────────────────────────────────────────────────────────────────
step "Peptidoform audit bundles"
${PYTHON} scripts/audit_peptidoforms.py \
    --results-dir     "${PIPELINE_DIR}" \
    --debug-dir       "${DEBUG_DIR}" \
    --comparison-dir  "${COMPARISON_DIR}" \
    --output-dir      "${COMPARISON_DIR}/audit_peptidoforms"
log "Audit bundles done."

# ─────────────────────────────────────────────────────────────────────────────
# Apex RT comparison: MuMDIA vs DIA-NN (all peptidoforms incl. below FDR)
# ─────────────────────────────────────────────────────────────────────────────
step "Apex comparison (MuMDIA vs DIA-NN)"
${PYTHON} scripts/analyze_apex_comparison.py \
    --mumdia-apex  "${DEBUG_DIR}/df_fragment_max_peptide_after_ms2pip.tsv" \
    --mumdia-psms  "${DEBUG_DIR}/df_psms_after_ms2pip.tsv" \
    --mokapot      "${PIPELINE_DIR}/xgboost.mokapot.psms.txt" \
    --diann        "diann_results/report_with_xic.tsv" \
    --outdir       "${COMPARISON_DIR}/apex_comparison"
log "Apex comparison done."

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
step "Done"
log "All outputs under: ${RUN_DIR}/"
du -sh "${PIPELINE_DIR}" "${DEBUG_DIR}" "${COMPARISON_DIR}"
