# MuMDIA: a self-contained image bundling the Rust engine plus the Python
# sidecars (mokapot rescoring, MS2PIP fragment intensities, DeepLC retention
# time) so the full high-sensitivity DIA recipe runs with no host setup:
#
#   docker run --rm -v "$PWD:/data" ghcr.io/compomics/mumdia \
#       run --fasta /data/proteome.fasta \
#           --mzml  /data/run.mzML \
#           --out-dir /data/out \
#           --config /opt/mumdia/config.dia.json
#
# The baked /opt/mumdia/config.dia.json wires the sidecars to the in-image conda
# envs and selects the Extended feature set + DIA apex settings.

# ---------- Stage 1: build the Rust binary ----------
FROM rust:1.96-bookworm AS build
# Override any (gitignored, machine-specific) .cargo/config.toml target dir.
ENV CARGO_TARGET_DIR=/build
WORKDIR /src
COPY rust/mumdia ./
RUN cargo build --release --locked --bin mumdia

# ---------- Stage 2: runtime with the sidecar conda envs ----------
FROM mambaorg/micromamba:1.5.10-bookworm-slim
USER root
ENV MAMBA_ROOT_PREFIX=/opt/conda

# git is needed to pip-install DeepLC from its pinned commit.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Two sidecar envs (rescore = mokapot + MS2PIP, deeplc = DeepLC + torch).
COPY env/docker-rescore.yml env/docker-deeplc.yml /tmp/env/
RUN micromamba create -y -n rescore -f /tmp/env/docker-rescore.yml \
    && micromamba create -y -n deeplc -f /tmp/env/docker-deeplc.yml \
    && micromamba clean -a -y \
    && rm -rf /tmp/env

# Engine binary, sidecar workers, and the baked DIA config.
COPY --from=build /build/release/mumdia /usr/local/bin/mumdia
COPY scripts /opt/mumdia/scripts
COPY docker/config.dia.json /opt/mumdia/config.dia.json

# mokapot logistic-regression is the recommended default rescorer.
ENV MUMDIA_RESCORE_MODEL=logreg

WORKDIR /data
ENTRYPOINT ["mumdia"]
CMD ["--help"]
