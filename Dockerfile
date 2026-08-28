# MuMDIA: a self-contained image bundling the Rust engine plus the Python
# sidecars (mokapot rescoring, MS2PIP fragment intensities, DeepLC retention
# time) so the full high-sensitivity DIA recipe runs with no host setup:
#
#   docker run --rm -v "$PWD:/data" --user "$(id -u):$(id -g)" \
#       ghcr.io/compomics/mumdia \
#       run --fasta /data/proteome.fasta \
#           --mzml  /data/run.mzML \
#           --out-dir /data/out \
#           --config /opt/mumdia/config.dia.json
#
# --user is REQUIRED whenever the engine writes into a bind mount: the image runs
# as an unprivileged user whose uid does not match yours, so without it even the
# mount point fails with `mkdir: cannot create directory '/data': Permission
# denied` (measured 2026-08-27). With it, the results are owned by you.
# Check the sidecar environments before a long run:
#   docker run --rm ghcr.io/compomics/mumdia \
#       doctor --config /opt/mumdia/config.dia.json
#
# The baked /opt/mumdia/config.dia.json wires the FASTA workflow to the in-image
# conda envs. /opt/mumdia/config.diann-lib.json selects imported-library
# fine-tuning plus the torch rescorer.

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

# Two sidecar envs (rescore = mokapot + MS2PIP, deeplc = DeepLC + torch).
#
# build-essential is installed for the pip step and PURGED in the same layer. It lets
# pip compile a sidecar dependency that ships only as an sdist, and it has no business
# surviving into a runtime image whose only job is one static binary plus two prebuilt
# conda envs: it shipped gcc, make and the C headers to every user, which is both a
# large surface and a post-exploitation convenience. Same layer, so the bytes never
# reach the published image rather than being deleted from a later one.
# git is not required: DeepLC is pinned to a PyPI version, not a repository commit
# (env/docker-deeplc.yml).
COPY env/docker-rescore.yml env/docker-deeplc.yml /tmp/env/
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && micromamba create -y -n rescore -f /tmp/env/docker-rescore.yml \
    && micromamba create -y -n deeplc -f /tmp/env/docker-deeplc.yml \
    && micromamba clean -a -y \
    && rm -rf /tmp/env \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Engine binary, sidecar workers, and the baked FASTA/library DIA configs.
COPY --from=build /build/release/mumdia /usr/local/bin/mumdia
COPY scripts /opt/mumdia/scripts
COPY docker/config.dia.json /opt/mumdia/config.dia.json
COPY docker/config.diann-lib.json /opt/mumdia/config.diann-lib.json
# Notices for the statically linked crates, as in the release archive.
COPY LICENSE THIRD_PARTY_LICENSES.md /opt/mumdia/

# mokapot logistic-regression is the recommended default rescorer.
ENV MUMDIA_RESCORE_MODEL=logreg

# Standard OCI metadata, so the published image says what it is and links back to
# the source. The version label is filled from the tag by the build workflow.
LABEL org.opencontainers.image.title="MuMDIA" \
      org.opencontainers.image.description="DIA proteomics search engine with bundled Python sidecars" \
      org.opencontainers.image.source="https://github.com/CompOmics/MuMDIA" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="CompOmics (Ghent University / VIB)"

WORKDIR /data

# Drop back to the base image's unprivileged user. Root was needed only to install
# apt packages and to create the conda envs under /opt/conda, which are read-only
# at run time.
#
# The container user's uid is assigned by the base image and will not match your
# host uid, so pass `--user "$(id -u):$(id -g)"` (as the usage example above does)
# whenever you bind-mount a directory the engine has to write to. Everything
# MuMDIA writes lands under --out-dir inside that mount, so no path in the image
# needs to be writable at run time.
USER $MAMBA_USER
ENTRYPOINT ["mumdia"]
CMD ["--help"]
