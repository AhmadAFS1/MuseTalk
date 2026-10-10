# syntax=docker/dockerfile:1.7
# Required named context: release (audited release.json, weights.tar.gz, native.tar.gz).
# CUDA_BASE must be an Ubuntu 22.04 CUDA 12.1 devel image pinned by actual digest.
# CUDA_RUNTIME_BASE must explicitly equal the manifest-selected final-stage base.
# No unvalidated native engine, invented digest, or default release is supplied.
ARG CUDA_BASE
ARG CUDA_RUNTIME_BASE
FROM ${CUDA_BASE} AS build
ARG CUDA_BASE
ARG CUDA_RUNTIME_BASE
ARG SOURCE_REVISION
ARG TARGETARCH
ARG BOOTSTRAP_PYTHON_VERSION
ARG RELEASE_CHANNEL=validated
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
WORKDIR /opt/musetalk/app
COPY . /opt/musetalk/app/
COPY --from=release /release.json /opt/musetalk/release.json
RUN test -f docker/musetalk/dependency_inventory.py && test "$TARGETARCH" = amd64 && \
    [[ "$CUDA_BASE" =~ @sha256:[0-9a-f]{64}$ ]] && \
    [[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] && \
    [[ "$BOOTSTRAP_PYTHON_VERSION" =~ ^[A-Za-z0-9.+:~_-]+$ ]] && \
    apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      --no-install-recommends "python3=$BOOTSTRAP_PYTHON_VERSION"
# Exact apt versions, including bootstrap python3, are recorded in the manifest.
# Archive mounts never become an image layer, avoiding a second copy of all weights.
RUN --mount=from=release,source=/,target=/release,ro \
    python3 docker/musetalk/release.py stage --root /opt/musetalk/app \
      --manifest /opt/musetalk/release.json --assets /release \
      --revision "$SOURCE_REVISION" --base "$CUDA_BASE" --runtime-base "$CUDA_RUNTIME_BASE" --bootstrap-python "$BOOTSTRAP_PYTHON_VERSION" \
      --channel "$RELEASE_CHANNEL" && \
    python3 docker/musetalk/release.py apt --manifest /opt/musetalk/release.json
ENV VENV_PATH=/opt/musetalk/venv \
    MUSETALK_INSTALL_STATE_FILE=/opt/musetalk/install_state.json \
    WEBRTC_NATIVE_VP8_DIR=/opt/musetalk/native_vp8 \
    HF_HOME=/opt/musetalk/app/models/hf-cache \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 docker/musetalk/release.py install --root /opt/musetalk/app \
      --manifest /opt/musetalk/release.json && \
    /opt/musetalk/venv/bin/python docker/musetalk/release.py cpu-check \
      --root /opt/musetalk/app --manifest /opt/musetalk/release.json
# A fresh final stage keeps build/install layers out of the shipped image.
# Its explicit base must match the manifest, including the legacy devel-base
# choice. All OS package pins and /opt payloads are retained; no runtime/library
# pruning is performed here. Full GPU/image acceptance is still required.
FROM ${CUDA_RUNTIME_BASE} AS runtime
ARG CUDA_RUNTIME_BASE
ARG SOURCE_REVISION
ARG BOOTSTRAP_PYTHON_VERSION
ARG RELEASE_CHANNEL=validated
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
RUN [[ "$CUDA_RUNTIME_BASE" =~ @sha256:[0-9a-f]{64}$ ]] && \
    [[ "$BOOTSTRAP_PYTHON_VERSION" =~ ^[A-Za-z0-9.+:~_-]+$ ]] && \
    apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      --no-install-recommends "python3=$BOOTSTRAP_PYTHON_VERSION"
COPY --from=build /opt/musetalk /opt/musetalk
WORKDIR /opt/musetalk/app
ENV VENV_PATH=/opt/musetalk/venv \
    MUSETALK_INSTALL_STATE_FILE=/opt/musetalk/install_state.json \
    WEBRTC_NATIVE_VP8_DIR=/opt/musetalk/native_vp8 \
    HF_HOME=/opt/musetalk/app/models/hf-cache \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
RUN python3 docker/musetalk/release.py runtime-base --manifest /opt/musetalk/release.json \
      --base "$CUDA_RUNTIME_BASE" && \
    python3 docker/musetalk/release.py apt --manifest /opt/musetalk/release.json && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    CUDA_VISIBLE_DEVICES="" /opt/musetalk/venv/bin/python docker/musetalk/release.py cpu-check \
      --root /opt/musetalk/app --manifest /opt/musetalk/release.json
LABEL org.opencontainers.image.title="MuseTalk r5 RTX 3090" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      io.musetalk.cuda-runtime-base="${CUDA_RUNTIME_BASE}" \
      io.musetalk.release-channel="${RELEASE_CHANNEL}" \
      org.opencontainers.image.description="Native sm86 r5; acceptance scope is in /opt/musetalk/release.json"
ENV PATH=/opt/musetalk/venv/bin:$PATH \
    MUSETALK_IMAGE_ROOT=/opt/musetalk/app \
    MUSETALK_RELEASE_MANIFEST=/opt/musetalk/release.json \
    MUSETALK_IMMUTABLE_RUNTIME=1 \
    AUTO_SETUP=0 SETUP_CLEAN=0 SETUP_SELFTEST=0 \
    MUSETALK_RECIPE=r5 MUSETALK_VERIFY_RECIPE=strict \
    MUSETALK_UNET_ENGINE_PROVISION=off MUSETALK_TAESD_TRT_PROVISION=off \
    MUSETALK_UNET_STAGEWISE_PROVISION=off MUSETALK_TAESD_TRT_BUILD=0 \
    MUSETALK_TAESD_TRT_STRICT=1 MUSETALK_VP8_FALLBACK=0 \
    LINGUA_WORKER_CALLBACK_REQUIRED=1
EXPOSE 8000/tcp
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "/bin/bash", "/opt/musetalk/app/scripts/vast_docker_onstart.sh"]
CMD ["serve"]
