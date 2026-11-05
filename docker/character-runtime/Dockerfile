# syntax=docker/dockerfile:1.7
# Private image: includes private H3 source and private-delivery preparation models.
ARG DEPENDENCY_BASE=ghcr.io/ahmadafs1/musetalk-rtx3090@sha256:7cd567bd77d421565c6813e5372561be31a8b3cab73f307430abd6b95d0067f6
FROM ${DEPENDENCY_BASE} AS build
ARG SOURCE_REVISION
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
USER root
WORKDIR /opt/character-runtime/app
COPY . /opt/character-runtime/app/
COPY --from=inputs /h3 /workspace/minimax-h3-workflows
COPY --from=inputs /preparation-weights /opt/character-runtime/app/
ENV REPO_ROOT=/opt/character-runtime/app WORKSPACE=/workspace \
    CHARACTER_WORKER_RUNTIME_MODE=prebuilt \
    CHARACTER_WORKER_RUNTIME_MANIFEST=/opt/character-runtime/manifest.json \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
RUN test "$(git rev-parse HEAD)" = "$SOURCE_REVISION" && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      openssh-server jq libgl1 libglib2.0-0 libsndfile1 && \
    mkdir -p /run/sshd && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    CHARACTER_WORKER_BUILD_RUNTIME=1 CHARACTER_WORKER_ALLOW_NO_S3=1 \
      bash character_factory/character_worker/install_character_worker.sh --build-runtime && \
    CUDA_VISIBLE_DEVICES="" python3 character_factory/character_worker/runtime_manifest.py record \
      --repo "$REPO_ROOT" --models docker/character-runtime/preparation-models.json && \
    CUDA_VISIBLE_DEVICES="" python3 character_factory/character_worker/runtime_manifest.py verify \
      --repo "$REPO_ROOT" --models docker/character-runtime/preparation-models.json && \
    rm -rf /root/.cache/pip /opt/musetalk/app/.git
LABEL org.opencontainers.image.title="Lingua character FL2VA preparation runtime" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      io.lingua.character.runtime="lingua-character-runtime/1" \
      io.lingua.character.gpu-qualification="required-before-promotion" \
      io.musetalk.release-channel="character-preparation-candidate"
EXPOSE 22 8010
# Vast executes the pinned template command; the default remains fail-closed.
ENTRYPOINT []
CMD ["/bin/bash", "-c", "echo 'Use the pinned character-factory template and runtime manifest' >&2; exit 64"]
