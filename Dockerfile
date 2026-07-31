# LearnM8 application image — thin layer on top of the pinned base.
#
#   docker build -t tonylac77/learnm8:v1.1 --build-arg GIT_SHA=$(git rev-parse --short HEAD) .
#   docker push tonylac77/learnm8:v1.1
#
# Everything heavy lives in learnm8-base, so this rebuild takes seconds and the
# pushed layer is a few MB — the cluster re-pulls only that layer.
#
# Tags are IMMUTABLE. Bump v1.1 -> v1.2 on every rebuild; never overwrite a tag.
# The cluster caches images by tag, and a mutable :latest is what silently split
# the May and June benchmark waves onto different code.

ARG BASE_TAG=cuda12.6-2026-07
FROM tonylac77/learnm8-base:${BASE_TAG}

WORKDIR /app

# Package metadata and source only. tests/ is deliberately copied AFTER the
# install so setuptools' `packages = {find = {}}` discovers just `learnm8` and
# does not install `tests` as a top-level package.
COPY pyproject.toml README.md LICENSE ./
COPY learnm8/ ./learnm8/

# Installed into the `learnm8` env created by environment.yml in the base image.
#
# --no-deps: every runtime dependency is already solved there by conda. Letting
# pip resolve them here would pull generic PyPI wheels over conda's CUDA-linked
# builds (notably matplotlib over matplotlib-base, and the torch/cuml stack),
# which is how a working GPU image quietly becomes a CPU one.
RUN conda run -n learnm8 pip install --no-deps --no-cache-dir .

# Available for in-container smoke tests; not installed as a package.
COPY tests/ ./tests/
COPY .coveragerc ./

# Provenance: which commit produced this image.
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.source="https://github.com/Tonylac77/LearnM8"
ENV LEARNM8_IMAGE_REVISION=${GIT_SHA}

CMD ["/bin/bash"]
