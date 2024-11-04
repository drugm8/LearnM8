ARG BASE_IMAGE=debian:12-slim

# Mutli-stage build to keep final image small. Otherwise end up with
# curl and openssl installed
FROM $BASE_IMAGE AS stage1
ARG TARGETARCH
ARG VERSION=2.0.2
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      ca-certificates \
      curl \
    && rm -rf /var/lib/apt /var/lib/dpkg /var/lib/cache /var/lib/log
COPY _download_micromamba.sh /usr/local/bin/
RUN _download_micromamba.sh

FROM $BASE_IMAGE

ARG CERT_SOURCE='/etc/ssl/certs/ca-certificates.crt'
ARG MAMBA_USER=mambauser
ARG MAMBA_USER_ID=57439
ARG MAMBA_USER_GID=57439
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_USER_ID=$MAMBA_USER_ID
ENV MAMBA_USER_GID=$MAMBA_USER_GID
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV ENV_NAME="base"
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=stage1 "${MAMBA_EXE}" "${MAMBA_EXE}"
COPY --from=stage1 "${CERT_SOURCE}" "${CERT_SOURCE}"
COPY _dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY _dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh \
    && /usr/local/bin/_dockerfile_setup_root_prefix.sh

USER $MAMBA_USER

WORKDIR /tmp

COPY ../environment.yml .

RUN micromamba install -y -n base -f environment.yml \
    && micromamba clean --all --yes

COPY ../HELLO_WORLD.py .

CMD ["python", "HELLO_WORLD.py"]