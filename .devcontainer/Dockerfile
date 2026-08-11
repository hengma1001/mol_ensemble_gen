# Docker Hub's official Python image (vendor-neutral).
FROM python:3

# Non-root user, which the devcontainers base image used to provide for us.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y --no-install-recommends git sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install 'flit>=3.8.0' pre-commit

ENV FLIT_ROOT_INSTALL=1

# Install dependencies only, from pyproject alone, so this layer caches
# independently of the source tree. The stub package directory must match
# [tool.flit.module] name, or flit resolves nothing.
COPY pyproject.toml .
RUN touch README.md \
    && mkdir -p src/mol_ensemble_gen \
    && python -m flit install --only-deps --deps develop \
    && rm -r pyproject.toml README.md src

USER $USERNAME
