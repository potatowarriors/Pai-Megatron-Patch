FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \\
    bash coreutils findutils grep sed gawk curl wget git jq tree less vim-tiny nano procps \\
    build-essential make unzip zip tar gzip bzip2 xz-utils sqlite3 tmux \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytest==8.4.1 numpy==2.3.3 pyyaml==6.0.2 requests==2.32.4 pandas==2.3.2
WORKDIR /app
