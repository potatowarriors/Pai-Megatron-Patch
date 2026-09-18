# alpha-ffmpeg-base:1 — task image base for the ffmpeg agent tasks.
# ubuntu:24.04 on purpose: its ffmpeg is 6.1.1, the same build the verifier was calibrated with
# (debian bookworm ships 5.1, which lacks -display_rotation used by the input generator).
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-numpy fonts-dejavu-core tmux asciinema bash coreutils findutils grep sed gawk jq less procps file \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
