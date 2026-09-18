#!/bin/bash
# tunnel.sh — expose a local OpenAI-compatible endpoint inside the gpu06 eval container.
# The Harbor process runs in the container, so "localhost:<RPORT>" there must reach the model here.
# Run by a person, in its own terminal (it stays in the foreground). 8199 belongs to the SFT eval fleet.
#
#   bash harbor/tunnel.sh [LOCAL_PORT=8000] [REMOTE_PORT=8299]
set -euo pipefail
LPORT="${1:-8000}"; RPORT="${2:-8299}"
SSHC="/home/work/vidsearch/.ssh-keys/config"
curl -s -m 5 -o /dev/null "http://localhost:$LPORT/v1/models" || { echo "no model server on localhost:$LPORT"; exit 1; }
echo "[tunnel] alpha-eval:localhost:$RPORT -> here:localhost:$LPORT  (Ctrl-C to close)"
exec ssh -F "$SSHC" -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
  -o ExitOnForwardFailure=yes -N -R "$RPORT:localhost:$LPORT" alpha-eval
