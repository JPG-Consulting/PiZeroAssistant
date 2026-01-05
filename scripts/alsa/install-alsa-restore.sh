#!/usr/bin/env bash
set -e

# ------------------------------------------------------------
# Install and enable ALSA restore systemd service
# Can be run standalone, independently of install.sh
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ALSA_SCRIPT_SRC="$REPO_ROOT/scripts/alsa/restore_respeaker_mic.sh"
ALSA_SCRIPT_DST="/usr/local/bin/restore_respeaker_mic.sh"

ALSA_SERVICE_SRC="$REPO_ROOT/systemd/alsa-restore.service"
ALSA_SERVICE_DST="/etc/systemd/system/alsa-restore.service"

log() {
  echo "[alsa-install] $1"
}

require_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: this script must be run as root"
    echo "Hint: sudo $0"
    exit 1
  fi
}

require_file() {
  if [ ! -f "$1" ]; then
    echo "ERROR: required file not found: $1"
    exit 1
  fi
}

# ------------------------------------------------------------

require_root

require_file "$ALSA_SCRIPT_SRC"
require_file "$ALSA_SERVICE_SRC"

log "Installing ALSA restore script"
cp "$ALSA_SCRIPT_SRC" "$ALSA_SCRIPT_DST"
chmod +x "$ALSA_SCRIPT_DST"

log "Installing systemd service"
cp "$ALSA_SERVICE_SRC" "$ALSA_SERVICE_DST"

log "Reloading systemd"
systemctl daemon-reexec
systemctl daemon-reload

log "Enabling alsa-restore service"
systemctl enable alsa-restore

log "Starting alsa-restore service"
systemctl restart alsa-restore

log "ALSA restore service installed and active"
