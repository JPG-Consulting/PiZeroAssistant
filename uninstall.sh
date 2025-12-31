#!/usr/bin/env bash
set -e

#############################################
# Offline Voice Assistant uninstaller
# - Safe to run multiple times
# - Does NOT delete repo files
#############################################

SERVICE_NAME="assistant"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"

log() {
  echo "[uninstall] $1"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command '$1' not found"
    exit 1
  }
}

require_cmd sudo
require_cmd systemctl

log "Stopping service (if running)"
sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true

log "Disabling service (if enabled)"
sudo systemctl disable "$SERVICE_NAME" 2>/dev/null || true

if [ -f "$SERVICE_DST" ]; then
  log "Removing systemd service file"
  sudo rm -f "$SERVICE_DST"
else
  log "Service file not found (already removed)"
fi

log "Reloading systemd"
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

log "Uninstall complete."
echo
log "Note:"
log "- Repository files were NOT deleted"
log "- Virtual environment was NOT removed"
log "- You may safely delete the repo directory manually if desired"
