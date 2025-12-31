#!/usr/bin/env bash
set -e

#############################################
# Offline Voice Assistant installer
# - Idempotent
# - Portable
# - Supports --dry-run and --no-service
#############################################

SERVICE_NAME="assistant"
DRY_RUN=false
NO_SERVICE=false

# ---- argument parsing ----------------------------------------

for arg in "$@"; do
  case "$arg" in
    --dry-run|-n)
      DRY_RUN=true
      ;;
    --no-service)
      NO_SERVICE=true
      ;;
    --help|-h)
      echo "Usage: ./install.sh [options]"
      echo
      echo "Options:"
      echo "  --dry-run, -n    Show what would be done without changing the system"
      echo "  --no-service     Do not install or manage systemd service"
      echo "  --help, -h       Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

# ---- helpers -------------------------------------------------

log() {
  echo "[install] $1"
}

run() {
  if $DRY_RUN; then
    echo "[dry-run] $*"
  else
    eval "$@"
  fi
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command '$1' not found"
    exit 1
  }
}

# ---- basic checks --------------------------------------------

require_cmd python3
require_cmd sudo

if ! $NO_SERVICE; then
  require_cmd systemctl
fi

# Resolve repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME="$(id -un)"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"

log "Repo root    : $REPO_ROOT"
log "Install user : $USER_NAME"

if $DRY_RUN; then
  log "DRY RUN MODE ENABLED — no changes will be made"
fi

if $NO_SERVICE; then
  log "NO-SERVICE MODE ENABLED — systemd will not be modified"
fi

# ---- system dependencies -------------------------------------

log "Installing system dependencies (apt)"

run "sudo apt-get update"
run "sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  libasound2-dev \
  libopenblas-dev \
  build-essential"

# ---- python virtual environment -------------------------------

if [ ! -d "$VENV_DIR" ]; then
  log "Creating Python virtual environment"
  run "python3 -m venv \"$VENV_DIR\""
else
  log "Virtual environment already exists"
fi

log "Installing/updating Python dependencies"
run "\"$PYTHON_BIN\" -m pip install --upgrade pip"
run "\"$PYTHON_BIN\" -m pip install -r \"$REPO_ROOT/requirements.txt\""

# ---- systemd service (optional) -------------------------------

if ! $NO_SERVICE; then
  TEMPLATE="$REPO_ROOT/systemd/assistant.service.in"

  if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: service template not found: $TEMPLATE"
    exit 1
  fi

  log "Generating systemd service from template"

  TMP_SERVICE="$(mktemp)"

  run "sed \
    -e 's|@USER@|$USER_NAME|g' \
    -e 's|@WORKDIR@|$REPO_ROOT|g' \
    -e 's|@PYTHON@|$PYTHON_BIN|g' \
    \"$TEMPLATE\" > \"$TMP_SERVICE\""

  log "Installing systemd service to $SERVICE_DST"
  run "sudo cp \"$TMP_SERVICE\" \"$SERVICE_DST\""
  run "sudo chmod 644 \"$SERVICE_DST\""
  run "rm -f \"$TMP_SERVICE\""

  log "Reloading systemd"
  run "sudo systemctl daemon-reexec"
  run "sudo systemctl daemon-reload"

  log "Enabling service '$SERVICE_NAME'"
  run "sudo systemctl enable \"$SERVICE_NAME\""

  log "Restarting service '$SERVICE_NAME'"
  run "sudo systemctl restart \"$SERVICE_NAME\""
else
  log "Skipping systemd service installation"
fi

# ---- status ---------------------------------------------------

log "Installation complete."

if $DRY_RUN; then
  log "Dry run finished — no changes were made."
elif ! $NO_SERVICE; then
  log "Service status:"
  systemctl --no-pager status "$SERVICE_NAME" || true
  echo
  log "Logs:"
  echo "  journalctl -u $SERVICE_NAME -f"
else
  log "Service was not installed. Run manually with:"
  echo "  PYTHONPATH=src $PYTHON_BIN -m assistant.main --config config/config.yaml"
fi
