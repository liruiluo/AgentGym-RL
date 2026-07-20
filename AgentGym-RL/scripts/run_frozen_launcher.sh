#!/usr/bin/env bash
# Snapshot a launcher into its run directory before executing it. This keeps a
# long-running Bash process independent from later edits to the source file.
set -euo pipefail

die() {
  printf 'run_frozen_launcher: %s\n' "$*" >&2
  exit 70
}

if [ "$#" -lt 2 ]; then
  die "usage: $0 RUN_DIR SOURCE_LAUNCHER [ARG ...]"
fi

RUN_DIR=$1
SOURCE_LAUNCHER=$2
shift 2

case "$SOURCE_LAUNCHER" in
  /*) ;;
  *)
    SOURCE_LAUNCHER=$(cd "$(dirname "$SOURCE_LAUNCHER")" && pwd -P)/$(basename "$SOURCE_LAUNCHER")
    ;;
esac

[ -f "$SOURCE_LAUNCHER" ] || die "source launcher is not a regular file: $SOURCE_LAUNCHER"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

SNAPSHOT_ROOT=$RUN_DIR/launcher_snapshots
mkdir -p "$SNAPSHOT_ROOT"

TEMP_DIR=""
cleanup_temp() {
  if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
    chmod -R u+w "$TEMP_DIR" 2>/dev/null || true
    rm -rf "$TEMP_DIR"
  fi
}
trap cleanup_temp EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

ATTEMPT=1
SOURCE_SHA=""
FROZEN_SHA=""
while [ "$ATTEMPT" -le 5 ]; do
  TEMP_DIR=$(mktemp -d "$SNAPSHOT_ROOT/.freeze.XXXXXX") || die "could not create snapshot directory"
  BEFORE_SHA=$(sha256sum "$SOURCE_LAUNCHER" | awk '{print $1}')
  cp "$SOURCE_LAUNCHER" "$TEMP_DIR/launcher.sh"
  AFTER_SHA=$(sha256sum "$SOURCE_LAUNCHER" | awk '{print $1}')
  COPY_SHA=$(sha256sum "$TEMP_DIR/launcher.sh" | awk '{print $1}')

  if [ "$BEFORE_SHA" = "$AFTER_SHA" ] && [ "$BEFORE_SHA" = "$COPY_SHA" ]; then
    SOURCE_SHA=$BEFORE_SHA
    FROZEN_SHA=$COPY_SHA
    break
  fi

  cleanup_temp
  TEMP_DIR=""
  ATTEMPT=$((ATTEMPT + 1))
  sleep 0.05
done

[ -n "$SOURCE_SHA" ] || die "source launcher changed during all snapshot attempts"

FROZEN_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SNAPSHOT_ID=$(date -u +%Y%m%dT%H%M%SZ)_pid$$_${SOURCE_SHA:0:12}
FINAL_DIR=$SNAPSHOT_ROOT/$SNAPSHOT_ID
[ ! -e "$FINAL_DIR" ] || die "snapshot path already exists: $FINAL_DIR"

{
  printf 'schema_version=1\n'
  printf 'frozen_at_utc=%s\n' "$FROZEN_AT"
  printf 'source_path=%s\n' "$SOURCE_LAUNCHER"
  printf 'source_sha256=%s\n' "$SOURCE_SHA"
  printf 'frozen_path=%s/launcher.sh\n' "$FINAL_DIR"
  printf 'frozen_sha256=%s\n' "$FROZEN_SHA"
  printf 'snapshot_attempts=%s\n' "$ATTEMPT"
} > "$TEMP_DIR/manifest.env"
printf '%s  %s/launcher.sh\n' "$FROZEN_SHA" "$FINAL_DIR" > "$TEMP_DIR/launcher.sha256"

chmod 0444 "$TEMP_DIR/launcher.sh" "$TEMP_DIR/manifest.env" "$TEMP_DIR/launcher.sha256"
mv "$TEMP_DIR" "$FINAL_DIR"
TEMP_DIR=""
chmod 0555 "$FINAL_DIR"

OBSERVED_SHA=$(sha256sum "$FINAL_DIR/launcher.sh" | awk '{print $1}')
[ "$OBSERVED_SHA" = "$FROZEN_SHA" ] || die "frozen launcher hash changed before exec"

trap - EXIT HUP INT TERM
printf '[launcher-freeze] source=%s sha256=%s frozen=%s\n' \
  "$SOURCE_LAUNCHER" "$SOURCE_SHA" "$FINAL_DIR/launcher.sh" >&2

exec env \
  AGENTMEMORY_FROZEN_LAUNCHER=1 \
  AGENTMEMORY_LAUNCHER_SOURCE_PATH="$SOURCE_LAUNCHER" \
  AGENTMEMORY_LAUNCHER_SOURCE_SHA256="$SOURCE_SHA" \
  AGENTMEMORY_LAUNCHER_SNAPSHOT_DIR="$FINAL_DIR" \
  AGENTMEMORY_LAUNCHER_SNAPSHOT_MANIFEST="$FINAL_DIR/manifest.env" \
  /usr/bin/env bash "$FINAL_DIR/launcher.sh" "$@"
