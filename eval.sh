#!/usr/bin/env bash
# Existing RT-Splatting helper repurposed for VIGOR's implementation history.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
case "${1:---snapshot}" in
  --watch)
    exec /usr/bin/python3 -u -c 'from utils.system_utils import watch_implementation; watch_implementation()'
    ;;
  --snapshot)
    exec /usr/bin/python3 -c 'from utils.system_utils import record_implementation_change; record_implementation_change()'
    ;;
  *)
    echo 'Usage: bash eval.sh [--snapshot|--watch]' >&2
    exit 2
    ;;
esac
