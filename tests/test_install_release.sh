#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

if ! grep -q 'releases/latest/download' install.sh; then
  echo "expected install.sh to download release assets" >&2
  exit 1
fi

if grep -q 'cargo install --git' install.sh; then
  echo "expected install.sh to stop compiling from source" >&2
  exit 1
fi

echo "install.sh release-download checks passed"
