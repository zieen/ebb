#!/usr/bin/env bash

set -euo pipefail

REPO_URL="https://github.com/zieen/ebb"
BIN_NAME="ebb"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_cmds() {
  local missing=0
  for cmd in curl tar uname mktemp; do
    if ! need_cmd "$cmd"; then
      echo "Missing required command: $cmd" >&2
      missing=1
    fi
  done

  if [[ "$missing" -ne 0 ]]; then
    exit 1
  fi
}

detect_platform_label() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Linux)
      case "$arch" in
        x86_64|amd64) echo "linux-x86_64" ;;
        *) return 1 ;;
      esac
      ;;
    Darwin)
      case "$arch" in
        x86_64) echo "macos-x86_64" ;;
        arm64)
          echo "macOS arm64 is not published yet. Please use the x86_64 build with Rosetta or add a macOS arm64 release target." >&2
          return 1
          ;;
        *) return 1 ;;
      esac
      ;;
    *)
      return 1
      ;;
  esac
}

download_url_for() {
  local asset_name="${1:?asset name required}"
  echo "${REPO_URL}/releases/latest/download/${asset_name}"
}

install_ebb() {
  local platform archive_name download_url temp_dir extracted_binary
  platform="$(detect_platform_label)" || {
    echo "Unsupported platform: $(uname -s) $(uname -m)" >&2
    exit 1
  }

  archive_name="ebb-${platform}.tar.gz"
  download_url="$(download_url_for "$archive_name")"
  temp_dir="$(mktemp -d)"

  trap 'rm -rf "$temp_dir"' EXIT

  echo "Downloading ${archive_name}..."
  curl -fsSL "$download_url" -o "$temp_dir/$archive_name"

  mkdir -p "$temp_dir/extract" "$INSTALL_DIR"
  tar -xzf "$temp_dir/$archive_name" -C "$temp_dir/extract"

  extracted_binary="$temp_dir/extract/$BIN_NAME"
  if [[ ! -f "$extracted_binary" ]]; then
    echo "Downloaded archive did not contain ${BIN_NAME}" >&2
    exit 1
  fi

  install -m 755 "$extracted_binary" "$INSTALL_DIR/$BIN_NAME"
}

print_path_hint() {
  if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
    echo "Add ${INSTALL_DIR} to your PATH if '${BIN_NAME}' is not found."
  fi
}

print_success() {
  echo
  echo "Installed ${BIN_NAME} to ${INSTALL_DIR}/${BIN_NAME}."
  print_path_hint
  echo "Run '${BIN_NAME} --help' or '${BIN_NAME} setup' to get started."
}

ensure_cmds
install_ebb
print_success
