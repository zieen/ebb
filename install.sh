#!/usr/bin/env bash

set -euo pipefail

REPO_URL="https://github.com/zieen/ebb.git"
BIN_NAME="ebb"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_cargo() {
  if need_cmd cargo; then
    return
  fi

  echo "Rust toolchain not found. Installing rustup..."
  curl https://sh.rustup.rs -sSf | sh -s -- -y

  export PATH="$HOME/.cargo/bin:$PATH"

  if ! need_cmd cargo; then
    echo "cargo is still not available after rustup installation." >&2
    echo "Open a new shell or add \$HOME/.cargo/bin to your PATH, then rerun this installer." >&2
    exit 1
  fi
}

install_ebb() {
  echo "Installing ${BIN_NAME} from ${REPO_URL}..."
  cargo install --git "${REPO_URL}" --locked --force "${BIN_NAME}"
}

print_success() {
  local cargo_bin_dir
  cargo_bin_dir="${CARGO_HOME:-$HOME/.cargo}/bin"

  echo
  echo "Installed ${BIN_NAME}."
  if [[ ":$PATH:" != *":${cargo_bin_dir}:"* ]]; then
    echo "Add ${cargo_bin_dir} to your PATH if '${BIN_NAME}' is not found."
  fi
  echo "Run '${BIN_NAME} --help' or '${BIN_NAME} setup' to get started."
}

ensure_cargo
install_ebb
print_success
