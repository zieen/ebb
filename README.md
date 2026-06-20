# ebb

`ebb` is a Rust terminal app for refining text with an LLM.

It lets you:

- install from prebuilt GitHub Release binaries
- choose an LLM vendor locally with `ebb setup`
- save the API key in a local `.env`
- send text from the command line and get a refined result back

## Install

Install the latest release with:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/zieen/ebb/main/install.sh)"
```

By default the installer places the binary in `~/.local/bin`.

If that directory is not already in your `PATH`, add it first:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

You can also override the install location:

```bash
INSTALL_DIR="$HOME/bin" /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/zieen/ebb/main/install.sh)"
```

## Quick Start

Run setup first:

```bash
ebb setup
```

The setup flow will:

- ask you to choose a vendor
- ask for the vendor API key
- save `LLM_VENDOR`, `LLM_MODEL`, and the provider key in a local `.env`

Supported vendors in the setup flow:

- OpenAI
- Gemini
- Anthropic
- DeepSeek

Then refine text from the terminal:

```bash
ebb i has a apple
```

## Example

```bash
$ ebb i has a apple
BB: I have an apple.

Mistakes:
- "has" should be "have"
- "a apple" should be "an apple"
```

## Local Development

Clone the repo and run it locally:

```bash
git clone https://github.com/zieen/ebb.git
cd ebb
cargo run -- setup
cargo run -- i has a apple
```

Useful commands:

```bash
cargo check
cargo test
```

## Releases

GitHub Actions builds release binaries for:

- Linux `x86_64`
- macOS `x86_64`
- Windows `x86_64`

The workflow lives in [release.yml](file:///Users/enzii/Projects/ebb/.github/workflows/release.yml) and runs on tags matching `v*`.

To publish a release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

That workflow packages and uploads:

- `ebb-linux-x86_64.tar.gz`
- `ebb-macos-x86_64.tar.gz`
- `ebb-windows-x86_64.tar.gz`

## Notes

- The installer currently supports Linux `x86_64` and macOS `x86_64`.
- macOS `arm64` is not published yet.
- `ebb` loads configuration from the local `.env` file at startup.
