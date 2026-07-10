# 🦙 llama-tune

A terminal UI that detects your machine's hardware (CPU, RAM, GPU/VRAM) and recommends optimal `llama.cpp` CLI parameters, then suggests GGUF models from Hugging Face that fit your hardware.

This project was developed with the assistance of [Claude Code](https://claude.com/claude-code), Anthropic's AI coding tool.

## Features

- **Hardware detection** — CPU (name, physical/logical core count), total/available RAM, and GPU VRAM (NVIDIA via NVML, Apple Silicon via unified memory, AMD/Intel/other vendors via DXGI on Windows, AMD/Intel via sysfs on Linux).
- **Parameter recommendations** — computes `--n-gpu-layers`, `--ctx-size`, `--threads`, `--batch-size`, `--ubatch-size`, `--no-mmap`/`--mlock`, and `--flash-attn`, each with a human-readable rationale explaining why that value was chosen for your hardware. `--n-gpu-layers` is recomputed for whichever model is selected in the Models tab (falling back to a generic full-offload recommendation when none is selected).
- **Model recommendations** — queries the Hugging Face API for GGUF models, filtered to those that fit within ~90% of your available VRAM (or RAM if no GPU), sorted by downloads. For each model, picks the highest-quality GGUF quantization (from `IQ1_S` up to `F16`) that still fits your machine's memory budget, and gives a direct download URL for that quant.
- **Installed model detection** — marks models in the list as already installed if a matching GGUF is found in the standard Hugging Face hub cache (`~/.cache/huggingface/hub`, or `HF_HOME`/`HUGGINGFACE_HUB_CACHE` if set) or in an optional local models directory set via `LLAMA_TUNE_MODELS_DIR`.
- **Launch llama.cpp directly** — configure the path to your `llama.cpp` executable (or the directory containing it, e.g. a version-tracking symlink) in the Settings tab (persisted between runs), then press `l` on a model in the Models tab to launch it as a background process with the recommended parameters already applied. Works with both older standalone `llama-cli`/`main` builds and newer builds that consolidated into a single `llama` binary with subcommands. If the model isn't installed yet, `l` downloads its best-fit quantization into the Hugging Face hub cache first, then launches automatically once the download finishes. A progress bar in the detail panel tracks the download, and `x` cancels it (deleting the partial file).
- **Quick relaunch** — the most recently launched model is remembered (shown in the Settings tab); press `L` from any tab to relaunch it, downloading it again first if it's no longer cached.

## Requirements

- Rust (2021 edition)
- An internet connection for the Models tab (queries `huggingface.co`)

## Building & running

```sh
cargo build --release
cargo run
```

NVIDIA VRAM detection via NVML is enabled by default (`nvidia` feature). To build without it:

```sh
cargo build --no-default-features
```

## Usage

| Key               | Action                                                |
| ----------------- | ------------------------------------------------------ |
| `1`/`2`/`3`/`4`   | Switch to Hardware / Parameters / Models / Settings tab |
| `Tab`             | Cycle through tabs                                     |
| `↑`/`k`, `↓`/`j`  | Select a model (Models tab)                             |
| `l`               | Launch the selected model (Models tab), downloading it first if not installed |
| `x`               | Cancel an in-progress download (Models tab)             |
| `L`               | Relaunch the last-launched model, from any tab          |
| `e` / `Enter`     | Edit the llama.cpp path (Settings tab)                  |
| `r`               | Re-detect hardware and re-fetch models                  |
| `q` / `Ctrl+C`    | Quit                                                    |

## Known limitations

- On Linux, integrated Intel GPUs are detected, but their reported "VRAM" is really a shared budget borrowed from system RAM, not true dedicated video memory — Intel's i915/Xe drivers don't expose a VRAM-total sysfs node the way AMD's `amdgpu` driver does, so there's no real figure to read.
- Discrete Intel Arc GPUs on Linux (which do have genuine dedicated VRAM) aren't sized accurately either, for the same reason: no sysfs node exposes their VRAM size, so they're treated the same as an integrated Intel GPU.
