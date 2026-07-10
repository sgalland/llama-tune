//! Pure heuristic computation of recommended llama.cpp CLI parameters from
//! detected hardware and (optionally) a selected model's requirements.

use crate::hardware::HardwareInfo;

const GB: u64 = 1_073_741_824;

/// A computed set of optimal llama.cpp CLI parameters.
#[derive(Debug, Clone)]
pub(crate) struct LlamaCppParams {
    /// Number of transformer layers to offload to GPU (-ngl)
    pub(crate) n_gpu_layers: i32,
    /// Context window size in tokens (--ctx-size)
    pub(crate) ctx_size: u32,
    /// CPU threads for token evaluation (--threads)
    pub(crate) threads: u32,
    /// Batch size for prompt processing (--batch-size)
    pub(crate) batch_size: u32,
    /// Micro-batch size (--ubatch-size)
    pub(crate) ubatch_size: u32,
    /// Whether to use mmap (--no-mmap disables it)
    pub(crate) use_mmap: bool,
    /// Whether to use mlock to keep model in RAM
    pub(crate) use_mlock: bool,
    /// Flash attention flag (--flash-attn)
    pub(crate) flash_attn: bool,
    /// Human-readable explanation of each parameter choice
    pub(crate) rationale: Vec<String>,
}

impl LlamaCppParams {
    /// Build the CLI string you'd pass to llama.cpp
    pub(crate) fn to_cli_string(&self) -> String {
        let mut args = Vec::new();
        args.push(format!("--n-gpu-layers {}", self.n_gpu_layers));
        args.push(format!("--ctx-size {}", self.ctx_size));
        args.push(format!("--threads {}", self.threads));
        args.push(format!("--batch-size {}", self.batch_size));
        args.push(format!("--ubatch-size {}", self.ubatch_size));
        if !self.use_mmap {
            args.push("--no-mmap".to_string());
        }
        if self.use_mlock {
            args.push("--mlock".to_string());
        }
        if self.flash_attn {
            args.push("--flash-attn".to_string());
        }
        args.join(" \\\n  ")
    }

    /// Build the args as a flat `["--flag", "value", ...]` vector suitable for
    /// `std::process::Command::args`, separate from `to_cli_string`'s
    /// display-oriented formatting.
    pub(crate) fn to_cli_args(&self) -> Vec<String> {
        let mut args = vec![
            "--n-gpu-layers".to_string(),
            self.n_gpu_layers.to_string(),
            "--ctx-size".to_string(),
            self.ctx_size.to_string(),
            "--threads".to_string(),
            self.threads.to_string(),
            "--batch-size".to_string(),
            self.batch_size.to_string(),
            "--ubatch-size".to_string(),
            self.ubatch_size.to_string(),
        ];
        if !self.use_mmap {
            args.push("--no-mmap".to_string());
        }
        if self.use_mlock {
            args.push("--mlock".to_string());
        }
        if self.flash_attn {
            args.push("--flash-attn".to_string());
        }
        args
    }
}

/// Describes a model's basic requirements so we can compute n-gpu-layers.
#[derive(Debug, Clone)]
pub(crate) struct ModelRequirements {
    /// Total model size on disk in bytes (GGUF file)
    pub(crate) file_size_bytes: u64,
    /// Number of transformer layers
    pub(crate) num_layers: u32,
    /// Quantization label (e.g. "Q4_K_M")
    pub(crate) quantization: String,
}

impl ModelRequirements {
    pub(crate) fn bytes_per_layer(&self) -> u64 {
        if self.num_layers == 0 {
            return self.file_size_bytes;
        }
        self.file_size_bytes / self.num_layers as u64
    }

    /// Approximate VRAM needed for KV cache at given ctx size (rough heuristic)
    pub(crate) fn kv_cache_bytes(&self, ctx_size: u32) -> u64 {
        // KV cache ≈ 2 * num_layers * ctx * head_dim * bytes_per_element
        // Without knowing head_dim, use empirical ~0.5MB per layer per 1k ctx tokens
        (self.num_layers as u64) * (ctx_size as u64 / 1024) * 512 * 1024
    }
}

/// Compute optimal parameters for the given hardware.
/// If `model` is provided, we compute layer offloading too.
pub(crate) fn compute(hw: &HardwareInfo, model: Option<&ModelRequirements>) -> LlamaCppParams {
    let mut rationale = Vec::new();

    // ── Threads ──────────────────────────────────────────────────────────────
    // Use physical cores. With full GPU offload, threads matter less.
    let threads = hw.cpu_physical_cores as u32;
    rationale.push(format!(
        "--threads {threads}: matches your {threads} physical CPU cores for best throughput"
    ));

    // ── Context size ─────────────────────────────────────────────────────────
    let vram = hw.primary_vram_bytes();
    let ram = hw.available_ram_bytes;

    let ctx_size = if vram >= 24 * GB {
        32768
    } else if vram >= 16 * GB {
        16384
    } else if vram >= 8 * GB {
        8192
    } else if vram >= 4 * GB {
        4096
    } else if ram >= 32 * GB {
        8192 // CPU-only with lots of RAM
    } else if ram >= 16 * GB {
        4096
    } else {
        2048
    };
    rationale.push(format!(
        "--ctx-size {ctx_size}: balanced for your VRAM ({}) / RAM ({})",
        hw.vram_display(),
        hw.ram_display()
    ));

    // ── GPU layers ───────────────────────────────────────────────────────────
    let n_gpu_layers = if !hw.has_gpu() {
        rationale.push("--n-gpu-layers 0: no GPU detected, running CPU-only".to_string());
        0
    } else if let Some(m) = model {
        compute_gpu_layers(hw, m, ctx_size, &mut rationale)
    } else {
        // No model selected yet — recommend full offload as a starting point
        rationale
            .push("--n-gpu-layers -1: offload all layers (adjust down if you get OOM)".to_string());
        -1
    };

    if hw.has_gpu() && !hw.has_dedicated_gpu() && n_gpu_layers != 0 {
        rationale.push(
            "Note: GPU memory is shared with system RAM (integrated graphics) — offload only \
             helps if your llama.cpp build has GPU backend support (Vulkan/Metal/etc.); a \
             CPU-only build ignores -ngl"
                .to_string(),
        );
    }

    // ── Batch sizes ──────────────────────────────────────────────────────────
    // Larger batch = faster prompt processing; smaller ubatch = less peak VRAM
    let batch_size = if vram >= 8 * GB || ram >= 32 * GB {
        2048
    } else {
        512
    };
    let ubatch_size = batch_size / 4;
    rationale.push(format!(
        "--batch-size {batch_size} --ubatch-size {ubatch_size}: optimizes prompt throughput"
    ));

    // ── mmap / mlock ─────────────────────────────────────────────────────────
    // mmap is good on systems with lots of RAM; mlock keeps pages hot but needs RAM
    let use_mmap = true;
    let use_mlock = ram >= 16 * GB && n_gpu_layers == 0;
    if use_mlock {
        rationale
            .push("--mlock: enough RAM to keep model pages locked (CPU-only mode)".to_string());
    }

    // ── Flash attention ──────────────────────────────────────────────────────
    // Beneficial when GPU layers > 0; saves VRAM on KV cache
    let flash_attn = hw.has_gpu() && n_gpu_layers != 0;
    if flash_attn {
        rationale.push("--flash-attn: reduces VRAM usage for KV cache on GPU".to_string());
    }

    LlamaCppParams {
        n_gpu_layers,
        ctx_size,
        threads,
        batch_size,
        ubatch_size,
        use_mmap,
        use_mlock,
        flash_attn,
        rationale,
    }
}

fn compute_gpu_layers(
    hw: &HardwareInfo,
    model: &ModelRequirements,
    ctx_size: u32,
    rationale: &mut Vec<String>,
) -> i32 {
    let vram = hw.primary_vram_bytes();
    let bytes_per_layer = model.bytes_per_layer();
    rationale.push(format!(
        "Layer size estimated from {} quantization: ~{}MB/layer",
        model.quantization,
        bytes_per_layer / 1_048_576
    ));

    if bytes_per_layer == 0 {
        rationale.push("--n-gpu-layers -1: no per-layer data; offloading all".to_string());
        return -1;
    }

    let kv_cache = model.kv_cache_bytes(ctx_size);
    // Reserve ~10% VRAM for CUDA/ROCm overhead + output layer
    let usable_vram = (vram as f64 * 0.90) as u64;

    if usable_vram < kv_cache {
        // Can't even fit the KV cache — go CPU only
        rationale.push(format!(
            "--n-gpu-layers 0: insufficient VRAM ({}) even for KV cache (~{}MB)",
            hw.vram_display(),
            kv_cache / 1_048_576
        ));
        return 0;
    }

    let vram_for_weights = usable_vram - kv_cache;
    let max_layers = (vram_for_weights / bytes_per_layer) as i32;
    let total_layers = model.num_layers as i32;

    if max_layers >= total_layers {
        rationale.push(format!(
            "--n-gpu-layers {total_layers}: full model fits in VRAM ({})",
            hw.vram_display()
        ));
        total_layers
    } else {
        rationale.push(format!(
            "--n-gpu-layers {max_layers}: partial offload — {max_layers}/{total_layers} layers fit in {}",
            hw.vram_display()
        ));
        max_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{GpuInfo, GpuVendor, HardwareInfo};

    fn hw(vram_gb: u64, dedicated: bool, ram_gb: u64) -> HardwareInfo {
        let gpus = if vram_gb > 0 {
            vec![GpuInfo {
                name: "Test GPU".to_string(),
                vram_bytes: vram_gb * GB,
                vendor: GpuVendor::Nvidia,
                dedicated,
            }]
        } else {
            Vec::new()
        };
        HardwareInfo {
            cpu_name: "Test CPU".to_string(),
            cpu_physical_cores: 8,
            cpu_logical_cores: 16,
            total_ram_bytes: ram_gb * GB,
            available_ram_bytes: ram_gb * GB,
            gpus,
        }
    }

    fn model(file_size_gb: u64, num_layers: u32, quant: &str) -> ModelRequirements {
        ModelRequirements {
            file_size_bytes: file_size_gb * GB,
            num_layers,
            quantization: quant.to_string(),
        }
    }

    #[test]
    fn no_gpu_runs_cpu_only() {
        let params = compute(&hw(0, false, 32), None);
        assert_eq!(params.n_gpu_layers, 0);
        assert!(!params.flash_attn);
        assert_eq!(params.threads, 8);
    }

    #[test]
    fn dedicated_gpu_no_model_offloads_all() {
        let params = compute(&hw(24, true, 32), None);
        assert_eq!(params.n_gpu_layers, -1);
        assert!(params.flash_attn);
    }

    #[test]
    fn ctx_size_scales_with_vram_tiers() {
        assert_eq!(compute(&hw(24, true, 8), None).ctx_size, 32768);
        assert_eq!(compute(&hw(16, true, 8), None).ctx_size, 16384);
        assert_eq!(compute(&hw(8, true, 8), None).ctx_size, 8192);
        assert_eq!(compute(&hw(4, true, 8), None).ctx_size, 4096);
    }

    #[test]
    fn ctx_size_falls_back_to_ram_tiers_without_gpu() {
        assert_eq!(compute(&hw(0, false, 32), None).ctx_size, 8192);
        assert_eq!(compute(&hw(0, false, 16), None).ctx_size, 4096);
        assert_eq!(compute(&hw(0, false, 8), None).ctx_size, 2048);
    }

    #[test]
    fn batch_size_scales_with_memory() {
        let big = compute(&hw(8, true, 8), None);
        assert_eq!(big.batch_size, 2048);
        assert_eq!(big.ubatch_size, 512);

        let small = compute(&hw(4, true, 8), None);
        assert_eq!(small.batch_size, 512);
        assert_eq!(small.ubatch_size, 128);
    }

    #[test]
    fn mlock_only_when_cpu_only_and_enough_ram() {
        assert!(compute(&hw(0, false, 16), None).use_mlock);
        assert!(!compute(&hw(0, false, 8), None).use_mlock);
        // Enough RAM but a GPU is offloading layers: mlock shouldn't apply.
        assert!(!compute(&hw(24, true, 32), None).use_mlock);
    }

    #[test]
    fn full_model_fits_in_vram() {
        let m = model(10, 40, "Q4_K_M");
        let params = compute(&hw(24, true, 32), Some(&m));
        assert_eq!(params.n_gpu_layers, 40);
    }

    #[test]
    fn partial_offload_when_model_exceeds_vram() {
        let m = model(80, 80, "Q8_0");
        let params = compute(&hw(8, true, 32), Some(&m));
        assert!(params.n_gpu_layers > 0);
        assert!(params.n_gpu_layers < 80);
    }

    #[test]
    fn falls_back_to_cpu_when_vram_cant_fit_kv_cache() {
        // A huge layer count blows up the estimated KV-cache size past what a
        // 1GB card can hold, even before accounting for the weights.
        let m = model(1, 1000, "Q4_K_M");
        let params = compute(&hw(1, true, 8), Some(&m));
        assert_eq!(params.n_gpu_layers, 0);
        assert!(!params.flash_attn);
    }

    #[test]
    fn zero_layers_offloads_all_for_lack_of_data() {
        let m = model(0, 0, "unknown");
        let params = compute(&hw(24, true, 32), Some(&m));
        assert_eq!(params.n_gpu_layers, -1);
    }

    #[test]
    fn bytes_per_layer_falls_back_to_file_size_when_layers_unknown() {
        let m = model(5, 0, "Q4_K_M");
        assert_eq!(m.bytes_per_layer(), 5 * GB);
    }

    #[test]
    fn bytes_per_layer_divides_evenly() {
        let m = model(40, 40, "Q4_K_M");
        assert_eq!(m.bytes_per_layer(), GB);
    }

    #[test]
    fn cli_string_reflects_flags() {
        let params = compute(&hw(0, false, 16), None);
        let s = params.to_cli_string();
        assert!(s.contains("--n-gpu-layers 0"));
        assert!(s.contains("--no-mmap") != params.use_mmap);
        assert!(s.contains("--mlock"));
        assert!(!s.contains("--flash-attn"));
    }

    #[test]
    fn cli_args_omit_disabled_flags() {
        let params = compute(&hw(24, true, 32), None);
        let args = params.to_cli_args();
        assert!(args.contains(&"--flash-attn".to_string()));
        assert!(!args.contains(&"--no-mmap".to_string()));
        assert!(!args.contains(&"--mlock".to_string()));
    }
}
