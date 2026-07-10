//! Detects the local machine's CPU/RAM (via `sysinfo`) and GPU/VRAM (via
//! per-vendor, per-platform backends: NVML, DXGI, sysfs, or macOS sysctl).

use anyhow::Result;
use bytesize::ByteSize;
use sysinfo::System;

/// Detected GPU info (vendor-agnostic)
#[derive(Debug, Clone)]
pub(crate) struct GpuInfo {
    pub(crate) name: String,
    pub(crate) vram_bytes: u64,
    pub(crate) vendor: GpuVendor,
    /// True for a genuine discrete VRAM chip; false when `vram_bytes` is a
    /// budget borrowed from system RAM (integrated/unified memory), which
    /// isn't capacity distinct from `HardwareInfo::total_ram_bytes`.
    pub(crate) dedicated: bool,
}

// `Apple` is only ever constructed by the macOS-specific `detect_apple_silicon`
// below (`#[cfg(target_os = "macos")]`), so non-macOS builds legitimately never
// construct it — allowed rather than removed since it's needed on that platform.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    #[allow(dead_code)]
    Apple,
    Other(String),
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuVendor::Nvidia => write!(f, "NVIDIA"),
            GpuVendor::Amd => write!(f, "AMD"),
            GpuVendor::Intel => write!(f, "Intel"),
            GpuVendor::Apple => write!(f, "Apple Silicon"),
            GpuVendor::Other(s) => write!(f, "{}", s),
        }
    }
}

/// Maps a PCI `VendorId` (as reported by DXGI on Windows) to a `GpuVendor`.
impl From<u32> for GpuVendor {
    fn from(vendor_id: u32) -> Self {
        match vendor_id {
            0x10DE => GpuVendor::Nvidia,
            0x1002 | 0x1022 => GpuVendor::Amd,
            0x8086 => GpuVendor::Intel,
            other => GpuVendor::Other(format!("0x{other:04X}")),
        }
    }
}

/// Full hardware snapshot
#[derive(Debug, Clone)]
pub(crate) struct HardwareInfo {
    pub(crate) cpu_name: String,
    pub(crate) cpu_physical_cores: usize,
    pub(crate) cpu_logical_cores: usize,
    pub(crate) total_ram_bytes: u64,
    pub(crate) available_ram_bytes: u64,
    pub(crate) gpus: Vec<GpuInfo>,
}

impl HardwareInfo {
    /// Best VRAM available: prefers a dedicated GPU's VRAM if one is present
    /// (even if it's not first in `gpus`), else the first GPU's (shared)
    /// figure, else 0.
    pub(crate) fn primary_vram_bytes(&self) -> u64 {
        self.gpus
            .iter()
            .find(|g| g.dedicated)
            .or_else(|| self.gpus.first())
            .map(|g| g.vram_bytes)
            .unwrap_or(0)
    }

    /// Sum of VRAM across only genuinely dedicated GPUs, excluding
    /// integrated/unified GPUs whose reported memory is borrowed from system
    /// RAM rather than additive to it.
    pub(crate) fn total_dedicated_vram_bytes(&self) -> u64 {
        self.gpus
            .iter()
            .filter(|g| g.dedicated)
            .map(|g| g.vram_bytes)
            .sum()
    }

    pub(crate) fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }

    pub(crate) fn has_dedicated_gpu(&self) -> bool {
        self.gpus.iter().any(|g| g.dedicated)
    }

    /// Memory budget model/quant fitting decisions are judged against:
    /// dedicated VRAM if a discrete GPU is present, else available RAM. An
    /// integrated/unified GPU's shared memory isn't used here since it's
    /// drawn from the same pool `available_ram_bytes` already reports, not
    /// capacity on top of it.
    pub(crate) fn model_memory_bytes(&self) -> u64 {
        if self.has_dedicated_gpu() {
            self.primary_vram_bytes()
        } else {
            self.available_ram_bytes
        }
    }

    pub(crate) fn ram_display(&self) -> String {
        ByteSize(self.total_ram_bytes).to_string()
    }

    pub(crate) fn available_ram_display(&self) -> String {
        ByteSize(self.available_ram_bytes).to_string()
    }

    pub(crate) fn vram_display(&self) -> String {
        if self.has_gpu() {
            ByteSize(self.primary_vram_bytes()).to_string()
        } else {
            "No GPU".to_string()
        }
    }
}

/// Scan the system and return a `HardwareInfo`. This is entirely synchronous
/// (sysinfo refreshes, NVML/DXGI/sysfs calls); callers should run it via
/// `tokio::task::spawn_blocking` rather than awaiting it directly on an async
/// runtime thread.
pub(crate) fn detect() -> Result<HardwareInfo> {
    // sysinfo refresh
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_name = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());

    let cpu_logical_cores = sys.cpus().len();
    // sysinfo exposes physical core count directly on most platforms; only
    // fall back to logical / 2 (assuming hyper-threading) if that's None.
    let cpu_physical_cores = sys
        .physical_core_count()
        .unwrap_or(cpu_logical_cores / 2)
        .max(1);

    let total_ram_bytes = sys.total_memory();
    let available_ram_bytes = sys.available_memory();

    let gpus = detect_gpus(total_ram_bytes);

    Ok(HardwareInfo {
        cpu_name,
        cpu_physical_cores,
        cpu_logical_cores,
        total_ram_bytes,
        available_ram_bytes,
        gpus,
    })
}

fn detect_gpus(
    #[cfg_attr(not(target_os = "linux"), allow(unused_variables))] total_ram_bytes: u64,
) -> Vec<GpuInfo> {
    let mut gpus: Vec<GpuInfo> = Vec::new();

    // --- NVIDIA NVML ---
    // sysinfo has no cross-vendor GPU API, so each vendor/platform needs its
    // own detection path (NVML, DXGI, sysfs, Apple's sysctl fallback below).
    #[cfg(feature = "nvidia")]
    {
        if let Ok(nvml_gpus) = detect_nvidia_nvml() {
            gpus.extend(nvml_gpus);
        }
    }

    // --- Apple Silicon fallback: if no GPU found on macOS, use sysctl unified memory ---
    #[cfg(target_os = "macos")]
    if gpus.is_empty() {
        if let Some(g) = detect_apple_silicon() {
            gpus.push(g);
        }
    }

    // --- Windows: DXGI adapter enumeration covers AMD/Intel/other vendors ---
    #[cfg(windows)]
    {
        for g in detect_windows_dxgi() {
            // NVML (above) already covers NVIDIA GPUs; skip DXGI's entry
            // so the same physical GPU isn't reported twice.
            if g.vendor == GpuVendor::Nvidia && gpus.iter().any(|e| e.vendor == GpuVendor::Nvidia) {
                continue;
            }
            gpus.push(g);
        }
    }

    // --- Linux: sysfs covers AMD (amdgpu exposes mem_info_vram_total) and
    // Intel (no VRAM node, so reported as a shared-memory budget instead) ---
    #[cfg(target_os = "linux")]
    gpus.extend(detect_linux_sysfs_gpus(total_ram_bytes));

    gpus
}

#[cfg(feature = "nvidia")]
fn detect_nvidia_nvml() -> Result<Vec<GpuInfo>> {
    use nvml_wrapper::Nvml;
    let nvml = Nvml::init()?;
    let count = nvml.device_count()?;
    let mut gpus = Vec::new();
    for i in 0..count {
        let device = nvml.device_by_index(i)?;
        let name = device.name()?;
        let mem = device.memory_info()?;
        gpus.push(GpuInfo {
            name,
            vram_bytes: mem.total,
            vendor: GpuVendor::Nvidia,
            dedicated: true,
        });
    }
    Ok(gpus)
}

#[cfg(target_os = "macos")]
fn detect_apple_silicon() -> Option<GpuInfo> {
    // sysctl hw.memsize gives total unified memory on Apple Silicon
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let mem_str = String::from_utf8(output.stdout).ok()?;
    let mem_bytes: u64 = mem_str.trim().parse().ok()?;

    // On Apple Silicon, machdep.cpu.brand_string may be absent; fall back to hw.chip.
    let cpu_name = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();

    let chip_name = std::process::Command::new("sysctl")
        .args(["-n", "hw.chip"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();

    let combined = format!("{cpu_name} {chip_name}").to_lowercase();

    if combined.contains("apple") {
        // Unified memory: GPU can use all of it, but we report total as VRAM
        let label = if chip_name.trim().is_empty() {
            cpu_name
        } else {
            chip_name
        };
        Some(GpuInfo {
            name: label.trim().to_string(),
            vram_bytes: mem_bytes,
            vendor: GpuVendor::Apple,
            // Unified memory: this is the same physical pool as total_ram_bytes,
            // not memory distinct from it.
            dedicated: false,
        })
    } else {
        None
    }
}

/// Enumerate GPUs via DXGI, which reports dedicated VRAM for any vendor
/// (AMD, Intel, and NVIDIA as a fallback if NVML isn't available) — unlike
/// WMI's `Win32_VideoController.AdapterRAM`, which is known to misreport
/// (often truncating to ~4GB) on modern drivers.
#[cfg(windows)]
fn detect_windows_dxgi() -> Vec<GpuInfo> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
    };

    let mut gpus = Vec::new();

    let factory: IDXGIFactory1 = match unsafe { CreateDXGIFactory1() } {
        Ok(f) => f,
        Err(_) => return gpus,
    };

    for i in 0.. {
        let adapter = match unsafe { factory.EnumAdapters1(i) } {
            Ok(a) => a,
            Err(_) => break, // DXGI_ERROR_NOT_FOUND: no more adapters
        };
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };

        // Skip the WARP software rasterizer and similar non-physical adapters.
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }

        // Integrated GPUs (Intel especially) report a tiny `DedicatedVideoMemory`
        // (a BIOS-reserved aperture, often ~128MB) but a large `SharedSystemMemory`
        // — the real budget the driver lets the GPU draw from system RAM, and the
        // figure Task Manager shows as "Shared GPU memory". Discrete GPUs are the
        // opposite: large dedicated VRAM, comparatively small shared figure. Taking
        // the max of the two gives the actually-usable memory budget in both cases.
        let dedicated_bytes = desc.DedicatedVideoMemory as u64;
        let shared_bytes = desc.SharedSystemMemory as u64;
        let vram_bytes = dedicated_bytes.max(shared_bytes);
        if vram_bytes == 0 {
            continue;
        }
        // Whichever figure is larger indicates which kind of memory this
        // adapter actually relies on: a discrete GPU reports large dedicated/
        // small shared, an integrated one the reverse.
        let dedicated = dedicated_bytes >= shared_bytes;

        let name_len = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        let name = String::from_utf16_lossy(&desc.Description[..name_len]);

        let vendor = GpuVendor::from(desc.VendorId);

        gpus.push(GpuInfo {
            name,
            vram_bytes,
            vendor,
            dedicated,
        });
    }

    gpus
}

/// Scan `/sys/class/drm` for AMD and Intel GPUs. Only `amdgpu` exposes
/// `mem_info_vram_total`; Intel's i915/Xe drivers have no equivalent node
/// since (almost) all Intel GPUs are integrated and have no VRAM chip of
/// their own — the same situation as an integrated GPU on Windows (see
/// `detect_windows_dxgi`'s `SharedSystemMemory` handling) or Apple Silicon
/// (`detect_apple_silicon`) above, both of which report a non-dedicated
/// budget borrowed from system RAM rather than skipping the device. Intel is
/// handled the same way here instead of being dropped entirely. This still
/// under/over-reports for the (rarer) discrete Intel Arc cards, which do have
/// real VRAM but no sysfs node exposing its size either. NVIDIA is
/// intentionally skipped in favor of the richer NVML path above.
#[cfg(target_os = "linux")]
fn detect_linux_sysfs_gpus(total_ram_bytes: u64) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return gpus;
    };

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        // Only primary device nodes ("card0"), not connector subdirectories
        // like "card0-DP-1".
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }

        let device_dir = entry.path().join("device");
        let vendor_id = std::fs::read_to_string(device_dir.join("vendor")).ok();

        match vendor_id.as_deref().map(str::trim) {
            Some("0x1002") => {
                let Some(vram_bytes) =
                    std::fs::read_to_string(device_dir.join("mem_info_vram_total"))
                        .ok()
                        .and_then(|s| s.trim().parse::<u64>().ok())
                        .filter(|&v| v > 0)
                else {
                    continue;
                };
                gpus.push(GpuInfo {
                    name: format!("{} GPU ({name})", GpuVendor::Amd),
                    vram_bytes,
                    vendor: GpuVendor::Amd,
                    // amdgpu's mem_info_vram_total reflects a real VRAM chip.
                    dedicated: true,
                });
            }
            Some("0x8086") => {
                if total_ram_bytes == 0 {
                    continue;
                }
                gpus.push(GpuInfo {
                    name: format!("{} GPU ({name})", GpuVendor::Intel),
                    vram_bytes: total_ram_bytes,
                    vendor: GpuVendor::Intel,
                    // No dedicated VRAM chip to report on almost all Intel
                    // GPUs; this is the same shared system-RAM pool as
                    // total_ram_bytes, not capacity on top of it.
                    dedicated: false,
                });
            }
            _ => continue,
        }
    }

    gpus
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, vram_gb: u64, vendor: GpuVendor, dedicated: bool) -> GpuInfo {
        GpuInfo {
            name: name.to_string(),
            vram_bytes: vram_gb * 1_073_741_824,
            vendor,
            dedicated,
        }
    }

    fn hw(gpus: Vec<GpuInfo>, ram_gb: u64, available_ram_gb: u64) -> HardwareInfo {
        HardwareInfo {
            cpu_name: "Test CPU".to_string(),
            cpu_physical_cores: 8,
            cpu_logical_cores: 16,
            total_ram_bytes: ram_gb * 1_073_741_824,
            available_ram_bytes: available_ram_gb * 1_073_741_824,
            gpus,
        }
    }

    #[test]
    fn no_gpus_report_zero_vram_and_no_gpu() {
        let info = hw(vec![], 32, 16);
        assert_eq!(info.primary_vram_bytes(), 0);
        assert!(!info.has_gpu());
        assert!(!info.has_dedicated_gpu());
        assert_eq!(info.vram_display(), "No GPU");
    }

    #[test]
    fn primary_vram_prefers_dedicated_even_if_not_first() {
        let info = hw(
            vec![
                gpu("Integrated", 8, GpuVendor::Intel, false),
                gpu("Discrete", 12, GpuVendor::Nvidia, true),
            ],
            32,
            16,
        );
        assert_eq!(info.primary_vram_bytes(), 12 * 1_073_741_824);
        assert!(info.has_dedicated_gpu());
    }

    #[test]
    fn primary_vram_falls_back_to_first_when_no_dedicated() {
        let info = hw(
            vec![
                gpu("Integrated A", 4, GpuVendor::Intel, false),
                gpu("Integrated B", 6, GpuVendor::Amd, false),
            ],
            32,
            16,
        );
        assert_eq!(info.primary_vram_bytes(), 4 * 1_073_741_824);
        assert!(!info.has_dedicated_gpu());
    }

    #[test]
    fn total_dedicated_vram_excludes_shared_gpus() {
        let info = hw(
            vec![
                gpu("Discrete 1", 8, GpuVendor::Nvidia, true),
                gpu("Discrete 2", 8, GpuVendor::Amd, true),
                gpu("Integrated", 16, GpuVendor::Intel, false),
            ],
            32,
            16,
        );
        assert_eq!(info.total_dedicated_vram_bytes(), 16 * 1_073_741_824);
    }

    #[test]
    fn model_memory_uses_vram_only_when_dedicated() {
        let dedicated = hw(vec![gpu("Discrete", 12, GpuVendor::Nvidia, true)], 32, 16);
        assert_eq!(dedicated.model_memory_bytes(), 12 * 1_073_741_824);

        let integrated = hw(vec![gpu("Integrated", 16, GpuVendor::Intel, false)], 32, 16);
        assert_eq!(integrated.model_memory_bytes(), 16 * 1_073_741_824);

        let none = hw(vec![], 32, 16);
        assert_eq!(none.model_memory_bytes(), 16 * 1_073_741_824);
    }

    #[test]
    fn gpu_vendor_display_formats_known_and_unknown_vendors() {
        assert_eq!(GpuVendor::Nvidia.to_string(), "NVIDIA");
        assert_eq!(GpuVendor::Amd.to_string(), "AMD");
        assert_eq!(GpuVendor::Intel.to_string(), "Intel");
        assert_eq!(GpuVendor::Apple.to_string(), "Apple Silicon");
        assert_eq!(GpuVendor::Other("0x1234".to_string()).to_string(), "0x1234");
    }
}
