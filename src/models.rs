//! Queries the Hugging Face API for GGUF models that fit the detected
//! hardware, and picks the best quant for a given memory budget.

use anyhow::{Context, Result};
use futures_util::stream::{self, StreamExt};
use serde::Deserialize;

use crate::hardware::HardwareInfo;

/// A model entry returned by the Hugging Face API
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct HfModel {
    #[serde(rename = "modelId")]
    pub(crate) model_id: String,
    pub(crate) downloads: Option<u64>,
    pub(crate) likes: Option<u32>,
    #[serde(rename = "lastModified")]
    pub(crate) last_modified: Option<String>,
    pub(crate) tags: Option<Vec<String>>,
    pub(crate) siblings: Option<Vec<HfSibling>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct HfSibling {
    pub(crate) rfilename: String,
    pub(crate) size: Option<u64>,
}

/// Known llama.cpp GGUF quantization types, ordered from lowest to highest
/// quality/bits-per-weight (per llama.cpp's quantize benchmarks). Used to pick
/// the best-quality quant that still fits a given memory budget.
const QUANT_QUALITY_ORDER: &[&str] = &[
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "Q2_K_S", "Q2_K", "IQ3_XXS", "IQ3_XS",
    "Q3_K_S", "IQ3_S", "IQ3_M", "Q3_K_M", "Q3_K_L", "IQ4_XS", "Q4_0", "IQ4_NL", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16",
];

impl HfModel {
    /// Quant labels and file sizes available for this repo, ordered from
    /// lowest to highest quality.
    fn quant_options(&self) -> Vec<(&'static str, u64)> {
        let sibs = self.siblings.as_deref().unwrap_or(&[]);
        QUANT_QUALITY_ORDER
            .iter()
            .filter_map(|&quant| {
                let file = sibs.iter().find(|s| matches_quant(&s.rfilename, quant))?;
                Some((quant, file.size?))
            })
            .collect()
    }

    fn find_quant_sibling(&self, quant: &str) -> Option<&HfSibling> {
        self.siblings
            .as_deref()?
            .iter()
            .find(|s| matches_quant(&s.rfilename, quant))
    }

    /// Size in bytes of a specific quant's GGUF file, if present.
    pub(crate) fn quant_size_bytes(&self, quant: &str) -> Option<u64> {
        self.find_quant_sibling(quant).and_then(|s| s.size)
    }

    /// Best quant for the given memory budget: the highest-quality quant whose
    /// GGUF file fits, falling back to the smallest available quant if even
    /// that doesn't fit (so we still recommend something).
    pub(crate) fn best_quant_for_memory(&self, mem_budget_bytes: u64) -> Option<&'static str> {
        let options = self.quant_options();
        options
            .iter()
            .rev()
            .find(|(_, size)| *size <= mem_budget_bytes)
            .or_else(|| options.first())
            .map(|(quant, _)| *quant)
    }

    /// Number of layers (inferred from tags, e.g. "7B" → 32, "13B" → 40, "70B" → 80)
    pub(crate) fn estimated_layers(&self) -> u32 {
        let id = self.model_id.to_lowercase();
        let tags = self.tags.as_deref().unwrap_or(&[]);
        let combined: String = format!("{} {}", id, tags.join(" ")).to_lowercase();

        if combined.contains("70b") || combined.contains("65b") {
            80
        } else if combined.contains("34b") || combined.contains("30b") {
            60
        } else if combined.contains("13b") || combined.contains("14b") {
            40
        } else if combined.contains("8b") || combined.contains("7b") {
            32
        } else if combined.contains("3b") || combined.contains("2b") {
            26
        } else if combined.contains("1b") || combined.contains("0.5b") {
            16
        } else {
            32 // fallback
        }
    }

    /// Return the download URL for a given quant's GGUF file, if present.
    pub(crate) fn download_url_for(&self, quant: &str) -> Option<String> {
        let file = self.find_quant_sibling(quant)?;
        Some(format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.model_id, file.rfilename
        ))
    }
}

/// True if `filename` names exactly `quant`, not merely contains it as a
/// prefix of a different/longer quant label. Many repos (e.g. bartowski's)
/// ship both a base quant and size variants sharing its prefix — `Q2_K.gguf`
/// alongside `Q2_K_L.gguf`, or `Q4_0_4_4.gguf`/`Q4_0_4_8.gguf` with no plain
/// `Q4_0.gguf` at all — so plain substring matching silently binds the wrong
/// file to a quant label. A match only counts if the characters immediately
/// before and after it aren't alphanumeric (and not `_`, since quant names are
/// themselves `_`-delimited, e.g. the boundary between "Q2_K" and "_L").
fn matches_quant(filename: &str, quant: &str) -> bool {
    let upper = filename.to_uppercase();
    upper.match_indices(quant).any(|(start, _)| {
        let before_ok = upper[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_ascii_alphanumeric());
        let end = start + quant.len();
        let after_ok = upper[end..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
        before_ok && after_ok
    })
}

/// Fetch GGUF models from HuggingFace that fit the hardware's memory budget,
/// ranked by downloads desc (HF's default top-downloaded GGUF repos).
pub(crate) async fn fetch_recommended_models(
    hw: &HardwareInfo,
    client: &reqwest::Client,
) -> Result<Vec<HfModel>> {
    let candidates = fetch_candidates(client, None).await?;
    filter_to_fitting(candidates, hw, client).await
}

/// Fetch GGUF models matching a free-text search term (HF's `search` param,
/// e.g. a model family name) that fit the hardware's memory budget. Lets a
/// user pull up a specific model directly instead of relying on it cracking
/// the download-ranked candidate pool `fetch_recommended_models` draws from —
/// a model whose downloads are split across many separate quantizer repos
/// (a single family quantized independently by several different users) may
/// never individually rank high enough to appear there.
pub(crate) async fn search_models(
    query: &str,
    hw: &HardwareInfo,
    client: &reqwest::Client,
) -> Result<Vec<HfModel>> {
    let candidates = fetch_candidates(client, Some(query)).await?;
    filter_to_fitting(candidates, hw, client).await
}

/// Query HuggingFace for candidate GGUF repos (ids/downloads only — sizes
/// require a follow-up per-model request, see `fetch_model_details`), sorted
/// by downloads desc. `search`, if given, narrows to HF's free-text match
/// (e.g. a model family name) instead of the unfiltered top-downloaded list.
async fn fetch_candidates(client: &reqwest::Client, search: Option<&str>) -> Result<Vec<HfModel>> {
    // Note: HF's `library` query param does *not* filter by GGUF — it silently
    // ignores unrecognized library names and returns generic top-downloaded
    // models. `filter=gguf` (matches the `gguf` tag) is the correct param.
    let url = "https://huggingface.co/api/models";
    let mut req = client.get(url).query(&[
        ("filter", "gguf"),
        ("sort", "downloads"),
        ("direction", "-1"),
        ("limit", "150"),
    ]);
    if let Some(q) = search {
        req = req.query(&[("search", q)]);
    }
    let resp = req
        .header("User-Agent", "llama-tune/0.1")
        .send()
        .await
        .context("failed to reach Hugging Face models API")?
        .error_for_status()
        .context("Hugging Face models API returned an error")?;

    resp.json()
        .await
        .context("failed to parse Hugging Face models API response")
}

/// Fetch real file sizes for `candidates` and filter to those with at least
/// one quant that fits within ~90% of the hardware's memory budget.
async fn filter_to_fitting(
    candidates: Vec<HfModel>,
    hw: &HardwareInfo,
    client: &reqwest::Client,
) -> Result<Vec<HfModel>> {
    let mem_limit = hw.model_memory_bytes();

    // HuggingFace's model search/list API never includes sibling file sizes —
    // not even with `full=true` — so `fetch_candidates` only gets us model
    // ids; actual GGUF sizes require this follow-up per-model request.
    // Fetch them bounded to a handful of requests in flight at once so we
    // don't hammer the API.
    let detailed: Vec<HfModel> = stream::iter(candidates)
        .map(|m| {
            let client = client.clone();
            async move { fetch_model_details(&client, &m.model_id).await.unwrap_or(m) }
        })
        .buffer_unordered(8)
        .collect()
        .await;

    // Filter to models with at least one quant that fits within ~90% of
    // available memory. Checked against the *smallest* available quant, not
    // the sum of every quant variant the repo offers — best_quant_for_memory
    // always falls back to the smallest one, so that's the true "can this
    // run at all" threshold; summing every variant (e.g. Q2_K through fp16)
    // wildly overestimates the footprint of the single file that would
    // actually be downloaded.
    let usable = (mem_limit as f64 * 0.90) as u64;
    let filtered: Vec<HfModel> = detailed
        .into_iter()
        .filter(|m| {
            match m.quant_options().iter().map(|(_, size)| *size).min() {
                Some(min_size) => min_size <= usable,
                None => true, // no recognized quant sizes — don't exclude on unknown info
            }
        })
        .take(20)
        .collect();

    Ok(filtered)
}

/// Fetch full metadata for a fixed set of already-installed model ids (no
/// fit-filtering — an installed model stays visible even if it no longer fits
/// the current hardware budget, since it's already on disk). Unlike
/// `filter_to_fitting`, a failed per-model lookup (deleted repo, offline) still
/// produces a minimal `HfModel` carrying just the id, so a local install never
/// disappears from the Models tab just because HuggingFace couldn't be reached.
pub(crate) async fn fetch_installed_models(
    model_ids: Vec<String>,
    client: &reqwest::Client,
) -> Vec<HfModel> {
    let mut models: Vec<HfModel> = stream::iter(model_ids)
        .map(|id| {
            let client = client.clone();
            async move {
                fetch_model_details(&client, &id).await.unwrap_or(HfModel {
                    model_id: id,
                    ..Default::default()
                })
            }
        })
        .buffer_unordered(8)
        .collect()
        .await;
    models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    models
}

/// Fetch a single model's full details, including sibling file sizes — only
/// available via this per-model endpoint with `blobs=true` (the list/search
/// endpoint used above never returns sizes, regardless of `full=true`).
async fn fetch_model_details(client: &reqwest::Client, model_id: &str) -> Result<HfModel> {
    let url = format!("https://huggingface.co/api/models/{model_id}");
    let resp = client
        .get(&url)
        .query(&[("blobs", "true")])
        .header("User-Agent", "llama-tune/0.1")
        .send()
        .await
        .with_context(|| format!("failed to reach Hugging Face for model `{model_id}`"))?
        .error_for_status()
        .with_context(|| format!("Hugging Face returned an error for model `{model_id}`"))?;
    resp.json()
        .await
        .with_context(|| format!("failed to parse Hugging Face response for model `{model_id}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_with_siblings(
        model_id: &str,
        tags: Option<Vec<&str>>,
        files: &[(&str, u64)],
    ) -> HfModel {
        HfModel {
            model_id: model_id.to_string(),
            tags: tags.map(|t| t.into_iter().map(String::from).collect()),
            siblings: Some(
                files
                    .iter()
                    .map(|(name, size)| HfSibling {
                        rfilename: name.to_string(),
                        size: Some(*size),
                    })
                    .collect(),
            ),
            ..Default::default()
        }
    }

    #[test]
    fn matches_quant_exact_boundaries_only() {
        assert!(matches_quant("model-Q4_K_M.gguf", "Q4_K_M"));
        assert!(matches_quant("model.q4_k_m.gguf", "Q4_K_M"));
        // Q2_K must not match inside Q2_K_L or Q2_K_S.
        assert!(!matches_quant("model-Q2_K_L.gguf", "Q2_K"));
        assert!(!matches_quant("model-Q2_K_S.gguf", "Q2_K"));
        assert!(matches_quant("model-Q2_K.gguf", "Q2_K"));
        // Q4_0 must not match inside Q4_0_4_4/Q4_0_4_8.
        assert!(!matches_quant("model-Q4_0_4_4.gguf", "Q4_0"));
        assert!(matches_quant("model-Q4_0.gguf", "Q4_0"));
    }

    #[test]
    fn quant_size_bytes_looks_up_matching_sibling() {
        let m = model_with_siblings("org/repo", None, &[("repo-Q4_K_M.gguf", 4_000_000_000)]);
        assert_eq!(m.quant_size_bytes("Q4_K_M"), Some(4_000_000_000));
        assert_eq!(m.quant_size_bytes("Q8_0"), None);
    }

    #[test]
    fn best_quant_for_memory_picks_highest_that_fits() {
        let m = model_with_siblings(
            "org/repo",
            None,
            &[
                ("repo-Q4_K_M.gguf", 4_000_000_000),
                ("repo-Q6_K.gguf", 6_000_000_000),
                ("repo-Q8_0.gguf", 8_000_000_000),
            ],
        );
        assert_eq!(m.best_quant_for_memory(7_000_000_000), Some("Q6_K"));
        assert_eq!(m.best_quant_for_memory(9_000_000_000), Some("Q8_0"));
    }

    #[test]
    fn best_quant_for_memory_falls_back_to_smallest_when_none_fit() {
        let m = model_with_siblings(
            "org/repo",
            None,
            &[
                ("repo-Q4_K_M.gguf", 4_000_000_000),
                ("repo-Q8_0.gguf", 8_000_000_000),
            ],
        );
        assert_eq!(m.best_quant_for_memory(1_000_000_000), Some("Q4_K_M"));
    }

    #[test]
    fn best_quant_for_memory_none_when_no_recognized_quants() {
        let m = model_with_siblings("org/repo", None, &[("repo-readme.md", 1_000)]);
        assert_eq!(m.best_quant_for_memory(8_000_000_000), None);
    }

    #[test]
    fn estimated_layers_matches_by_size_in_model_id() {
        let m = model_with_siblings("org/Llama-70B-GGUF", None, &[]);
        assert_eq!(m.estimated_layers(), 80);
        let m = model_with_siblings("org/model-13b", None, &[]);
        assert_eq!(m.estimated_layers(), 40);
        let m = model_with_siblings("org/model-7b", None, &[]);
        assert_eq!(m.estimated_layers(), 32);
        let m = model_with_siblings("org/model-1b", None, &[]);
        assert_eq!(m.estimated_layers(), 16);
    }

    #[test]
    fn estimated_layers_matches_by_tags_when_id_has_no_hint() {
        let m = model_with_siblings("org/some-model", Some(vec!["34b", "gguf"]), &[]);
        assert_eq!(m.estimated_layers(), 60);
    }

    #[test]
    fn estimated_layers_falls_back_when_no_size_hint() {
        let m = model_with_siblings("org/mystery-model", None, &[]);
        assert_eq!(m.estimated_layers(), 32);
    }

    #[test]
    fn download_url_for_builds_resolve_url() {
        let m = model_with_siblings("org/repo", None, &[("repo-Q4_K_M.gguf", 4_000_000_000)]);
        assert_eq!(
            m.download_url_for("Q4_K_M"),
            Some("https://huggingface.co/org/repo/resolve/main/repo-Q4_K_M.gguf".to_string())
        );
    }

    #[test]
    fn download_url_for_none_when_quant_absent() {
        let m = model_with_siblings("org/repo", None, &[("repo-Q4_K_M.gguf", 4_000_000_000)]);
        assert_eq!(m.download_url_for("Q8_0"), None);
    }

    #[test]
    fn quant_options_orders_low_to_high_quality() {
        let m = model_with_siblings(
            "org/repo",
            None,
            &[
                ("repo-Q8_0.gguf", 8_000_000_000),
                ("repo-Q4_K_M.gguf", 4_000_000_000),
            ],
        );
        let options = m.quant_options();
        assert_eq!(
            options,
            vec![("Q4_K_M", 4_000_000_000), ("Q8_0", 8_000_000_000)]
        );
    }
}
