//! Scans the local filesystem (Hugging Face hub cache, plus an optional
//! configured models directory) to find already-downloaded GGUF models.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use crate::models::HfModel;

/// Determine which of the given models already have a local GGUF copy,
/// checking both the standard Hugging Face hub cache and an optional
/// user-configured local models directory (`LLAMA_TUNE_MODELS_DIR`). Maps
/// `model_id` -> path to the local `.gguf` file, so callers can both show an
/// "installed" badge and launch llama.cpp directly against that file.
///
/// This does blocking filesystem I/O; call it via `spawn_blocking`.
pub(crate) fn scan_installed_candidates(models: &[HfModel]) -> HashMap<String, PathBuf> {
    let hf_cache = hf_hub_cache_dir();
    let local_files = std::env::var("LLAMA_TUNE_MODELS_DIR")
        .ok()
        .map(|dir| list_gguf_files(Path::new(&dir)))
        .unwrap_or_default();

    models
        .iter()
        .filter_map(|m| {
            let path = hf_cache
                .as_deref()
                .and_then(|cache| hf_repo_cached(cache, &m.model_id))
                .or_else(|| matches_local_file(&m.model_id, &local_files))?;
            Some((m.model_id.clone(), path))
        })
        .collect()
}

/// Find every already-downloaded model in the Hugging Face hub cache,
/// independent of any particular candidate list — unlike `scan_installed_candidates`,
/// which only checks whether specific known models are cached. This is what
/// lets the Models tab show installed models even when they don't appear in
/// the current search/recommended results (fell out of the download-rank
/// cutoff, filtered by a search term, etc). Doesn't cover
/// `LLAMA_TUNE_MODELS_DIR`, since matching an arbitrary local filename back
/// to a model id requires a candidate id to normalize and compare against.
pub(crate) fn scan_hub_cache() -> HashMap<String, PathBuf> {
    let Some(cache) = hf_hub_cache_dir() else {
        return HashMap::new();
    };
    let Ok(entries) = fs::read_dir(&cache) else {
        return HashMap::new();
    };

    entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let rest = name.to_str()?.strip_prefix("models--")?;
            let (org, repo) = rest.split_once("--")?;
            let model_id = format!("{org}/{repo}");
            let path = pick_model_gguf(&entry.path().join("snapshots"))?;
            Some((model_id, path))
        })
        .collect()
}

/// Pick the file most likely to be the actual model weights out of every
/// `.gguf` file across a repo's cached snapshots. Repos with multimodal
/// variants ship a `mmproj-*.gguf` file (the vision/audio projector) alongside
/// the real model — it's meant to be passed via `--mmproj`, not `-m`, and
/// isn't a runnable model on its own, so it's excluded whenever a
/// non-projector candidate exists. Among what's left, the largest file wins:
/// actual model weights are always far bigger than a projector, and this
/// doesn't require knowing which specific quant was originally downloaded.
fn pick_model_gguf(snapshots: &Path) -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .flat_map(|snapshot| {
            fs::read_dir(snapshot.path())
                .ok()
                .into_iter()
                .flatten()
                .flatten()
        })
        .map(|f| f.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .collect();

    let is_mmproj = |p: &PathBuf| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.to_lowercase().contains("mmproj"))
    };
    let largest = |files: &[PathBuf]| -> Option<PathBuf> {
        files
            .iter()
            .max_by_key(|p| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .cloned()
    };

    let non_projector: Vec<PathBuf> = candidates
        .iter()
        .filter(|p| !is_mmproj(p))
        .cloned()
        .collect();
    largest(&non_projector).or_else(|| largest(&candidates))
}

/// Resolve the Hugging Face hub cache directory the same way `huggingface_hub` does:
/// `HUGGINGFACE_HUB_CACHE` if set, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`.
/// Exposed to `download.rs`, which writes new downloads into this same directory.
pub(crate) fn hf_hub_cache_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return Some(PathBuf::from(dir));
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(home).join("hub"));
    }
    dirs::home_dir().map(|h| h.join(".cache").join("huggingface").join("hub"))
}

/// A model repo is considered cached if its `models--org--name` folder has a
/// snapshot containing at least one `.gguf` file; returns that file's path
/// (see `pick_model_gguf` for how it's chosen when there's more than one).
fn hf_repo_cached(cache_dir: &Path, model_id: &str) -> Option<PathBuf> {
    let folder = format!("models--{}", model_id.replace('/', "--"));
    pick_model_gguf(&cache_dir.join(folder).join("snapshots"))
}

fn list_gguf_files(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Heuristic: does any local `.gguf` file look like it belongs to this model
/// repo? Matches on the repo's short name (the part after the last `/`) with
/// separators and casing stripped, since download filenames rarely match the
/// repo id exactly (e.g. `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`).
fn matches_local_file(model_id: &str, files: &[PathBuf]) -> Option<PathBuf> {
    let short_name = model_id.rsplit('/').next().unwrap_or(model_id);
    let normalized_name = normalize(short_name);
    // Avoid matching on names so short they'd produce false positives.
    if normalized_name.len() < 4 {
        return None;
    }
    files
        .iter()
        .find(|f| {
            f.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| normalize(n).contains(&normalized_name))
        })
        .cloned()
}

fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_file(dir: &Path, name: &str, size: usize) {
        fs::write(dir.join(name), vec![0u8; size]).unwrap();
    }

    #[test]
    fn pick_model_gguf_selects_largest_across_snapshots() {
        let root = tempdir().unwrap();
        let snap_a = root.path().join("snap-a");
        let snap_b = root.path().join("snap-b");
        fs::create_dir_all(&snap_a).unwrap();
        fs::create_dir_all(&snap_b).unwrap();
        write_file(&snap_a, "small.gguf", 100);
        write_file(&snap_b, "large.gguf", 500);

        let picked = pick_model_gguf(root.path()).unwrap();
        assert_eq!(picked.file_name().unwrap(), "large.gguf");
    }

    #[test]
    fn pick_model_gguf_excludes_mmproj_when_alternative_exists() {
        let root = tempdir().unwrap();
        let snap = root.path().join("snap");
        fs::create_dir_all(&snap).unwrap();
        // mmproj file is deliberately the larger of the two, so a naive
        // largest-file-wins pick without mmproj filtering would choose wrong.
        write_file(&snap, "mmproj-model.gguf", 900);
        write_file(&snap, "model-Q4_K_M.gguf", 400);

        let picked = pick_model_gguf(root.path()).unwrap();
        assert_eq!(picked.file_name().unwrap(), "model-Q4_K_M.gguf");
    }

    #[test]
    fn pick_model_gguf_falls_back_to_mmproj_when_only_option() {
        let root = tempdir().unwrap();
        let snap = root.path().join("snap");
        fs::create_dir_all(&snap).unwrap();
        write_file(&snap, "mmproj-model.gguf", 900);

        let picked = pick_model_gguf(root.path()).unwrap();
        assert_eq!(picked.file_name().unwrap(), "mmproj-model.gguf");
    }

    #[test]
    fn pick_model_gguf_none_when_no_gguf_files() {
        let root = tempdir().unwrap();
        let snap = root.path().join("snap");
        fs::create_dir_all(&snap).unwrap();
        write_file(&snap, "readme.md", 10);

        assert!(pick_model_gguf(root.path()).is_none());
    }

    #[test]
    fn hf_repo_cached_finds_gguf_in_snapshots() {
        let cache = tempdir().unwrap();
        let snapshots = cache
            .path()
            .join("models--org--repo-name")
            .join("snapshots")
            .join("abc123");
        fs::create_dir_all(&snapshots).unwrap();
        write_file(&snapshots, "weights.gguf", 200);

        let found = hf_repo_cached(cache.path(), "org/repo-name").unwrap();
        assert_eq!(found.file_name().unwrap(), "weights.gguf");
    }

    #[test]
    fn hf_repo_cached_none_when_repo_not_present() {
        let cache = tempdir().unwrap();
        fs::create_dir_all(cache.path()).unwrap();
        assert!(hf_repo_cached(cache.path(), "org/missing-repo").is_none());
    }

    #[test]
    fn list_gguf_files_is_case_insensitive_and_filters_extension() {
        let dir = tempdir().unwrap();
        write_file(dir.path(), "a.gguf", 10);
        write_file(dir.path(), "b.GGUF", 10);
        write_file(dir.path(), "c.bin", 10);

        let mut names: Vec<String> = list_gguf_files(dir.path())
            .into_iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert_eq!(names, vec!["a.gguf".to_string(), "b.GGUF".to_string()]);
    }

    #[test]
    fn matches_local_file_finds_normalized_substring_match() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("Meta-Llama-3-8B-Instruct.Q4_K_M.gguf");
        fs::write(&file, b"x").unwrap();
        let files = vec![file.clone()];

        let found = matches_local_file("meta-llama/Meta-Llama-3-8B-Instruct", &files);
        assert_eq!(found, Some(file));
    }

    #[test]
    fn matches_local_file_none_when_short_name_too_short() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("ab-model.gguf");
        fs::write(&file, b"x").unwrap();
        let files = vec![file];

        // Short name "ab" normalizes to <4 chars, so it should short-circuit
        // to None rather than risk a false-positive substring match.
        assert_eq!(matches_local_file("org/ab", &files), None);
    }

    #[test]
    fn matches_local_file_none_when_no_file_matches() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("unrelated-file.gguf");
        fs::write(&file, b"x").unwrap();
        let files = vec![file];

        assert_eq!(matches_local_file("org/some-other-model", &files), None);
    }

    #[test]
    fn normalize_strips_non_alphanumeric_and_lowercases() {
        assert_eq!(normalize("Meta-Llama_3.8B!"), "metallama38b");
    }
}
