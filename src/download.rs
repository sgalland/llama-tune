//! Streams a GGUF file to disk into the Hugging Face hub cache layout, with
//! progress reporting and cooperative cancellation.

use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::installed::hf_hub_cache_dir;

/// Download `url` (a GGUF file belonging to `model_id`) into the standard
/// Hugging Face hub cache layout (`models--org--name/snapshots/download/<filename>`),
/// so `installed::scan_installed_candidates` finds it exactly like an official download.
/// Streams to disk in chunks, calling `on_progress(downloaded, total)` after
/// each one — a plain callback rather than a channel tied to the caller's
/// message type, so this module doesn't need to know anything about `AppMsg`.
/// Checks `cancel` after every chunk and, if set, deletes the partial file
/// and returns an error (the caller distinguishes cancellation from a real
/// failure by checking the same flag, since it holds a clone).
pub(crate) async fn download_to_cache(
    client: &reqwest::Client,
    model_id: &str,
    url: &str,
    mut on_progress: impl FnMut(u64, Option<u64>),
    cancel: Arc<AtomicBool>,
) -> Result<PathBuf> {
    let cache_dir =
        hf_hub_cache_dir().context("could not determine Hugging Face cache directory")?;
    let filename = url.rsplit('/').next().unwrap_or("model.gguf").to_string();
    let folder = format!("models--{}", model_id.replace('/', "--"));
    let dest_dir = cache_dir.join(folder).join("snapshots").join("download");
    tokio::fs::create_dir_all(&dest_dir)
        .await
        .with_context(|| format!("failed to create {}", dest_dir.display()))?;

    let dest_path = dest_dir.join(&filename);
    let tmp_path = dest_dir.join(format!("{filename}.part"));

    // The shared client has a short default timeout sized for JSON API calls;
    // a multi-gigabyte model download needs its own much longer budget, or
    // any real download would abort partway through with a timeout error.
    let resp = client
        .get(url)
        .header("User-Agent", "llama-tune/0.1")
        .timeout(Duration::from_secs(6 * 60 * 60))
        .send()
        .await?
        .error_for_status()?;
    let total = resp.content_length();

    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("failed to create {}", tmp_path.display()))?;
    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::Relaxed) {
            drop(file);
            let _ = tokio::fs::remove_file(&tmp_path).await;
            bail!("cancelled");
        }
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        on_progress(downloaded, total);
    }
    file.flush().await?;
    drop(file);

    tokio::fs::rename(&tmp_path, &dest_path).await?;
    Ok(dest_path)
}
