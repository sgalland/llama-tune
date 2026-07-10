//! Terminal ownership, the tokio event loop, and the `mpsc` channel wiring
//! that carries results from spawned async tasks back into `App` state.

mod app;
mod config;
mod download;
mod hardware;
mod installed;
mod launch;
mod models;
mod params;
mod ui;

use std::{
    collections::HashMap,
    io,
    path::PathBuf,
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use anyhow::Result;
use app::{App, AppTab, LoadState};
use crossterm::{
    event::{
        DisableMouseCapture, EnableMouseCapture, Event, EventStream, KeyCode, KeyEvent,
        KeyEventKind, KeyModifiers,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::StreamExt;
use models::HfModel;
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc;

#[derive(Debug)]
enum AppMsg {
    HardwareReady(hardware::HardwareInfo),
    HardwareError(String),
    ModelsReady(Vec<models::HfModel>),
    ModelsError(String),
    InstalledScanned(HashMap<String, PathBuf>),
    InstalledModelsReady(Vec<models::HfModel>),
    DownloadProgress {
        model_id: String,
        downloaded: u64,
        total: Option<u64>,
    },
    DownloadComplete {
        model: HfModel,
        path: PathBuf,
    },
    DownloadError {
        model_id: String,
        error: String,
    },
    DownloadCancelled {
        model_id: String,
    },
    LaunchExitedEarly {
        model_id: String,
        code: Option<i32>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // ── Terminal setup ────────────────────────────────────────────────────────
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run(&mut terminal).await;

    // ── Restore terminal ──────────────────────────────────────────────────────
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

async fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut app = App::new();
    let (tx, mut rx) = mpsc::channel::<AppMsg>(16);

    // Spawn hardware detection
    spawn_hardware_detect(tx.clone());

    // HTTP client (shared across model fetches)
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    // Independent scan of already-installed models, so they show up in the
    // Models tab regardless of hardware detection or the active search —
    // unlike the per-search-list scan below (AppMsg::ModelsReady), this
    // doesn't wait on either.
    spawn_installed_scan(tx.clone(), http.clone());

    // `crossterm::event::poll`/`read` block the calling OS thread; awaiting
    // `EventStream` instead lets this task yield to the tokio runtime while
    // waiting for a keypress, so a blocked terminal read can't stall other
    // spawned work (downloads, model fetches, etc.) sharing the runtime.
    let mut events = EventStream::new();

    loop {
        terminal.draw(|f| ui::draw(f, &app))?;

        tokio::select! {
            maybe_event = events.next() => {
                if let Some(Ok(Event::Key(key))) = maybe_event {
                    // Windows reports both press and release key events
                    // (unlike Unix terminals); without this, every keypress
                    // fires twice.
                    if key.kind != KeyEventKind::Release {
                        // Dismiss any prior launch/download status message on
                        // the next keypress, so the status bar reverts to
                        // showing the keybinding hints instead of leaving a
                        // stale message on screen forever. Whatever the
                        // keypress does below (if anything) can still set a
                        // fresh one right after this.
                        app.launch_status = None;
                        handle_key_event(&mut app, key, &tx, &http);
                    }
                }
            }
            Some(msg) = rx.recv() => {
                handle_app_msg(&mut app, msg, &tx, &http);
                // Drain any further already-queued messages without waiting,
                // so a burst (e.g. several download-progress ticks) is fully
                // applied before the next redraw instead of trickling in one
                // per frame.
                while let Ok(msg) = rx.try_recv() {
                    handle_app_msg(&mut app, msg, &tx, &http);
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

/// Apply one message from the async task channel to `app` state.
fn handle_app_msg(app: &mut App, msg: AppMsg, tx: &mpsc::Sender<AppMsg>, http: &reqwest::Client) {
    match msg {
        AppMsg::HardwareReady(hw) => {
            // Kick off model fetch now that we know the hardware,
            // re-running whatever search (if any) is currently active
            // so refresh ([r]) doesn't drop it.
            spawn_model_fetch(
                hw.clone(),
                app.search_query.clone(),
                tx.clone(),
                http.clone(),
            );
            app.hw_state = LoadState::Ready(hw);
            app.recompute_params();
        }
        AppMsg::HardwareError(e) => {
            app.hw_state = LoadState::Error(e);
        }
        AppMsg::ModelsReady(m) => {
            app.model_list_state.select(Some(0));
            // Scan for already-installed copies now that we know which models exist
            let tx3 = tx.clone();
            let m2 = m.clone();
            tokio::spawn(async move {
                let installed = tokio::task::spawn_blocking(move || installed::scan_installed_candidates(&m2))
                    .await
                    .unwrap_or_default();
                let _ = tx3.send(AppMsg::InstalledScanned(installed)).await;
            });
            app.models_state = LoadState::Ready(m);
            app.recompute_params();
        }
        AppMsg::ModelsError(e) => {
            app.models_state = LoadState::Error(e);
        }
        AppMsg::InstalledScanned(set) => {
            // Merged, not replaced: this arrives from two independent
            // sources (the full hub-cache scan and the per-search-list
            // scan below), so one shouldn't clobber the other's finds.
            app.installed.extend(set);
        }
        AppMsg::InstalledModelsReady(models) => {
            app.installed_models = models;
            app.recompute_params();
        }
        AppMsg::DownloadProgress {
            model_id,
            downloaded,
            total,
        } => {
            // Update in place rather than replacing wholesale, so the
            // existing `cancel` flag (checked by the in-flight download
            // task) stays the same shared `Arc` the UI can still signal.
            if let Some(dl) = &mut app.download {
                if dl.model_id == model_id {
                    dl.downloaded = downloaded;
                    dl.total = total;
                }
            }
        }
        AppMsg::DownloadComplete { model, path } => {
            app.download = None;
            app.installed.insert(model.model_id.clone(), path.clone());
            if let Some(child) = app.launch_now(&model, &path) {
                monitor_launch(tx.clone(), model.model_id.clone(), child);
            }
        }
        AppMsg::DownloadError { model_id, error } => {
            app.download = None;
            app.launch_status = Some(format!("Download failed for {model_id}: {error}"));
        }
        AppMsg::DownloadCancelled { model_id } => {
            app.download = None;
            app.launch_status = Some(format!("Download of {model_id} cancelled."));
        }
        AppMsg::LaunchExitedEarly { model_id, code } => {
            let code_str = code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            app.launch_status = Some(format!(
                "llama.cpp for {model_id} exited immediately (code {code_str}) — check its console window for details."
            ));
        }
    }
}

/// Apply one key event to `app` state, dispatching to the active edit-mode
/// input buffer (Settings path / Models search box) or the global keybindings.
fn handle_key_event(
    app: &mut App,
    key: KeyEvent,
    tx: &mpsc::Sender<AppMsg>,
    http: &reqwest::Client,
) {
    if app.current_tab == AppTab::Settings && app.settings_editing {
        // While editing, keystrokes go into the input buffer instead
        // of triggering the global keybindings below.
        match key.code {
            KeyCode::Enter => app.confirm_editing_path(),
            KeyCode::Esc => app.cancel_editing_path(),
            KeyCode::Backspace => {
                app.settings_input.pop();
            }
            KeyCode::Char(c) => app.settings_input.push(c),
            _ => {}
        }
        return;
    }

    if app.current_tab == AppTab::Models && app.search_editing {
        match key.code {
            KeyCode::Enter => {
                let query = app.search_input.trim().to_string();
                app.search_editing = false;
                app.search_input.clear();
                // An empty query clears search and reverts to the
                // hardware-ranked recommended list.
                app.search_query = if query.is_empty() { None } else { Some(query) };
                if let LoadState::Ready(hw) = &app.hw_state {
                    app.models_state = LoadState::Loading;
                    app.model_list_state.select(Some(0));
                    spawn_model_fetch(
                        hw.clone(),
                        app.search_query.clone(),
                        tx.clone(),
                        http.clone(),
                    );
                } else {
                    app.launch_status = Some("Hardware not detected yet.".to_string());
                }
            }
            KeyCode::Esc => app.cancel_search_editing(),
            KeyCode::Backspace => {
                app.search_input.pop();
            }
            KeyCode::Char(c) => app.search_input.push(c),
            _ => {}
        }
        return;
    }

    match (key.code, key.modifiers) {
        (KeyCode::Char('q'), _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
            app.should_quit = true;
        }
        (KeyCode::Char('1'), _) => app.current_tab = AppTab::Hardware,
        (KeyCode::Char('2'), _) => app.current_tab = AppTab::Parameters,
        (KeyCode::Char('3'), _) => app.current_tab = AppTab::Models,
        (KeyCode::Char('4'), _) => app.current_tab = AppTab::Settings,
        (KeyCode::Tab, _) => {
            app.current_tab = match app.current_tab {
                AppTab::Hardware => AppTab::Parameters,
                AppTab::Parameters => AppTab::Models,
                AppTab::Models => AppTab::Settings,
                AppTab::Settings => AppTab::Hardware,
            };
        }
        (KeyCode::Down, _) | (KeyCode::Char('j'), _) => {
            if app.current_tab == AppTab::Models {
                app.next_model();
            }
        }
        (KeyCode::Up, _) | (KeyCode::Char('k'), _) => {
            if app.current_tab == AppTab::Models {
                app.prev_model();
            }
        }
        (KeyCode::Char('l'), _) => {
            if app.current_tab == AppTab::Models {
                if let Some(model) = app.selected_model().cloned() {
                    launch_or_download(app, tx, http, model);
                } else {
                    app.launch_status = Some("No model selected.".to_string());
                }
            }
        }
        (KeyCode::Char('L'), _) => {
            relaunch_last(app, tx, http);
        }
        (KeyCode::Char('/'), _) => {
            if app.current_tab == AppTab::Models {
                app.start_search();
            }
        }
        (KeyCode::Char('x'), _) => {
            if let Some(dl) = &app.download {
                dl.cancel.store(true, Ordering::Relaxed);
            }
        }
        (KeyCode::Char('e'), _) | (KeyCode::Enter, _) => {
            if app.current_tab == AppTab::Settings {
                app.start_editing_path();
            }
        }
        (KeyCode::Char('r'), _) => {
            // Refresh: re-detect hardware, models, and installed state.
            // Installed state is cleared up front (rather than
            // merged) so a model uninstalled since the last scan
            // doesn't linger as a stale "installed" entry forever.
            app.hw_state = LoadState::Loading;
            app.models_state = LoadState::Loading;
            app.installed.clear();
            app.installed_models.clear();
            app.params = None;
            spawn_hardware_detect(tx.clone());
            spawn_installed_scan(tx.clone(), http.clone());
        }
        _ => {}
    }
}

/// Spawn hardware detection on a blocking thread (NVML/DXGI/sysfs calls are
/// synchronous) and report the result back over `tx`. Shared by the initial
/// detection and the [r] refresh so both handle the panic case identically.
fn spawn_hardware_detect(tx: mpsc::Sender<AppMsg>) {
    tokio::spawn(async move {
        match tokio::task::spawn_blocking(hardware::detect).await {
            Ok(Ok(hw)) => {
                let _ = tx.send(AppMsg::HardwareReady(hw)).await;
            }
            Ok(Err(e)) => {
                let _ = tx.send(AppMsg::HardwareError(format!("{e:#}"))).await;
            }
            Err(e) => {
                let _ = tx
                    .send(AppMsg::HardwareError(format!(
                        "hardware detection task panicked: {e}"
                    )))
                    .await;
            }
        }
    });
}

/// Fetch models for the Models tab: `query` selects a free-text search
/// (`models::search_models`) instead of the default hardware-ranked
/// recommended list (`models::fetch_recommended_models`) when `Some`.
/// Shared by the initial hardware-ready fetch, [r] refresh, and search
/// submission so all three land on the same `AppMsg::ModelsReady/Error`.
fn spawn_model_fetch(
    hw: hardware::HardwareInfo,
    query: Option<String>,
    tx: mpsc::Sender<AppMsg>,
    http: reqwest::Client,
) {
    tokio::spawn(async move {
        let result = match &query {
            Some(q) => models::search_models(q, &hw, &http).await,
            None => models::fetch_recommended_models(&hw, &http).await,
        };
        match result {
            Ok(m) => {
                let _ = tx.send(AppMsg::ModelsReady(m)).await;
            }
            Err(e) => {
                let _ = tx.send(AppMsg::ModelsError(format!("{e:#}"))).await;
            }
        }
    });
}

/// Scan the Hugging Face hub cache for already-installed models (independent
/// of any search/recommended list — see `installed::scan_hub_cache`) and
/// fetch their full metadata, so the Models tab can show them right away.
/// Sends `AppMsg::InstalledScanned` first (id -> path only, fast, no network)
/// and `AppMsg::InstalledModelsReady` once metadata fetches complete.
fn spawn_installed_scan(tx: mpsc::Sender<AppMsg>, http: reqwest::Client) {
    tokio::spawn(async move {
        let installed = tokio::task::spawn_blocking(installed::scan_hub_cache)
            .await
            .unwrap_or_default();
        let ids: Vec<String> = installed.keys().cloned().collect();
        let _ = tx.send(AppMsg::InstalledScanned(installed)).await;

        let models = models::fetch_installed_models(ids, &http).await;
        let _ = tx.send(AppMsg::InstalledModelsReady(models)).await;
    });
}

/// Poll `child` for up to a few seconds after launch and, if it exits
/// unsuccessfully within that grace period, report `AppMsg::LaunchExitedEarly`.
/// `spawn_llama_cpp` only confirms the OS-level process start succeeded — a
/// launch that succeeds there but then dies moments later (corrupt GGUF, OOM,
/// an arg some specific build rejects) would otherwise still show "Launched"
/// with no indication anything went wrong. A still-running process after the
/// grace period is assumed healthy and isn't polled further.
fn monitor_launch(tx: mpsc::Sender<AppMsg>, model_id: String, mut child: std::process::Child) {
    tokio::spawn(async move {
        const GRACE_PERIOD: Duration = Duration::from_secs(3);
        const POLL_INTERVAL: Duration = Duration::from_millis(200);

        let mut waited = Duration::ZERO;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        let _ = tx
                            .send(AppMsg::LaunchExitedEarly {
                                model_id,
                                code: status.code(),
                            })
                            .await;
                    }
                    return;
                }
                Ok(None) => {}
                Err(_) => return,
            }
            if waited >= GRACE_PERIOD {
                return;
            }
            tokio::time::sleep(POLL_INTERVAL).await;
            waited += POLL_INTERVAL;
        }
    });
}

/// Launch `model` if it's already installed; otherwise download its
/// recommended-quant GGUF into the Hugging Face cache first (via a spawned
/// task reporting `AppMsg::DownloadProgress`/`Complete`/`Error`), then launch
/// automatically once the download finishes (see `AppMsg::DownloadComplete`).
fn launch_or_download(
    app: &mut App,
    tx: &mpsc::Sender<AppMsg>,
    http: &reqwest::Client,
    model: HfModel,
) {
    if let Some(path) = app.installed.get(&model.model_id).cloned() {
        if let Some(child) = app.launch_now(&model, &path) {
            monitor_launch(tx.clone(), model.model_id.clone(), child);
        }
        return;
    }
    if app.download.is_some() {
        app.launch_status = Some("A download is already in progress.".to_string());
        return;
    }
    // Validate the configured llama.cpp path up front, before committing to a
    // potentially multi-gigabyte, multi-minute download — otherwise a bad path
    // only surfaced as an error after the download finished, which looked like
    // no error at all to anyone who didn't wait that long.
    match &app.config.llama_cpp_path {
        None => {
            app.launch_status =
                Some("Set the llama.cpp path in the Settings tab [4] first.".to_string());
            return;
        }
        Some(exe) => {
            if let Err(e) = launch::resolve_executable(std::path::Path::new(exe)) {
                app.launch_status = Some(format!("Launch failed: {e:#}"));
                return;
            }
        }
    }
    let Some(quant) = app.best_quant_for(&model) else {
        app.launch_status = Some("No suitable quantization found for your hardware.".to_string());
        return;
    };
    let Some(url) = model.download_url_for(quant) else {
        app.launch_status = Some(format!("No {quant} download available for this model."));
        return;
    };

    app.launch_status = Some(format!("Downloading {} ({quant})…", model.model_id));
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    app.download = Some(app::DownloadState {
        model_id: model.model_id.clone(),
        downloaded: 0,
        total: None,
        cancel: cancel.clone(),
    });

    let model_id = model.model_id.clone();
    let http2 = http.clone();
    let tx2 = tx.clone();
    tokio::spawn(async move {
        let progress_tx = tx2.clone();
        let progress_model_id = model_id.clone();
        // A plain sync callback (not the async `tx2.send`) since download.rs
        // doesn't know about `AppMsg` — `try_send` rather than blocking on a
        // full channel, since dropping an intermediate progress update is
        // harmless and a stalled download loop isn't worth avoiding it.
        let report_progress = move |downloaded, total| {
            let _ = progress_tx.try_send(AppMsg::DownloadProgress {
                model_id: progress_model_id.clone(),
                downloaded,
                total,
            });
        };
        match download::download_to_cache(&http2, &model_id, &url, report_progress, cancel.clone())
            .await
        {
            // Carry the already-known `HfModel` through rather than looking it
            // up again via `find_model`/`display_models()` on receipt: right
            // after a fresh download, the model is momentarily excluded from
            // both halves of that merged list (no longer a search/recommended
            // result now that it's "installed", not yet in `installed_models`,
            // which is only populated by a separate, later-arriving async
            // scan) — so a lookup here would silently find nothing and skip
            // the auto-launch below.
            Ok(path) => {
                let _ = tx2.send(AppMsg::DownloadComplete { model, path }).await;
            }
            Err(e) => {
                let msg = if cancel.load(Ordering::Relaxed) {
                    AppMsg::DownloadCancelled { model_id }
                } else {
                    AppMsg::DownloadError {
                        model_id,
                        error: format!("{e:#}"),
                    }
                };
                let _ = tx2.send(msg).await;
            }
        }
    });
}

/// Relaunch the most recently launched model (see `Config::last_launched_model_id`),
/// downloading it again first if it's no longer in the cache. Works from any
/// tab so it's a quick "run it again" shortcut.
fn relaunch_last(app: &mut App, tx: &mpsc::Sender<AppMsg>, http: &reqwest::Client) {
    let Some(model_id) = app.config.last_launched_model_id.clone() else {
        app.launch_status = Some("No previously launched model yet.".to_string());
        return;
    };
    let Some(model) = app.find_model(&model_id).cloned() else {
        app.launch_status = Some(format!(
            "Last launched model ({model_id}) isn't in the current list — refresh with [r]."
        ));
        return;
    };
    launch_or_download(app, tx, http, model);
}
