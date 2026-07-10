//! UI/application state (`App`) plus small mutation methods — no async I/O
//! happens here, only state changes driven by `main.rs`'s event loop.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc},
};

use ratatui::widgets::ListState;

use crate::{
    config::Config,
    hardware::HardwareInfo,
    launch,
    models::HfModel,
    params::{self, LlamaCppParams, ModelRequirements},
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AppTab {
    Hardware,
    Parameters,
    Models,
    Settings,
}

/// Generic loading state wrapper
#[derive(Debug, Clone)]
pub(crate) enum LoadState<T> {
    Loading,
    Ready(T),
    Error(String),
}

/// Progress of an in-flight model download, for display in the Models tab.
#[derive(Debug, Clone)]
pub(crate) struct DownloadState {
    pub(crate) model_id: String,
    pub(crate) downloaded: u64,
    pub(crate) total: Option<u64>,
    /// Set to request cancellation; checked by `download::download_to_cache`
    /// between chunks. Shared (not per-DownloadState-clone-independent) so
    /// setting it from the UI actually reaches the in-flight download task.
    pub(crate) cancel: Arc<AtomicBool>,
}

pub(crate) struct App {
    pub(crate) current_tab: AppTab,
    pub(crate) hw_state: LoadState<HardwareInfo>,
    pub(crate) params: Option<LlamaCppParams>,
    pub(crate) models_state: LoadState<Vec<HfModel>>,
    pub(crate) model_list_state: ListState,
    /// model_id -> local `.gguf` path, for models already downloaded (HF cache
    /// or configured local dir)
    pub(crate) installed: HashMap<String, PathBuf>,
    /// Full metadata for already-installed models, fetched independently of
    /// `models_state` so installed models stay visible in the Models tab
    /// regardless of the active search/recommended list (see `display_models`).
    pub(crate) installed_models: Vec<HfModel>,
    pub(crate) should_quit: bool,
    /// Persisted settings (currently just the llama.cpp executable path)
    pub(crate) config: Config,
    /// Whether the Settings tab is currently editing the llama.cpp path
    pub(crate) settings_editing: bool,
    /// In-progress edit buffer for the llama.cpp path
    pub(crate) settings_input: String,
    /// Result of the last launch attempt, shown in the Models tab
    pub(crate) launch_status: Option<String>,
    /// Progress of an in-flight download, if any (only one at a time)
    pub(crate) download: Option<DownloadState>,
    /// Active Models-tab search term, if any. `None` shows the
    /// hardware-ranked recommended list instead of search results.
    pub(crate) search_query: Option<String>,
    /// Whether the Models tab search box is currently being edited
    pub(crate) search_editing: bool,
    /// In-progress edit buffer for the search box
    pub(crate) search_input: String,
}

impl App {
    pub(crate) fn new() -> Self {
        let mut model_list_state = ListState::default();
        model_list_state.select(Some(0));
        Self {
            current_tab: AppTab::Hardware,
            hw_state: LoadState::Loading,
            params: None,
            models_state: LoadState::Loading,
            model_list_state,
            installed: HashMap::new(),
            installed_models: Vec::new(),
            should_quit: false,
            config: Config::load(),
            settings_editing: false,
            settings_input: String::new(),
            launch_status: None,
            download: None,
            search_query: None,
            search_editing: false,
            search_input: String::new(),
        }
    }

    /// Models tab list contents: already-installed models (fetched
    /// independently of the active search, so they stay visible no matter
    /// what's currently recommended) followed by the current
    /// search/recommended results, skipping any that are already listed as
    /// installed to avoid showing the same model twice. Installed models are
    /// only prepended for the recommended list, not search results — a
    /// search is for finding something new, so an already-installed match
    /// would just be noise.
    pub(crate) fn display_models(&self) -> Vec<&HfModel> {
        let mut list: Vec<&HfModel> = Vec::new();
        if self.search_query.is_none() {
            list.extend(self.installed_models.iter());
        }
        if let LoadState::Ready(models) = &self.models_state {
            list.extend(
                models
                    .iter()
                    .filter(|m| !self.installed.contains_key(&m.model_id)),
            );
        }
        list
    }

    pub(crate) fn next_model(&mut self) {
        let len = self.display_models().len();
        if len == 0 {
            return;
        }
        let i = match self.model_list_state.selected() {
            Some(i) => (i + 1) % len,
            None => 0,
        };
        self.model_list_state.select(Some(i));
        self.recompute_params();
    }

    pub(crate) fn prev_model(&mut self) {
        let len = self.display_models().len();
        if len == 0 {
            return;
        }
        let i = match self.model_list_state.selected() {
            Some(0) | None => len - 1,
            Some(i) => i - 1,
        };
        self.model_list_state.select(Some(i));
        self.recompute_params();
    }

    pub(crate) fn selected_model(&self) -> Option<&HfModel> {
        let idx = self.model_list_state.selected()?;
        self.display_models().into_iter().nth(idx)
    }

    /// Look up a model by id in the combined installed + search/recommended
    /// list, regardless of what's selected in the UI (used for relaunching
    /// the last-launched model).
    pub(crate) fn find_model(&self, model_id: &str) -> Option<&HfModel> {
        self.display_models()
            .into_iter()
            .find(|m| m.model_id == model_id)
    }

    /// Best-quality quant of `model` that fits within the current hardware's
    /// memory budget (VRAM if a GPU is present, else available RAM). `None`
    /// while hardware isn't known yet.
    pub(crate) fn best_quant_for(&self, model: &HfModel) -> Option<&'static str> {
        let LoadState::Ready(hw) = &self.hw_state else {
            return None;
        };
        let budget = (hw.model_memory_bytes() as f64 * 0.90) as u64;
        model.best_quant_for_memory(budget)
    }

    /// Recompute `params` from the current hardware and (if any) the selected
    /// model, so `-ngl` reflects that specific model's layer count/size rather
    /// than always assuming full offload. No-op while hardware isn't ready yet.
    pub(crate) fn recompute_params(&mut self) {
        let LoadState::Ready(hw) = &self.hw_state else {
            return;
        };

        let model_req = self.selected_model().map(|m| {
            let quant = self.best_quant_for(m);
            ModelRequirements {
                file_size_bytes: quant.and_then(|q| m.quant_size_bytes(q)).unwrap_or(0),
                num_layers: m.estimated_layers(),
                quantization: quant.unwrap_or("unknown").to_string(),
            }
        });

        self.params = Some(params::compute(hw, model_req.as_ref()));
    }

    /// Begin editing the Models tab search box, seeding the input buffer with
    /// whatever search is currently active (if any) so it can be tweaked.
    pub(crate) fn start_search(&mut self) {
        self.search_input = self.search_query.clone().unwrap_or_default();
        self.search_editing = true;
    }

    pub(crate) fn cancel_search_editing(&mut self) {
        self.search_editing = false;
        self.search_input.clear();
    }

    /// Begin editing the llama.cpp path in the Settings tab, seeding the input
    /// buffer with whatever's currently configured.
    pub(crate) fn start_editing_path(&mut self) {
        self.settings_input = self.config.llama_cpp_path.clone().unwrap_or_default();
        self.settings_editing = true;
    }

    pub(crate) fn cancel_editing_path(&mut self) {
        self.settings_editing = false;
        self.settings_input.clear();
    }

    /// Persist the edited path to disk and exit editing mode.
    pub(crate) fn confirm_editing_path(&mut self) {
        let trimmed = self.settings_input.trim();
        self.config.llama_cpp_path = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        };
        if let Err(e) = self.config.save() {
            self.launch_status = Some(format!("Failed to save settings: {e:#}"));
        }
        self.settings_editing = false;
        self.settings_input.clear();
    }

    /// Launch llama.cpp for `model`, which must already have a local file at
    /// `path`. Computes params for this specific model rather than trusting
    /// `self.params` (which tracks whatever's selected in the UI, not
    /// necessarily `model` — relaunching the last-launched model may target a
    /// different one). Sets `launch_status` with the outcome and, on success,
    /// remembers `model` as the last-launched one for quick relaunch.
    ///
    /// Returns the spawned child process on success so the caller can watch
    /// for an early exit (a launch that succeeds at the OS level but then
    /// dies moments later, e.g. a corrupt GGUF or an arg some build rejects)
    /// and surface that separately, since this function only knows the
    /// process *started*, not that it kept running.
    pub(crate) fn launch_now(
        &mut self,
        model: &HfModel,
        path: &std::path::Path,
    ) -> Option<std::process::Child> {
        let Some(exe) = self.config.llama_cpp_path.as_deref() else {
            self.launch_status =
                Some("Set the llama.cpp path in the Settings tab [4] first.".to_string());
            return None;
        };
        let LoadState::Ready(hw) = &self.hw_state else {
            self.launch_status = Some("Hardware not detected yet.".to_string());
            return None;
        };

        let quant = self.best_quant_for(model);
        let model_req = ModelRequirements {
            file_size_bytes: quant.and_then(|q| model.quant_size_bytes(q)).unwrap_or(0),
            num_layers: model.estimated_layers(),
            quantization: quant.unwrap_or("unknown").to_string(),
        };
        let params = params::compute(hw, Some(&model_req));

        match launch::spawn_llama_cpp(exe, path, &params) {
            Ok(launch::LaunchedProcess { message, child }) => {
                self.config.last_launched_model_id = Some(model.model_id.clone());
                self.launch_status = Some(match self.config.save() {
                    Ok(()) => message,
                    Err(e) => {
                        format!("{message} (warning: failed to save last-launched model: {e:#})")
                    }
                });
                Some(child)
            }
            Err(e) => {
                self.launch_status = Some(format!("Launch failed: {e:#}"));
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::HardwareInfo;

    // Builds an App via a direct struct literal rather than App::new(), since
    // App::new() calls Config::load() and would read the developer's real
    // config file.
    fn test_app() -> App {
        let mut model_list_state = ListState::default();
        model_list_state.select(Some(0));
        App {
            current_tab: AppTab::Models,
            hw_state: LoadState::Loading,
            params: None,
            models_state: LoadState::Loading,
            model_list_state,
            installed: HashMap::new(),
            installed_models: Vec::new(),
            should_quit: false,
            config: Config::default(),
            settings_editing: false,
            settings_input: String::new(),
            launch_status: None,
            download: None,
            search_query: None,
            search_editing: false,
            search_input: String::new(),
        }
    }

    fn model(id: &str) -> HfModel {
        HfModel {
            model_id: id.to_string(),
            ..Default::default()
        }
    }

    fn ready_hw() -> HardwareInfo {
        HardwareInfo {
            cpu_name: "Test CPU".to_string(),
            cpu_physical_cores: 8,
            cpu_logical_cores: 16,
            total_ram_bytes: 32 * 1_073_741_824,
            available_ram_bytes: 16 * 1_073_741_824,
            gpus: Vec::new(),
        }
    }

    #[test]
    fn display_models_prepends_installed_when_no_search() {
        let mut app = test_app();
        app.installed_models = vec![model("org/installed-a")];
        app.models_state = LoadState::Ready(vec![model("org/recommended-a")]);

        let ids: Vec<&str> = app
            .display_models()
            .iter()
            .map(|m| m.model_id.as_str())
            .collect();
        assert_eq!(ids, vec!["org/installed-a", "org/recommended-a"]);
    }

    #[test]
    fn display_models_dedups_installed_from_recommended_list() {
        let mut app = test_app();
        app.installed_models = vec![model("org/shared")];
        app.installed
            .insert("org/shared".to_string(), PathBuf::from("/tmp/shared.gguf"));
        app.models_state = LoadState::Ready(vec![model("org/shared"), model("org/other")]);

        let ids: Vec<&str> = app
            .display_models()
            .iter()
            .map(|m| m.model_id.as_str())
            .collect();
        assert_eq!(ids, vec!["org/shared", "org/other"]);
    }

    #[test]
    fn display_models_excludes_installed_during_active_search() {
        let mut app = test_app();
        app.installed_models = vec![model("org/installed-a")];
        app.search_query = Some("query".to_string());
        app.models_state = LoadState::Ready(vec![model("org/search-result")]);

        let ids: Vec<&str> = app
            .display_models()
            .iter()
            .map(|m| m.model_id.as_str())
            .collect();
        assert_eq!(ids, vec!["org/search-result"]);
    }

    #[test]
    fn next_model_wraps_around() {
        let mut app = test_app();
        app.models_state = LoadState::Ready(vec![model("a"), model("b")]);
        app.model_list_state.select(Some(1));

        app.next_model();
        assert_eq!(app.model_list_state.selected(), Some(0));
    }

    #[test]
    fn prev_model_wraps_around() {
        let mut app = test_app();
        app.models_state = LoadState::Ready(vec![model("a"), model("b")]);
        app.model_list_state.select(Some(0));

        app.prev_model();
        assert_eq!(app.model_list_state.selected(), Some(1));
    }

    #[test]
    fn next_model_is_noop_on_empty_list() {
        let mut app = test_app();
        app.model_list_state.select(Some(0));

        app.next_model();
        assert_eq!(app.model_list_state.selected(), Some(0));
    }

    #[test]
    fn selected_model_and_find_model_locate_by_id() {
        let mut app = test_app();
        app.models_state = LoadState::Ready(vec![model("org/a"), model("org/b")]);
        app.model_list_state.select(Some(1));

        assert_eq!(app.selected_model().unwrap().model_id, "org/b");
        assert_eq!(app.find_model("org/a").unwrap().model_id, "org/a");
        assert!(app.find_model("org/missing").is_none());
    }

    #[test]
    fn best_quant_for_none_while_hardware_loading() {
        let app = test_app();
        assert_eq!(app.best_quant_for(&model("org/a")), None);
    }

    #[test]
    fn recompute_params_noop_while_hardware_loading() {
        let mut app = test_app();
        app.recompute_params();
        assert!(app.params.is_none());
    }

    #[test]
    fn recompute_params_sets_params_once_hardware_ready() {
        let mut app = test_app();
        app.hw_state = LoadState::Ready(ready_hw());
        app.models_state = LoadState::Ready(vec![model("org/a")]);
        app.model_list_state.select(Some(0));

        app.recompute_params();
        assert!(app.params.is_some());
    }

    #[test]
    fn start_search_seeds_input_from_active_query() {
        let mut app = test_app();
        app.search_query = Some("existing".to_string());

        app.start_search();
        assert!(app.search_editing);
        assert_eq!(app.search_input, "existing");
    }

    #[test]
    fn cancel_search_editing_clears_input_and_editing_flag() {
        let mut app = test_app();
        app.search_editing = true;
        app.search_input = "typed text".to_string();

        app.cancel_search_editing();
        assert!(!app.search_editing);
        assert!(app.search_input.is_empty());
    }

    #[test]
    fn start_editing_path_seeds_input_from_config() {
        let mut app = test_app();
        app.config.llama_cpp_path = Some("/usr/local/bin/llama-cli".to_string());

        app.start_editing_path();
        assert!(app.settings_editing);
        assert_eq!(app.settings_input, "/usr/local/bin/llama-cli");
    }

    #[test]
    fn cancel_editing_path_clears_input_and_editing_flag() {
        let mut app = test_app();
        app.settings_editing = true;
        app.settings_input = "typed text".to_string();

        app.cancel_editing_path();
        assert!(!app.settings_editing);
        assert!(app.settings_input.is_empty());
    }
}
