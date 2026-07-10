//! Pure `ratatui` rendering, driven entirely off `&App` — no state mutation
//! happens here.

use bytesize::ByteSize;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Gauge, List, ListItem, Paragraph, Row, Table, Tabs, Wrap},
    Frame,
};

use crate::{
    app::{App, AppTab, LoadState},
    hardware::GpuVendor,
    models::HfModel,
};

const ACCENT: Color = Color::Cyan;
const DIM: Color = Color::DarkGray;
const GOOD: Color = Color::Green;
const WARN: Color = Color::Yellow;
const ERR: Color = Color::Red;

pub(crate) fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title / tabs
            Constraint::Min(0),    // body
            Constraint::Length(1), // status bar
        ])
        .split(area);

    draw_tabs(frame, app, chunks[0]);
    draw_body(frame, app, chunks[1]);
    draw_statusbar(frame, app, chunks[2]);
}

fn draw_tabs(frame: &mut Frame, app: &App, area: Rect) {
    let tab_titles: Vec<Line> = vec![
        Line::from("  Hardware [1]  "),
        Line::from("  Parameters [2]  "),
        Line::from("  Models [3]  "),
        Line::from("  Settings [4]  "),
    ];
    let selected = match app.current_tab {
        AppTab::Hardware => 0,
        AppTab::Parameters => 1,
        AppTab::Models => 2,
        AppTab::Settings => 3,
    };
    let tabs = Tabs::new(tab_titles)
        .select(selected)
        .block(Block::default().borders(Borders::ALL).title(Span::styled(
            " 🦙 llama-tune ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
        )
        .style(Style::default().fg(Color::White));
    frame.render_widget(tabs, area);
}

fn draw_body(frame: &mut Frame, app: &App, area: Rect) {
    match app.current_tab {
        AppTab::Hardware => draw_hardware(frame, app, area),
        AppTab::Parameters => draw_parameters(frame, app, area),
        AppTab::Models => draw_models(frame, app, area),
        AppTab::Settings => draw_settings(frame, app, area),
    }
}

// ── Hardware tab ─────────────────────────────────────────────────────────────

fn draw_hardware(frame: &mut Frame, app: &App, area: Rect) {
    match &app.hw_state {
        LoadState::Loading => {
            let p = Paragraph::new("Scanning hardware…")
                .style(Style::default().fg(WARN))
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title(" Hardware "));
            frame.render_widget(p, area);
        }
        LoadState::Error(e) => {
            let p = Paragraph::new(format!("Error: {e}"))
                .style(Style::default().fg(ERR))
                .block(Block::default().borders(Borders::ALL).title(" Hardware "));
            frame.render_widget(p, area);
        }
        LoadState::Ready(hw) => {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([Constraint::Length(10), Constraint::Min(0)])
                .split(area);

            let outer = Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(" Hardware ", Style::default().fg(ACCENT)));
            frame.render_widget(outer, area);

            // CPU / RAM summary table
            let rows = vec![
                Row::new(vec![
                    Cell::from("CPU").style(Style::default().fg(DIM)),
                    Cell::from(hw.cpu_name.clone()),
                ]),
                Row::new(vec![
                    Cell::from("Cores (phys / logi)").style(Style::default().fg(DIM)),
                    Cell::from(format!(
                        "{} / {}",
                        hw.cpu_physical_cores, hw.cpu_logical_cores
                    )),
                ]),
                Row::new(vec![
                    Cell::from("Total RAM").style(Style::default().fg(DIM)),
                    Cell::from(hw.ram_display()).style(Style::default().fg(GOOD)),
                ]),
                Row::new(vec![
                    Cell::from("Available RAM").style(Style::default().fg(DIM)),
                    Cell::from(hw.available_ram_display()),
                ]),
            ];
            let table = Table::new(rows, [Constraint::Length(22), Constraint::Min(0)])
                .block(Block::default().borders(Borders::NONE));
            frame.render_widget(table, chunks[0]);

            // GPU list
            if hw.gpus.is_empty() {
                let p = Paragraph::new("No GPU detected — will recommend CPU-only parameters.")
                    .style(Style::default().fg(WARN))
                    .block(Block::default().borders(Borders::TOP).title(" GPUs "));
                frame.render_widget(p, chunks[1]);
            } else {
                let items: Vec<ListItem> = hw
                    .gpus
                    .iter()
                    .map(|g| {
                        let color = match g.vendor {
                            GpuVendor::Nvidia => Color::Green,
                            GpuVendor::Amd => Color::Red,
                            GpuVendor::Intel => Color::Blue,
                            GpuVendor::Apple => Color::Magenta,
                            GpuVendor::Other(_) => Color::White,
                        };
                        ListItem::new(Line::from(vec![
                            Span::styled(
                                format!("[{}] ", g.vendor),
                                Style::default().fg(color).add_modifier(Modifier::BOLD),
                            ),
                            Span::raw(g.name.clone()),
                            Span::styled(
                                if g.dedicated {
                                    format!("  ({})", ByteSize(g.vram_bytes))
                                } else {
                                    format!("  ({} shared with RAM)", ByteSize(g.vram_bytes))
                                },
                                Style::default().fg(ACCENT),
                            ),
                        ]))
                    })
                    .collect();

                let title = if hw.has_dedicated_gpu() {
                    format!(
                        " GPUs (total VRAM: {}) ",
                        ByteSize(hw.total_dedicated_vram_bytes())
                    )
                } else {
                    " GPUs (memory shared with RAM) ".to_string()
                };
                let list = List::new(items).block(
                    Block::default()
                        .borders(Borders::TOP)
                        .title(Span::styled(title, Style::default().fg(ACCENT))),
                );
                frame.render_widget(list, chunks[1]);
            }
        }
    }
}

// ── Parameters tab ───────────────────────────────────────────────────────────

fn draw_parameters(frame: &mut Frame, app: &App, area: Rect) {
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" Parameters ", Style::default().fg(ACCENT)));
    frame.render_widget(outer, area);

    let inner = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(area);

    match (&app.hw_state, &app.params) {
        (LoadState::Ready(_), Some(p)) => {
            // CLI string
            let title = match app.selected_model() {
                Some(m) => format!(" Recommended llama.cpp flags — for {} ", m.model_id),
                None => " Recommended llama.cpp flags (generic — no model selected) ".to_string(),
            };
            let cli_block = Block::default().borders(Borders::ALL).title(title);
            let cli_text = Paragraph::new(p.to_cli_string())
                .style(Style::default().fg(GOOD))
                .block(cli_block)
                .wrap(Wrap { trim: false });
            frame.render_widget(cli_text, inner[0]);

            // Rationale
            let items: Vec<ListItem> = p
                .rationale
                .iter()
                .map(|r| {
                    ListItem::new(Line::from(vec![
                        Span::styled("• ", Style::default().fg(ACCENT)),
                        Span::raw(r.clone()),
                    ]))
                })
                .collect();
            let list = List::new(items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Why these values? "),
            );
            frame.render_widget(list, inner[1]);
        }
        (LoadState::Loading, _) => {
            let p = Paragraph::new("Detecting hardware…")
                .style(Style::default().fg(WARN))
                .alignment(Alignment::Center);
            frame.render_widget(p, inner[0]);
        }
        (LoadState::Error(e), _) => {
            let p = Paragraph::new(format!("Hardware error: {e}")).style(Style::default().fg(ERR));
            frame.render_widget(p, inner[0]);
        }
        _ => {
            let p = Paragraph::new("Waiting for hardware data…").alignment(Alignment::Center);
            frame.render_widget(p, inner[0]);
        }
    }
}

// ── Models tab ───────────────────────────────────────────────────────────────

pub(crate) fn draw_models(frame: &mut Frame, app: &App, area: Rect) {
    let title = match &app.search_query {
        Some(q) => format!(" Model Search: \"{q}\" "),
        None => " Model Recommendations ".to_string(),
    };
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(title, Style::default().fg(ACCENT)));
    frame.render_widget(outer, area);

    // Reserve a row for the search box only while it's being edited; margin(1)
    // here accounts for the outer block's border, so the horizontal split
    // below (for the list/detail panes) doesn't re-apply its own margin.
    let outer_inner = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(if app.search_editing { 3 } else { 0 }),
            Constraint::Min(0),
        ])
        .split(area);

    if app.search_editing {
        let search_box = Paragraph::new(format!("/{}", app.search_input))
            .style(Style::default().fg(ACCENT))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Search (Enter to submit, Esc to cancel, empty = show recommended) "),
            );
        frame.render_widget(search_box, outer_inner[0]);
    }

    let inner = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(outer_inner[1]);

    // Installed models (fetched independently of the active search) come
    // first, followed by whatever's currently loaded for search/recommended —
    // so an install stays visible even while a search is loading or its
    // results don't include it.
    let combined = app.display_models();

    if combined.is_empty() {
        match &app.models_state {
            LoadState::Loading => {
                let msg = match &app.search_query {
                    Some(q) => format!("Searching for \"{q}\"…"),
                    None => "Fetching models from Hugging Face…".to_string(),
                };
                let p = Paragraph::new(msg)
                    .style(Style::default().fg(WARN))
                    .alignment(Alignment::Center);
                frame.render_widget(p, inner[0]);
            }
            LoadState::Error(e) => {
                let p = Paragraph::new(format!("Error: {e}"))
                    .style(Style::default().fg(ERR))
                    .wrap(Wrap { trim: true });
                frame.render_widget(p, inner[0]);
            }
            LoadState::Ready(_) => {
                let msg = match &app.search_query {
                    Some(q) => format!("No models matching \"{q}\" fit your hardware."),
                    None => "No models found that fit your hardware.".to_string(),
                };
                let p = Paragraph::new(msg)
                    .style(Style::default().fg(WARN))
                    .alignment(Alignment::Center);
                frame.render_widget(p, inner[0]);
            }
        }
        return;
    }

    draw_model_list(frame, app, inner[0], &combined);
    draw_model_detail(frame, app, inner[1], &combined);
}

fn draw_model_list(frame: &mut Frame, app: &App, area: Rect, combined: &[&HfModel]) {
    let items: Vec<ListItem> = combined
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let selected = app.model_list_state.selected() == Some(i);
            let style = if selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            // Size of the quant that would actually be downloaded for
            // this model, not the repo's combined size across every
            // quant variant it offers.
            let sz_str = match app.best_quant_for(m).and_then(|q| m.quant_size_bytes(q)) {
                Some(sz) => format!(" [{}]", ByteSize(sz)),
                None => String::new(),
            };
            let mut spans = vec![Span::styled(m.model_id.clone(), style)];
            if app.installed.contains_key(&m.model_id) {
                spans.push(Span::styled(
                    " ✓ installed",
                    Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
                ));
            }
            spans.push(Span::styled(sz_str, Style::default().fg(DIM)));
            ListItem::new(Line::from(spans))
        })
        .collect();

    let mut list_state = app.model_list_state.clone();
    let list = List::new(items)
        .block(Block::default().borders(Borders::RIGHT).title(" Models "))
        .highlight_symbol("▶ ");
    frame.render_stateful_widget(list, area, &mut list_state);
}

fn draw_model_detail(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    combined: &[&HfModel],
) {
    if let Some(idx) = app.model_list_state.selected() {
        if let Some(m) = combined.get(idx).copied() {
            let best_quant = app.best_quant_for(m);
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Model: ", Style::default().fg(DIM)),
                    Span::styled(
                        m.model_id.clone(),
                        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Status: ", Style::default().fg(DIM)),
                    if app.installed.contains_key(&m.model_id) {
                        Span::styled(
                            "Installed",
                            Style::default().fg(GOOD).add_modifier(Modifier::BOLD),
                        )
                    } else {
                        Span::styled("Not installed", Style::default().fg(DIM))
                    },
                ]),
                Line::from(vec![
                    Span::styled("Downloads: ", Style::default().fg(DIM)),
                    Span::raw(
                        m.downloads
                            .map(|d| d.to_string())
                            .unwrap_or_else(|| "N/A".into()),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Likes: ", Style::default().fg(DIM)),
                    Span::raw(
                        m.likes
                            .map(|l| l.to_string())
                            .unwrap_or_else(|| "N/A".into()),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Last updated: ", Style::default().fg(DIM)),
                    Span::raw(m.last_modified.clone().unwrap_or_else(|| "N/A".into())),
                ]),
                Line::from(vec![
                    Span::styled("Est. layers: ", Style::default().fg(DIM)),
                    Span::raw(m.estimated_layers().to_string()),
                ]),
                Line::from(vec![
                    Span::styled("Best quant for your machine: ", Style::default().fg(DIM)),
                    Span::styled(
                        best_quant.unwrap_or("N/A").to_string(),
                        Style::default().fg(GOOD),
                    ),
                ]),
            ];

            if let Some(quant) = best_quant {
                if let Some(url) = m.download_url_for(quant) {
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled(
                        "Download URL:",
                        Style::default().fg(DIM),
                    )));
                    lines.push(Line::from(Span::styled(
                        url,
                        Style::default()
                            .fg(Color::Blue)
                            .add_modifier(Modifier::UNDERLINED),
                    )));
                }
            }

            // Tags
            if let Some(tags) = &m.tags {
                let tag_str = tags
                    .iter()
                    .filter(|t| !t.starts_with("base_model"))
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled("Tags: ", Style::default().fg(DIM)),
                    Span::raw(tag_str),
                ]));
            }

            let active_download = app.download.as_ref().filter(|dl| dl.model_id == m.model_id);

            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                if active_download.is_some() {
                    "Downloading… press [x] to cancel"
                } else if app.installed.contains_key(&m.model_id) {
                    "Press [l] to launch llama.cpp with this model"
                } else {
                    "Press [l] to download this model to the HF cache and launch it"
                },
                Style::default().fg(DIM),
            )));

            if let Some(status) = &app.launch_status {
                let color = if status.starts_with("Launched") {
                    GOOD
                } else {
                    WARN
                };
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    status.clone(),
                    Style::default().fg(color),
                )));
            }

            let detail_area = if active_download.is_some() {
                let split = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(0), Constraint::Length(2)])
                    .split(area);
                split[0]
            } else {
                area
            };

            let detail = Paragraph::new(lines)
                .block(Block::default().borders(Borders::NONE).title(" Details "))
                .wrap(Wrap { trim: true });
            frame.render_widget(detail, detail_area);

            if let Some(dl) = active_download {
                let gauge_area = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(0), Constraint::Length(2)])
                    .split(area)[1];

                let (ratio, label) = match dl.total {
                    Some(total) if total > 0 => (
                        (dl.downloaded as f64 / total as f64).clamp(0.0, 1.0),
                        format!(
                            "{} / {} ({:.0}%)",
                            ByteSize(dl.downloaded),
                            ByteSize(total),
                            dl.downloaded as f64 / total as f64 * 100.0
                        ),
                    ),
                    _ => (0.0, format!("{} downloaded", ByteSize(dl.downloaded))),
                };

                let gauge = Gauge::default()
                    .block(Block::default().borders(Borders::TOP).title(" Download "))
                    .gauge_style(Style::default().fg(WARN))
                    .ratio(ratio)
                    .label(label);
                frame.render_widget(gauge, gauge_area);
            }
        }
    }
}

// ── Settings tab ─────────────────────────────────────────────────────────────

fn draw_settings(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" Settings ", Style::default().fg(ACCENT)));
    let inner_area = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = vec![
        Line::from(vec![
            Span::styled("llama.cpp path: ", Style::default().fg(DIM)),
            if app.settings_editing {
                Span::styled(
                    format!("{}_", app.settings_input),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                )
            } else {
                match &app.config.llama_cpp_path {
                    Some(p) => Span::styled(p.clone(), Style::default().fg(GOOD)),
                    None => Span::styled("(not set)", Style::default().fg(WARN)),
                }
            },
        ]),
        Line::from(""),
    ];

    if app.settings_editing {
        lines.push(Line::from(Span::styled(
            "Type the path to your llama.cpp executable, or a directory containing it (e.g. llama.exe/llama-cli.exe/main). [Enter] save, [Esc] cancel.",
            Style::default().fg(DIM),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "[e] or [Enter] to edit the path used when launching llama.cpp from the Models tab.",
            Style::default().fg(DIM),
        )));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("Last launched model: ", Style::default().fg(DIM)),
        match &app.config.last_launched_model_id {
            Some(id) => Span::styled(id.clone(), Style::default().fg(GOOD)),
            None => Span::styled("(none yet)", Style::default().fg(DIM)),
        },
    ]));
    lines.push(Line::from(Span::styled(
        "[L] Relaunch it from any tab (downloading again first if needed).",
        Style::default().fg(DIM),
    )));

    let p = Paragraph::new(lines).wrap(Wrap { trim: true });
    frame.render_widget(p, inner_area);
}

// ── Status bar ───────────────────────────────────────────────────────────────

fn draw_statusbar(frame: &mut Frame, app: &App, area: Rect) {
    // Shown regardless of tab, unconditionally — not just on non-Models tabs.
    // The Models tab's detail panel *also* shows `launch_status`, but that
    // panel doesn't scroll: with enough model info/tags/download-URL content
    // above it, the (last-appended) status line can be clipped off the bottom
    // of a normal-sized terminal and never render at all. The status bar is a
    // single fixed line, so it's the only spot that's reliably visible no
    // matter how much other content the current tab is showing.
    // An in-flight download takes priority over both the key hints and
    // `launch_status`: it's the one thing whose state keeps changing on its
    // own, so it needs to stay visible on every tab (not just Models) and
    // regardless of what's selected/searched — the Models tab's detail-panel
    // Gauge only shows when the downloading model happens to be selected.
    if let Some(dl) = &app.download {
        let text = match dl.total {
            Some(total) if total > 0 => format!(
                "Downloading {}: {} / {} ({:.0}%) — [x] Cancel",
                dl.model_id,
                ByteSize(dl.downloaded),
                ByteSize(total),
                dl.downloaded as f64 / total as f64 * 100.0
            ),
            _ => format!(
                "Downloading {}: {} downloaded — [x] Cancel",
                dl.model_id,
                ByteSize(dl.downloaded)
            ),
        };
        let bar = Paragraph::new(format!(" {text} ")).style(Style::default().fg(WARN));
        frame.render_widget(bar, area);
        return;
    }

    if let Some(status) = &app.launch_status {
        let color = if status.starts_with("Launched") {
            GOOD
        } else {
            WARN
        };
        let bar = Paragraph::new(format!(" {status} ")).style(Style::default().fg(color));
        frame.render_widget(bar, area);
        return;
    }

    let models_keys = if app.download.is_some() {
        " [1/2/3/4] Tabs  [↑↓] Select model  [x] Cancel download  [/] Search  [L] Relaunch last  [r] Refresh  [q] Quit "
    } else {
        " [1/2/3/4] Tabs  [↑↓] Select model  [l] Launch/download  [/] Search  [L] Relaunch last  [r] Refresh  [q] Quit "
    };
    let keys = match app.current_tab {
        AppTab::Models if app.search_editing => " [Enter] Search  [Esc] Cancel ",
        AppTab::Models => models_keys,
        AppTab::Settings if app.settings_editing => " [Enter] Save  [Esc] Cancel ",
        AppTab::Settings => " [1/2/3/4] Tabs  [e] Edit path  [L] Relaunch last  [q] Quit ",
        _ => " [1/2/3/4] Tabs  [L] Relaunch last  [r] Refresh  [q] Quit ",
    };
    let status = Paragraph::new(keys).style(Style::default().fg(DIM));
    frame.render_widget(status, area);
}
