//! Resolves the configured llama.cpp executable and launches it as a
//! detached background process with the computed CLI parameters.

use std::path::{Path, PathBuf};
use std::process::{Child, Command};

use anyhow::{bail, Context, Result};

use crate::params::LlamaCppParams;

/// llama.cpp's interactive CLI binary has been renamed across releases
/// (`main` historically, then `llama-cli`, and some current builds ship
/// just `llama`) — checked in this order so a directory can be resolved
/// regardless of which naming scheme it shipped with.
const CANDIDATE_NAMES: &[&str] = &["llama-cli", "llama", "main"];

/// If `path` is already a file, use it as-is. If it's a directory (e.g. a
/// version-tracking symlink like `current` pointing at a versioned build
/// folder), search it for a known llama.cpp CLI binary name instead of
/// requiring the user to know the exact filename, which changes across
/// llama.cpp releases. `Command::new` on a directory fails at the OS level
/// with no window or process ever appearing, which is why this used to look
/// like "llama.cpp doesn't start" with no visible explanation.
pub(crate) fn resolve_executable(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        for name in CANDIDATE_NAMES {
            let candidate = path.join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        bail!(
            "`{}` is a directory, and none of the expected llama.cpp binaries ({}) were found inside it. Point the Settings-tab path at the actual executable.",
            path.display(),
            CANDIDATE_NAMES.join(", ")
        );
    }
    bail!("`{}` does not exist", path.display());
}

/// Some recent llama.cpp releases consolidated `llama-cli`/`llama-server`/etc.
/// into a single `llama` binary that dispatches via a subcommand (`llama cli
/// -m ...`, `llama serve ...`) instead of taking flags directly — running it
/// with no subcommand exits immediately with "unknown command '-m'", which
/// looks exactly like "llama.cpp doesn't start" since the flag-only args this
/// tool built are the first thing on the command line. Only that unified
/// binary is named exactly `llama`; `llama-cli`/`main` are the older
/// standalone binaries and still take flags directly with no subcommand.
fn is_unified_binary(exe: &Path) -> bool {
    exe.file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("llama"))
}

/// Outcome of successfully spawning llama.cpp: a short status message for
/// display, plus the `Child` so the caller can watch for an early exit.
pub(crate) struct LaunchedProcess {
    pub(crate) message: String,
    pub(crate) child: Child,
}

/// Launch llama.cpp as a detached background process for `model_path`, passing
/// the computed parameters as CLI args. Returns a short status message plus
/// the spawned `Child` on success, so the caller can optionally watch for an
/// early exit (e.g. a bad model file or an arg some specific build rejects)
/// without this function itself blocking on the child.
pub(crate) fn spawn_llama_cpp(
    exe_path: &str,
    model_path: &Path,
    params: &LlamaCppParams,
) -> Result<LaunchedProcess> {
    let exe = resolve_executable(Path::new(exe_path))?;
    let mut cmd = Command::new(&exe);

    let unified = is_unified_binary(&exe);
    if unified {
        cmd.arg("cli");
    }
    cmd.arg("-m").arg(model_path);
    for arg in params.to_cli_args() {
        // The unified binary's `--flash-attn` takes an explicit on/off/auto
        // value where the older standalone binaries treat it as a bare
        // boolean flag (present = enabled).
        if unified && arg == "--flash-attn" {
            cmd.arg(&arg).arg("on");
        } else {
            cmd.arg(&arg);
        }
    }

    // Detach fully so the child survives independently of llama-tune and
    // doesn't share/corrupt our TUI's stdio.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;
        cmd.creation_flags(CREATE_NEW_CONSOLE);
    }
    #[cfg(not(windows))]
    {
        use std::process::Stdio;
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::null());
    }

    let child = cmd
        .spawn()
        .with_context(|| format!("failed to launch `{}`", exe.display()))?;

    let message = format!("Launched llama.cpp (pid {})", child.id());
    Ok(LaunchedProcess { message, child })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn touch(path: &Path) {
        std::fs::write(path, b"").unwrap();
    }

    #[test]
    fn resolve_executable_accepts_direct_file() {
        let dir = tempdir().unwrap();
        let exe = dir
            .path()
            .join(format!("llama-cli{}", std::env::consts::EXE_SUFFIX));
        touch(&exe);

        let resolved = resolve_executable(&exe).unwrap();
        assert_eq!(resolved, exe);
    }

    #[test]
    fn resolve_executable_finds_known_binary_in_directory() {
        let dir = tempdir().unwrap();
        let exe = dir
            .path()
            .join(format!("llama{}", std::env::consts::EXE_SUFFIX));
        touch(&exe);

        let resolved = resolve_executable(dir.path()).unwrap();
        assert_eq!(resolved, exe);
    }

    #[test]
    fn resolve_executable_prefers_first_candidate_name() {
        let dir = tempdir().unwrap();
        touch(
            &dir.path()
                .join(format!("llama-cli{}", std::env::consts::EXE_SUFFIX)),
        );
        touch(
            &dir.path()
                .join(format!("main{}", std::env::consts::EXE_SUFFIX)),
        );

        let resolved = resolve_executable(dir.path()).unwrap();
        assert_eq!(
            resolved.file_name().unwrap().to_str().unwrap(),
            format!("llama-cli{}", std::env::consts::EXE_SUFFIX)
        );
    }

    #[test]
    fn resolve_executable_errs_when_directory_has_no_known_binary() {
        let dir = tempdir().unwrap();
        touch(&dir.path().join("readme.txt"));

        let err = resolve_executable(dir.path()).unwrap_err();
        assert!(err
            .to_string()
            .contains("none of the expected llama.cpp binaries"));
    }

    #[test]
    fn resolve_executable_errs_when_path_missing() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");

        let err = resolve_executable(&missing).unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn is_unified_binary_matches_exact_stem_case_insensitively() {
        assert!(is_unified_binary(Path::new("/usr/local/bin/llama")));
        assert!(is_unified_binary(Path::new("C:\\tools\\LLAMA.exe")));
        assert!(!is_unified_binary(Path::new("/usr/local/bin/llama-cli")));
        assert!(!is_unified_binary(Path::new("/usr/local/bin/main")));
    }
}
