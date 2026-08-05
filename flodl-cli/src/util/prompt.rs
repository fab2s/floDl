//! Interactive terminal prompts.
//!
//! Reads from the terminal directly (`/dev/tty` on Unix, `CONIN$` on Windows)
//! so prompts work even when stdin is piped.

use std::io::{self, BufRead, Write};

/// Open the terminal for reading, bypassing stdin.
fn open_tty() -> io::Result<Box<dyn BufRead>> {
    #[cfg(unix)]
    {
        use std::fs::File;
        let f = File::open("/dev/tty")?;
        Ok(Box::new(io::BufReader::new(f)))
    }
    #[cfg(windows)]
    {
        use std::fs::OpenOptions;
        let f = OpenOptions::new().read(true).open("CONIN$")?;
        Ok(Box::new(io::BufReader::new(f)))
    }
    #[cfg(not(any(unix, windows)))]
    {
        Ok(Box::new(io::BufReader::new(io::stdin())))
    }
}

fn read_line(tty: &mut dyn BufRead) -> String {
    let mut buf = String::new();
    let _ = tty.read_line(&mut buf);
    buf.trim().to_string()
}

/// Detect a usable controlling terminal, mirroring [`open_tty`]'s
/// platform paths — so callers can error loudly BEFORE a prompt would
/// silently fall back to its default.
pub fn has_tty() -> bool {
    #[cfg(unix)]
    {
        std::fs::File::open("/dev/tty").is_ok()
    }
    #[cfg(windows)]
    {
        std::fs::OpenOptions::new()
            .read(true)
            .open("CONIN$")
            .is_ok()
    }
    #[cfg(not(any(unix, windows)))]
    {
        true
    }
}

/// Ask a yes/no question. Returns `default` on empty input.
///
/// Prompt should NOT include the `[Y/n]` suffix -- it is appended automatically.
pub fn ask_yn(prompt: &str, default: bool) -> bool {
    let suffix = if default { "[Y/n]" } else { "[y/N]" };
    print!("{} {} ", prompt, suffix);
    let _ = io::stdout().flush();

    let mut tty = match open_tty() {
        Ok(t) => t,
        Err(_) => return default,
    };
    let answer = read_line(&mut *tty);

    match answer.as_str() {
        "" => default,
        s if s.starts_with('y') || s.starts_with('Y') => true,
        s if s.starts_with('n') || s.starts_with('N') => false,
        _ => default,
    }
}

/// Present a numbered menu and return the selected index (1-based).
///
/// Returns `default` (1-based) on empty or invalid input.
pub fn ask_choice(prompt: &str, options: &[&str], default: usize) -> usize {
    for (i, opt) in options.iter().enumerate() {
        println!("    {}) {}", i + 1, opt);
    }
    println!();
    print!("{} [{}]: ", prompt, default);
    let _ = io::stdout().flush();

    let mut tty = match open_tty() {
        Ok(t) => t,
        Err(_) => return default,
    };
    let answer = read_line(&mut *tty);

    if answer.is_empty() {
        return default;
    }
    match answer.parse::<usize>() {
        Ok(n) if n >= 1 && n <= options.len() => n,
        _ => default,
    }
}

/// Ask for free-text input with a default value.
///
/// Returns `default` on empty input.
#[allow(dead_code)]
pub fn ask_text(prompt: &str, default: &str) -> String {
    if default.is_empty() {
        print!("{}: ", prompt);
    } else {
        print!("{} [{}]: ", prompt, default);
    }
    let _ = io::stdout().flush();

    let mut tty = match open_tty() {
        Ok(t) => t,
        Err(_) => return default.to_string(),
    };
    let answer = read_line(&mut *tty);

    if answer.is_empty() {
        default.to_string()
    } else {
        answer
    }
}
