//! AI coding assistant skill management.
//!
//! Detects the user's AI tool, copies the right adapter and skill files.
//! Skills live in `ai/skills/` (universal) and `ai/adapters/<tool>/` (tool-specific).

use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Embedded adapters (for when we're not in a repo checkout)
// ---------------------------------------------------------------------------

const CLAUDE_ADAPTER: &str = include_str!("../assets/skills/claude-port.md");
const SKILL_GUIDE: &str = include_str!("../assets/skills/port-guide.md");
const SKILL_INSTRUCTIONS: &str = include_str!("../assets/skills/port-instructions.md");

// ---------------------------------------------------------------------------
// Skill registry
// ---------------------------------------------------------------------------

struct SkillInfo {
    name: &'static str,
    description: &'static str,
}

const SKILLS: &[SkillInfo] = &[SkillInfo {
    name: "port",
    description: "Port PyTorch scripts to flodl",
}];

// ---------------------------------------------------------------------------
// Tool detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Tool {
    Claude,
    Cursor,
}

impl Tool {
    fn name(&self) -> &'static str {
        match self {
            Tool::Claude => "Claude Code",
            Tool::Cursor => "Cursor",
        }
    }
}

/// Detect which AI tools are present in the current directory.
fn detect_tools() -> Vec<Tool> {
    let mut tools = Vec::new();
    if Path::new(".claude").is_dir() || Path::new(".claude").exists() {
        tools.push(Tool::Claude);
    }
    if Path::new(".cursor").is_dir() || Path::new(".cursorrules").exists() {
        tools.push(Tool::Cursor);
    }
    tools
}

fn parse_tool(name: &str) -> Option<Tool> {
    match name.to_lowercase().as_str() {
        "claude" | "claude-code" => Some(Tool::Claude),
        "cursor" => Some(Tool::Cursor),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Source locator
// ---------------------------------------------------------------------------

/// Find the ai/ directory in a repo checkout (walk up from cwd).
fn find_ai_dir() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    for _ in 0..5 {
        let candidate = dir.join("ai/skills");
        if candidate.is_dir() {
            return Some(dir.join("ai"));
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

/// Install skills for the detected (or specified) AI tool.
pub fn install(tool_override: Option<&str>, skill_filter: Option<&str>) -> Result<(), String> {
    let tools = if let Some(name) = tool_override {
        vec![
            parse_tool(name)
                .ok_or_else(|| format!("unknown tool: '{}'. Supported: claude, cursor", name))?,
        ]
    } else {
        let detected = detect_tools();
        if detected.is_empty() {
            // Default to Claude if nothing detected
            println!("No AI tool config detected. Defaulting to Claude Code.");
            println!("  (Override with: fdl skill install --tool cursor)");
            println!();
            vec![Tool::Claude]
        } else {
            detected
        }
    };

    let ai_dir = find_ai_dir();

    for tool in &tools {
        match tool {
            Tool::Claude => install_claude(&ai_dir, skill_filter)?,
            Tool::Cursor => install_cursor(&ai_dir, skill_filter)?,
        }
    }

    Ok(())
}

fn install_claude(ai_dir: &Option<PathBuf>, skill_filter: Option<&str>) -> Result<(), String> {
    let skills_to_install: Vec<&SkillInfo> = SKILLS
        .iter()
        .filter(|s| skill_filter.is_none() || skill_filter == Some(s.name))
        .collect();

    if skills_to_install.is_empty() {
        return Err(format!(
            "unknown skill: '{}'. Available: {}",
            skill_filter.unwrap_or(""),
            SKILLS.iter().map(|s| s.name).collect::<Vec<_>>().join(", ")
        ));
    }

    for skill in &skills_to_install {
        let skill_dir = PathBuf::from(format!(".claude/skills/{}", skill.name));
        let updating = skill_dir.join("SKILL.md").exists();
        fs::create_dir_all(&skill_dir)
            .map_err(|e| format!("cannot create {}: {}", skill_dir.display(), e))?;

        // Install SKILL.md (adapter)
        let adapter_content = if let Some(ai) = ai_dir {
            let adapter_path = ai.join("adapters/claude/port-skill.md");
            fs::read_to_string(&adapter_path).unwrap_or_else(|_| CLAUDE_ADAPTER.to_string())
        } else {
            CLAUDE_ADAPTER.to_string()
        };
        write_file(&skill_dir.join("SKILL.md"), &adapter_content)?;

        // Install universal skill files alongside the adapter
        let guide_content = if let Some(ai) = ai_dir {
            let path = ai.join(format!("skills/{}/guide.md", skill.name));
            fs::read_to_string(&path).unwrap_or_else(|_| SKILL_GUIDE.to_string())
        } else {
            SKILL_GUIDE.to_string()
        };
        write_file(&skill_dir.join("guide.md"), &guide_content)?;

        let instructions_content = if let Some(ai) = ai_dir {
            let path = ai.join(format!("skills/{}/instructions.md", skill.name));
            fs::read_to_string(&path).unwrap_or_else(|_| SKILL_INSTRUCTIONS.to_string())
        } else {
            SKILL_INSTRUCTIONS.to_string()
        };
        write_file(&skill_dir.join("instructions.md"), &instructions_content)?;

        let verb = if updating { "Updated" } else { "Installed" };
        println!("  {} /{} skill for Claude Code", verb, skill.name);
        println!("    -> .claude/skills/{}/SKILL.md", skill.name);
        println!("    -> .claude/skills/{}/guide.md", skill.name);
        println!("    -> .claude/skills/{}/instructions.md", skill.name);
    }

    println!();
    println!("Claude Code skills ready. Try: /port my_model.py");
    Ok(())
}

fn install_cursor(ai_dir: &Option<PathBuf>, skill_filter: Option<&str>) -> Result<(), String> {
    if skill_filter.is_some() && skill_filter != Some("port") {
        return Err(format!("unknown skill: '{}'", skill_filter.unwrap_or("")));
    }

    // For Cursor, append porting context to .cursorrules
    let rules_path = PathBuf::from(".cursorrules");
    let existing = fs::read_to_string(&rules_path).unwrap_or_default();

    if existing.contains("flodl porting") {
        println!("  Cursor rules already contain flodl porting context.");
        return Ok(());
    }

    let guide_content = if let Some(ai) = ai_dir {
        let path = ai.join("skills/port/guide.md");
        fs::read_to_string(&path).unwrap_or_else(|_| SKILL_GUIDE.to_string())
    } else {
        SKILL_GUIDE.to_string()
    };

    let cursor_block = format!(
        "\n\n# flodl porting\n\n\
         When asked to port PyTorch code to flodl, follow this guide:\n\n\
         {}\n",
        guide_content
    );

    let new_content = format!("{}{}", existing, cursor_block);
    write_file(&rules_path, &new_content)?;

    println!("  Installed flodl porting context for Cursor");
    println!("    -> .cursorrules (appended)");
    println!();
    println!("Cursor ready. Ask: \"Port this PyTorch code to flodl\"");
    Ok(())
}

fn write_file(path: &Path, content: &str) -> Result<(), String> {
    fs::write(path, content).map_err(|e| format!("cannot write {}: {}", path.display(), e))
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

pub fn list() {
    println!("Available skills:");
    println!();
    for skill in SKILLS {
        println!("  {:<12} {}", skill.name, skill.description);
    }
    println!();

    let tools = detect_tools();
    if tools.is_empty() {
        println!("No AI tool detected. Install with: fdl skill install");
    } else {
        println!("Detected tools:");
        for tool in &tools {
            let installed = check_installed(tool);
            let status = if installed {
                "installed"
            } else {
                "not installed"
            };
            println!("  {:<16} {}", tool.name(), status);
        }
        println!();
        if tools.iter().any(|t| !check_installed(t)) {
            println!("Run: fdl skill install");
        }
    }
}

fn check_installed(tool: &Tool) -> bool {
    match tool {
        Tool::Claude => Path::new(".claude/skills/port/SKILL.md").exists(),
        Tool::Cursor => fs::read_to_string(".cursorrules")
            .map(|c| c.contains("flodl porting"))
            .unwrap_or(false),
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

pub fn print_usage() {
    println!("fdl skill -- manage AI coding assistant skills");
    println!();
    println!("USAGE:");
    println!("    fdl skill <command> [options]");
    println!();
    println!("COMMANDS:");
    println!("    install            Install skills for detected AI tool");
    println!("        --tool <name>  Force a specific tool (claude, cursor)");
    println!("        --skill <name> Install only one skill");
    println!("    list               Show available skills and detected tools");
    println!();
    println!("SUPPORTED TOOLS:");
    println!("    claude             Claude Code (.claude/skills/)");
    println!("    cursor             Cursor (.cursorrules)");
    println!();
    println!("EXAMPLES:");
    println!("    fdl skill install              # auto-detect tool, install all skills");
    println!("    fdl skill install --tool claude # force Claude Code");
    println!("    fdl skill list                 # show what's available");
}
