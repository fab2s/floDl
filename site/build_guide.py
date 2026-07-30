#!/usr/bin/env python3
"""Generate site/guide/*.md from site/_stubs/ frontmatter + docs/ content.

Each stub in site/_stubs/ contains Jekyll frontmatter with a `source:` key
pointing to the docs/ file that holds the actual content. This script:

1. Reads the stub's frontmatter
2. Reads the source markdown
3. Rewrites internal links (NN-file.md -> /guide/slug)
4. Strips trailing navigation sections
5. Writes the combined result to site/guide/

Run before Jekyll: python3 site/build_guide.py
"""

import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUBS_DIR = os.path.join(REPO_ROOT, "site", "_stubs")
GUIDE_DIR = os.path.join(REPO_ROOT, "site", "guide")

# Embedded skill assets: flodl-cli/assets/skills/ is the copy include_str!'d
# into the fdl binary (the out-of-repo fallback for `fdl skill install`).
# crates.io only packages the crate dir, so it cannot include_str! ../../ai/;
# this keeps the in-crate copy fresh from the ai/ sources as a side effect of
# every site build. Also available as `make sync-skills` and enforced at
# release by ci/release/09-skill-assets.sh. Keep the three lists in sync.
SKILL_ASSETS = [
    ("ai/skills/port/guide.md", "flodl-cli/assets/skills/port-guide.md"),
    ("ai/skills/port/instructions.md", "flodl-cli/assets/skills/port-instructions.md"),
    ("ai/adapters/claude/port-skill.md", "flodl-cli/assets/skills/claude-port.md"),
]

# Link rewrites: (pattern, replacement)
# Order matters — anchored variants before bare filenames.
LINK_REWRITES = [
    (r"\(00-rust-primer\.md\)", "(/guide/rust-primer)"),
    (r"\(01-tensors\.md\)", "(/guide/tensors)"),
    (r"\(02-autograd\.md\)", "(/guide/autograd)"),
    (r"\(03-modules\.md\)", "(/guide/modules)"),
    (r"\(04-training\.md(#[^)]+)\)", r"(/guide/training\1)"),
    (r"\(04-training\.md\)", "(/guide/training)"),
    (r"\(05-graph-builder\.md\)", "(/guide/graph-builder)"),
    (r"\(06-advanced-graphs\.md\)", "(/guide/advanced-graphs)"),
    (r"\(07-visualization\.md\)", "(/guide/visualization)"),
    (r"\(08-utilities\.md\)", "(/guide/utilities)"),
    (r"\(09-monitor\.md\)", "(/guide/monitor)"),
    (r"\(10-graph-tree\.md(#[^)]+)\)", r"(/guide/graph-tree\1)"),
    (r"\(10-graph-tree\.md\)", "(/guide/graph-tree)"),
    (r"\(11-multi-gpu\.md\)", "(/guide/multi-gpu)"),
    (r"\(12-async-ddp\.md\)", "(/guide/async-ddp)"),
    (r"\(13-data-loading\.md\)", "(/guide/data-loading)"),
    (r"\(14-flodl-hf\.md\)", "(/guide/flodl-hf)"),
    (r"\(\.\./pytorch_migration\.md\)", "(/guide/migration)"),
    (r"\(\.\./ddp\.md\)", "(/guide/ddp-reference)"),
    (r"\(\.\./troubleshooting\.md\)", "(/guide/troubleshooting)"),
    # Relative links to examples (from tutorials)
    (r"\(\.\./\.\./flodl/examples/([^)]+)\)", r"(https://github.com/flodl-labs/flodl/tree/main/flodl/examples/\1)"),
    # From docs/ root level (troubleshooting.md, pytorch_migration.md)
    (r"\(ddp\.md\)", "(/guide/ddp-reference)"),
    (r"\(ddp\.md(#[^)]+)\)", r"(/guide/ddp-reference\1)"),
    # ddp-benchmark.md is not a guide page; the site carries a hand-built
    # summary at /ddp-benchmark (full report stays on GitHub).
    (r"\(ddp-benchmark\.md\)", "(/ddp-benchmark)"),
    (r"\(pytorch_migration\.md\)", "(/guide/migration)"),
    (r"\(troubleshooting\.md\)", "(/guide/troubleshooting)"),
    (r"\(tutorials/00-rust-primer\.md\)", "(/guide/rust-primer)"),
    (r"\(tutorials/01-tensors\.md\)", "(/guide/tensors)"),
    (r"\(tutorials/10-graph-tree\.md\)", "(/guide/graph-tree)"),
    (r"\(tutorials/11-multi-gpu\.md\)", "(/guide/multi-gpu)"),
    (r"\(tutorials/12-async-ddp\.md\)", "(/guide/async-ddp)"),
    (r"\(tutorials/13-data-loading\.md\)", "(/guide/data-loading)"),
    (r"\(tutorials/14-flodl-hf\.md\)", "(/guide/flodl-hf)"),
    (r"\(cli\.md(#[^)]+)\)", r"(/guide/cli\1)"),
    (r"\(cli\.md\)", "(/guide/cli)"),
    (r"\(design/trainer-execution-tiers\.md\)", "(https://github.com/flodl-labs/flodl/blob/main/docs/design/trainer-execution-tiers.md)"),
]

NAV_LINE_RE = re.compile(
    r"^(Next:|Previous[ a-z]*:|\[.*?\]\(.*?\)[ |]*$)"
)


def parse_stub(path):
    """Return (frontmatter_text, source_path) from a stub file."""
    with open(path) as f:
        text = f.read()

    # Split on --- delimiters
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, None

    yaml_block = parts[1]
    frontmatter = f"---{yaml_block}---\n"

    # Extract source: field
    for line in yaml_block.strip().splitlines():
        line = line.strip()
        if line.startswith("source:"):
            source = line.split(":", 1)[1].strip().strip('"').strip("'")
            return frontmatter, source

    return frontmatter, None


def rewrite_links(text):
    """Rewrite docs/ relative links to /guide/ absolute links."""
    for pattern, replacement in LINK_REWRITES:
        text = re.sub(pattern, replacement, text)
    return text


def strip_trailing_nav(text):
    """Remove trailing navigation from docs/ files.

    Works backwards from end of file, stripping lines that are navigation
    (Next:/Previous:/link lines or blank). If a --- divider is reached
    and only nav follows it, the --- is stripped too.
    """
    lines = text.rstrip().split("\n")

    # Walk backwards, skip nav and blank lines
    cut = len(lines)
    while cut > 0:
        stripped = lines[cut - 1].strip()
        if stripped == "":
            cut -= 1
        elif NAV_LINE_RE.match(stripped):
            cut -= 1
        elif stripped == "---":
            cut -= 1  # also strip the --- if only nav follows
            break
        else:
            break

    if cut == len(lines):
        return text  # nothing to strip

    return "\n".join(lines[:cut]).rstrip() + "\n"


def strip_source_from_frontmatter(frontmatter):
    """Remove the source: line from frontmatter (not needed in output)."""
    lines = frontmatter.splitlines(keepends=True)
    return "".join(l for l in lines if not l.strip().startswith("source:"))


def git_last_modified(path):
    """ISO-8601 timestamp of the last commit touching `path`, or None if not in git."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI", "--", path],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def inject_last_modified(frontmatter, iso_ts):
    """Insert (or replace) `last_modified_at:` in a `---\\n…\\n---\\n` block."""
    lines = frontmatter.splitlines(keepends=True)
    lines = [l for l in lines if not l.strip().startswith("last_modified_at:")]
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "---":
            lines.insert(i, f"last_modified_at: {iso_ts}\n")
            break
    return "".join(lines)


def sync_skill_assets():
    """Copy the ai/ skill sources into the crate's embedded-asset dir."""
    for src_rel, dst_rel in SKILL_ASSETS:
        src = os.path.join(REPO_ROOT, src_rel)
        dst = os.path.join(REPO_ROOT, dst_rel)
        if not os.path.isfile(src):
            print(f"error: skill source not found: {src}", file=sys.stderr)
            sys.exit(1)
        shutil.copyfile(src, dst)
    print(f"synced {len(SKILL_ASSETS)} skill assets into flodl-cli/assets/skills/ from ai/")


def main():
    if not os.path.isdir(STUBS_DIR):
        print(f"error: {STUBS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    sync_skill_assets()

    os.makedirs(GUIDE_DIR, exist_ok=True)
    count = 0

    for filename in sorted(os.listdir(STUBS_DIR)):
        if not filename.endswith(".md"):
            continue

        stub_path = os.path.join(STUBS_DIR, filename)
        frontmatter, source_rel = parse_stub(stub_path)

        if not frontmatter or not source_rel:
            print(f"skip: {filename} (no source: in frontmatter)", file=sys.stderr)
            continue

        source_path = os.path.join(REPO_ROOT, source_rel)
        if not os.path.isfile(source_path):
            print(f"error: source not found: {source_path}", file=sys.stderr)
            sys.exit(1)

        with open(source_path) as f:
            content = f.read()

        content = rewrite_links(content)
        content = strip_trailing_nav(content)
        clean_frontmatter = strip_source_from_frontmatter(frontmatter)

        lastmod = git_last_modified(source_path)
        if lastmod:
            clean_frontmatter = inject_last_modified(clean_frontmatter, lastmod)

        output_path = os.path.join(GUIDE_DIR, filename)
        with open(output_path, "w") as f:
            f.write(clean_frontmatter)
            f.write("\n")
            f.write(content)

        count += 1

    print(f"generated {count} guide pages in site/guide/")


if __name__ == "__main__":
    main()
