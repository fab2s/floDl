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


def stub_field(frontmatter, key):
    """Read a scalar field out of a stub's frontmatter block."""
    for line in frontmatter.splitlines():
        line = line.strip()
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return None


# ── Navigation, derived from site/_data/guide_nav.yml ────────────────────
#
# The manifest is the single source of truth for grouping and order. prev/next
# and the docs/ index are computed from it, so they cannot drift out of sync
# with the sidebar the way four hand-kept lists did.

NAV_PATH = os.path.join(REPO_ROOT, "site", "_data", "guide_nav.yml")


def load_nav():
    """Parse guide_nav.yml without a YAML dependency.

    The schema is deliberately flat (groups -> members -> two scalar keys), so
    a tiny reader beats adding PyYAML to a build that otherwise needs nothing.
    Anything unexpected is a loud error, never a silent skip.
    """
    if not os.path.isfile(NAV_PATH):
        print(f"error: nav manifest not found: {NAV_PATH}", file=sys.stderr)
        sys.exit(1)

    groups, group = [], None
    with open(NAV_PATH) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].rstrip() if not raw.strip().startswith("#") else ""
            if not line.strip():
                continue
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())

            if stripped == "groups:":
                continue
            if stripped.startswith("- label:"):
                group = {
                    "label": stripped.split(":", 1)[1].strip().strip('"'),
                    "fold": False,
                    "open": True,
                    "numbered": False,
                    "members": [],
                }
                groups.append(group)
                continue
            if group is None:
                print(f"error: {NAV_PATH}:{lineno}: content before first group", file=sys.stderr)
                sys.exit(1)
            if stripped.startswith("- stub:"):
                group["members"].append(
                    {
                        "stub": stripped.split(":", 1)[1].strip().strip('"'),
                        "title": None,
                        "blurb": None,
                    }
                )
                continue
            if ":" not in stripped:
                print(f"error: {NAV_PATH}:{lineno}: cannot parse {stripped!r}", file=sys.stderr)
                sys.exit(1)

            key, val = (p.strip() for p in stripped.split(":", 1))
            val = val.strip('"')
            if key in ("title", "blurb") and group["members"] and indent >= 8:
                group["members"][-1][key] = val
            elif key in ("fold", "open", "numbered"):
                group[key] = val == "true"
            elif key == "members":
                continue
            else:
                print(f"error: {NAV_PATH}:{lineno}: unknown key {key!r}", file=sys.stderr)
                sys.exit(1)

    for g in groups:
        if not g["members"]:
            print(f"error: nav group {g['label']!r} has no members", file=sys.stderr)
            sys.exit(1)
        for m in g["members"]:
            for field in ("title", "blurb"):
                if not m[field]:
                    print(
                        f"error: nav member {m['stub']!r} in {g['label']!r} has no {field}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
    return groups


def nav_chain(groups):
    """Flatten the manifest into the linear prev/next reading order."""
    return [(m["stub"], m["title"], g["label"]) for g in groups for m in g["members"]]


def rewrite_links(text):
    """Rewrite docs/ relative links to /guide/ absolute links."""
    for pattern, replacement in LINK_REWRITES:
        text = re.sub(pattern, replacement, text)
    return text


def strip_trailing_nav(text):
    """Remove trailing navigation from docs/ files.

    When the generated marker is present, cut there — exact beats heuristic,
    and the marker sits *above* the divider so the backwards walk would
    otherwise stop at the divider and leave the comment in the page.

    Otherwise: works backwards from end of file, stripping lines that are
    navigation (Next:/Previous:/link lines or blank). If a --- divider is
    reached and only nav follows it, the --- is stripped too.
    """
    if NAV_TAIL_MARKER in text:
        text = text.split(NAV_TAIL_MARKER)[0]

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
    """Drop the fields the build owns: `source:` (not needed in output) and any
    hand-written prev/next (now derived from the nav manifest)."""
    drop = ("source:", "prev_url:", "prev_title:", "next_url:", "next_title:")
    lines = frontmatter.splitlines(keepends=True)
    return "".join(l for l in lines if not l.strip().startswith(drop))


def inject_nav(frontmatter, prev, nxt):
    """Write derived prev/next into a `---\\n…\\n---\\n` block.

    `prev`/`nxt` are (permalink, title) pairs or None at the chain's ends.
    """
    add = []
    if prev:
        add += [f'prev_url: {prev[0]}\n', f'prev_title: "{prev[1]}"\n']
    if nxt:
        add += [f'next_url: {nxt[0]}\n', f'next_title: "{nxt[1]}"\n']
    if not add:
        return frontmatter
    lines = frontmatter.splitlines(keepends=True)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "---":
            lines[i:i] = add
            break
    return "".join(lines)


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


SIDEBAR_PATH = os.path.join(REPO_ROOT, "site", "_includes", "sidebar.html")
DOCS_INDEX_PATH = os.path.join(REPO_ROOT, "docs", "README.md")

GENERATED_BANNER = "GENERATED by site/build_guide.py from site/_data/guide_nav.yml"


def esc(text):
    """Escape text destined for HTML (titles carry `&`, e.g. "Heterogeneous & …")."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_sidebar(groups, permalinks):
    """Emit _includes/sidebar.html from the manifest.

    Folding groups use <details>/<summary>: native, keyboard-navigable, and no
    JavaScript state to keep. The group holding the active page is forced open
    via Liquid so a fold never hides where the reader currently is.
    """
    out = [
        f"<!-- {GENERATED_BANNER}. Do not edit by hand. -->",
        '<aside class="sidebar" id="sidebar">',
    ]
    for g in groups:
        urls = [permalinks[m["stub"]] for m in g["members"]]
        label_html = esc(g["label"])
        links = []
        for i, m in enumerate(g["members"], 1):
            url = permalinks[m["stub"]]
            text = f"{i}. {m['title']}" if g["numbered"] else m["title"]
            links.append(
                f"    <a href=\"{{{{ '{url}' | relative_url }}}}\""
                f"{{% if page.url == '{url}' %}} class=\"active\"{{% endif %}}>{esc(text)}</a>"
            )

        if g["fold"]:
            if g["open"]:
                # Open by default: no need to test for the active page.
                attr = " open"
            else:
                # Collapsed by default, but never collapsed over the active page.
                cond = " or ".join(f"page.url == '{u}'" for u in urls)
                attr = f"{{% if {cond} %}} open{{% endif %}}"
            out += [
                f'  <details class="sidebar-group"{attr}>',
                f'    <summary class="sidebar-label">{label_html}</summary>',
                *links,
                "  </details>",
            ]
        else:
            out += [
                '  <div class="sidebar-group">',
                f'    <div class="sidebar-label">{label_html}</div>',
                *links,
                "  </div>",
            ]

    out += [
        '  <div class="sidebar-group">',
        '    <div class="sidebar-label">Reference</div>',
        '    <a href="https://docs.rs/flodl">API Docs &rarr;</a>',
        "  </div>",
        "</aside>",
        "",
    ]
    with open(SIDEBAR_PATH, "w") as f:
        f.write("\n".join(out))
    print(f"generated sidebar with {len(groups)} groups")


GUIDE_INDEX_PATH = os.path.join(GUIDE_DIR, "index.html")

GUIDE_INDEX_INTRO = (
    "From zero to training in 15 minutes. Coming from PyTorch, start with the "
    "porting guide; it routes you to the Rust primer if you want the language "
    "first, and the migration reference is there to look things up in."
)


def render_guide_index(groups, permalinks):
    """Emit site/guide/index.html — the guide landing page, cards and all.

    The blurbs live in the manifest so this page cannot drift from the sidebar
    the way it had (it still carried "Tutorial N" numbering after the headings
    dropped it, and never gained the Mac page).
    """
    out = [
        "---",
        "layout: guide",
        "title: Guide",
        "permalink: /guide/",
        "---",
        "",
        f"<!-- {GENERATED_BANNER}. Do not edit by hand. -->",
        "",
        "<h1>Guide</h1>",
        '<p style="color: var(--dim); margin-bottom: 32px;">',
        f"  {GUIDE_INDEX_INTRO}",
        "</p>",
        "",
    ]
    for g in groups:
        out += [
            '<h2 style="font-size: 16px; border: none; padding: 0;'
            ' margin-bottom: 16px;">' + esc(g["label"]) + "</h2>",
            "",
            '<div class="guide-grid">',
        ]
        for i, m in enumerate(g["members"], 1):
            url = permalinks[m["stub"]]
            eyebrow = f"{i}" if g["numbered"] else esc(g["label"])
            out += [
                f"  <a href=\"{{{{ '{url}' | relative_url }}}}\" class=\"guide-card\">",
                f'    <span class="num">{eyebrow}</span>',
                f"    <h3>{esc(m['title'])}</h3>",
                f"    <p>{esc(m['blurb'])}</p>",
                "  </a>",
            ]
        out += ["</div>", ""]

    with open(GUIDE_INDEX_PATH, "w") as f:
        f.write("\n".join(out))
    print("generated site/guide/index.html landing page")


def render_docs_index(groups, sources):
    """Emit docs/README.md — the GitHub-side index and reading order.

    GitHub browsers had no entry point: a flat file list with the reading order
    only visible once you were already inside a tutorial.
    """
    out = [
        "<!-- " + GENERATED_BANNER + ". Do not edit by hand. -->",
        "",
        "# floDl documentation",
        "",
        "The rendered version of these docs, with search and navigation, is at",
        "[flodl.dev/guide](https://flodl.dev/guide/). This page is the same",
        "material in reading order for browsing the repository directly.",
        "",
    ]
    for g in groups:
        out += [f"## {g['label']}", ""]
        for i, m in enumerate(g["members"], 1):
            rel = os.path.relpath(os.path.join(REPO_ROOT, sources[m["stub"]]),
                                  os.path.join(REPO_ROOT, "docs"))
            num = f"{i}. " if g["numbered"] else "- "
            out.append(f"{num}[{m['title']}]({rel})")
        out.append("")

    out += [
        "## Not in the guide",
        "",
        "- [Benchmarks](benchmark.md) - flodl vs PyTorch (summary at"
        " [flodl.dev/benchmark](https://flodl.dev/benchmark))",
        "- [DDP benchmark](ddp-benchmark.md) - multi-GPU results (summary at"
        " [flodl.dev/ddp-benchmark](https://flodl.dev/ddp-benchmark))",
        "- [Distributed architecture](distributed/architecture.md) - internals:"
        " role topology, lifecycle, reduce backends",
        "- [Release process](release.md) - contributor reference",
        "- [`design/`](design/) - design notes and rationale",
        "",
    ]
    with open(DOCS_INDEX_PATH, "w") as f:
        f.write("\n".join(out))
    print("generated docs/README.md index")


NAV_TAIL_MARKER = "<!-- nav: generated by site/build_guide.py — do not edit below -->"


def write_source_nav_tails(chain, sources):
    """Rewrite the Previous/Next tails at the bottom of each docs/ source.

    These tails serve GitHub readers only (the site build strips them and uses
    the frontmatter pair instead), and they were the fourth hand-kept ordering
    in the system — the one that had already drifted to a stale title. Deriving
    them from the manifest makes that class of drift impossible.

    The block is delimited by NAV_TAIL_MARKER so regeneration is exact rather
    than heuristic. On the first run there is no marker yet, so the existing
    hand-written tail is removed with the same heuristic the site build uses.
    """
    changed = 0
    for idx, (stub, _title, _group) in enumerate(chain):
        src_rel = sources[stub]
        path = os.path.join(REPO_ROOT, src_rel)
        src_dir = os.path.dirname(path)

        def link(other_idx):
            other_stub, other_title, _ = chain[other_idx]
            rel = os.path.relpath(os.path.join(REPO_ROOT, sources[other_stub]), src_dir)
            return f"[{other_title}]({rel})"

        parts = []
        if idx > 0:
            parts.append(f"Previous: {link(idx - 1)}")
        if idx + 1 < len(chain):
            parts.append(f"Next: {link(idx + 1)}")
        if not parts:
            continue

        with open(path) as f:
            text = f.read()

        if NAV_TAIL_MARKER in text:
            body = text.split(NAV_TAIL_MARKER)[0]
        else:
            body = strip_trailing_nav(text)

        # Drop trailing rules/blanks so the divider below is written exactly
        # once. Without this the separator accumulates on every run (it would
        # sit inside `body` on the next pass) — the generated block therefore
        # owns the divider and everything after the marker.
        body = body.rstrip()
        while body.endswith("---"):
            body = body[: -len("---")].rstrip()

        new = f"{body}\n\n{NAV_TAIL_MARKER}\n\n---\n\n{' | '.join(parts)}\n"
        if new != text:
            with open(path, "w") as f:
                f.write(new)
            changed += 1

    print(f"regenerated nav tails in {changed} docs/ sources")


def main():
    if not os.path.isdir(STUBS_DIR):
        print(f"error: {STUBS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    sync_skill_assets()

    groups = load_nav()
    chain = nav_chain(groups)

    # Read every stub the manifest names, up front: a manifest entry with no
    # stub (or a stub with no source) is a loud error, not a silent gap.
    stubs, permalinks, sources = {}, {}, {}
    for stub, title, group_label in chain:
        path = os.path.join(STUBS_DIR, f"{stub}.md")
        if not os.path.isfile(path):
            print(f"error: nav names {stub!r} ({group_label}) but {path} is missing",
                  file=sys.stderr)
            sys.exit(1)
        frontmatter, source_rel = parse_stub(path)
        if not frontmatter or not source_rel:
            print(f"error: {path} has no source: in frontmatter", file=sys.stderr)
            sys.exit(1)
        permalink = stub_field(frontmatter, "permalink")
        if not permalink:
            print(f"error: {path} has no permalink:", file=sys.stderr)
            sys.exit(1)
        stubs[stub] = (frontmatter, source_rel)
        permalinks[stub] = permalink
        sources[stub] = source_rel

    # Any stub the manifest forgot would silently vanish from the nav.
    on_disk = {f[:-3] for f in os.listdir(STUBS_DIR) if f.endswith(".md")}
    orphans = sorted(on_disk - set(stubs))
    if orphans:
        print(f"error: stubs missing from site/_data/guide_nav.yml: {', '.join(orphans)}",
              file=sys.stderr)
        sys.exit(1)

    # Settle the sources' own nav tails before reading them for the site build.
    write_source_nav_tails(chain, sources)

    os.makedirs(GUIDE_DIR, exist_ok=True)
    count = 0

    for idx, (stub, title, _group) in enumerate(chain):
        frontmatter, source_rel = stubs[stub]
        source_path = os.path.join(REPO_ROOT, source_rel)
        if not os.path.isfile(source_path):
            print(f"error: source not found: {source_path}", file=sys.stderr)
            sys.exit(1)

        with open(source_path) as f:
            content = f.read()

        content = rewrite_links(content)
        content = strip_trailing_nav(content)
        clean_frontmatter = strip_source_from_frontmatter(frontmatter)

        prev = (permalinks[chain[idx - 1][0]], chain[idx - 1][1]) if idx > 0 else None
        nxt = ((permalinks[chain[idx + 1][0]], chain[idx + 1][1])
               if idx + 1 < len(chain) else None)
        clean_frontmatter = inject_nav(clean_frontmatter, prev, nxt)

        lastmod = git_last_modified(source_path)
        if lastmod:
            clean_frontmatter = inject_last_modified(clean_frontmatter, lastmod)

        with open(os.path.join(GUIDE_DIR, f"{stub}.md"), "w") as f:
            f.write(clean_frontmatter)
            f.write("\n")
            f.write(content)

        count += 1

    render_sidebar(groups, permalinks)
    render_guide_index(groups, permalinks)
    render_docs_index(groups, sources)

    print(f"generated {count} guide pages in site/guide/")


if __name__ == "__main__":
    main()
