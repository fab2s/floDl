#!/usr/bin/env python3
"""Markdown link, anchor and code-fence validator for the tracked doc corpus.

Four checks:

  1. FENCE BALANCE  -- a file with an unterminated code fence. This is the
     failure a doc *split* creates: a seam landing inside a fenced block
     severs the block in both halves.

  2. LINK TARGETS   -- every relative link/image/reference-definition target
     resolves to a file that exists, relative to the linking file's own
     directory.

  3. ANCHORS        -- every `#fragment` on a relative `.md` target (and every
     same-page `#fragment`) matches a heading in the target file, using
     GitHub's slug algorithm including its duplicate-suffix rule.

  4. GUIDE URLS     -- every hand-written `/guide/...` site URL, anywhere in the
     repo (markdown, HTML, Rust, YAML), matches a real published permalink from
     `site/_stubs/*.md`.

Checks 3 and 4 are the ones that motivated this script, and both guard failures
that had already shipped:

  - A link checker that only verifies the target FILE exists reports a corpus
    clean while anchors rot underneath it. Five broken anchors had accumulated
    unnoticed here, and splitting a doc produces them wholesale.
  - Splitting a doc changes guide URLs, and hand-written `/guide/...` links live
    all over the repo -- blog posts, `site/index.html`, a layout, and the
    `fdl init` scaffold in Rust. Ten of them were live 404s after one split.

FENCE AWARENESS IS LOAD-BEARING, not a nicety. Docs quote command output that
contains `##` lines and shell comments that contain `# 1. Something`. Read
without fence tracking, those become phantom headings -- 30 of them in one file
here -- so a link to a heading that does not exist can validate as fine. The
same blindness inflates section line-counts in any audit built on it.
"""

import os
import re
import subprocess
import sys

# Excluded from all checks:
#   CHANGELOG.md          -- records historical state; its links describe what
#                            files were called at the time and must not be
#                            "fixed" into the present, which would falsify it.
#   docs/design/archive/  -- self-declared historical artifacts. graph-tree.md
#                            opens by warning that its own relative links were
#                            written from a previous location.
#   site/_site, .jekyll-cache -- build output.
EXCLUDE_PREFIXES = (
    "site/_site/",
    "site/.jekyll-cache/",
    "docs/design/archive/",
)
EXCLUDE_FILES = ("CHANGELOG.md",)

# Not repo paths, so checks 2/3 cannot resolve them:
#   `scheme:` / `//`  -- external URLs and protocol links.
#   `/...`            -- site-root-relative URLs, resolved by Jekyll at serve
#                        time. The `/guide/` subset is validated by check 4.
#   `{` / `}`         -- Liquid expressions (`{{ '/x' | relative_url }}`).
SKIP_TARGET = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//|/)", re.IGNORECASE)

FENCE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})(.*)$")
HEADING = re.compile(r"^(\s{0,3})(#{1,6})\s+(.*?)\s*#*\s*$")

# `--fdl-refs` mode: inline-code `fdl <cmd>` tokens, for 03-lint-docs.sh's
# check C. It lives here rather than in shell so the repo has ONE fence-aware
# markdown reader instead of two -- a second one is a second thing that can be
# fence-blind, which is precisely the bug this mode fixes.
FDL_REF = re.compile(r"`fdl ([a-z][a-z0-9-]*)")
FDL_REF_SCOPE = ("docs/", "README.md", "flodl-cli/README.md", "ROADMAP.md")
FDL_REF_EXCLUDE = ("docs/design/",)

# Inline links and images: `[label](target)`, `![alt](target)`.
INLINE_LINK = re.compile(r"!?\[(?:[^\]\\]|\\.)*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'(][^)]*)?\s*\)")
# Link reference definitions: `[label]: target`. `(?!\^)` excludes footnote
# definitions (`[^1]: **Note**: ...`), which are not links at all.
REF_DEF = re.compile(r"^\s{0,3}\[(?!\^)(?:[^\]\\]|\\.)+\]:\s*<?([^\s>]+)>?")
# Raw HTML anchors, which docs use for tables and badges.
HTML_HREF = re.compile(r"""<a\s[^>]*href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
# Explicit ids a heading can carry: kramdown `{#id}` and inline HTML anchors.
KRAMDOWN_ID = re.compile(r"\{#([A-Za-z0-9_-]+)\}\s*$")
HTML_ID = re.compile(r"""<a\s[^>]*(?:name|id)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
CODE_SPAN = re.compile(r"(`+)(.+?)\1")

# Check 4. The leading `(?<![\w.-])` keeps the filesystem path `site/guide/...`
# out of a check about URLs.
#
# A dot is allowed only INSIDE a segment, never at the end. The guide is
# versioned in the URL (`/guide/0.8.x/tensors`) and a class without the dot
# stopped at the first one: every such link matched as the nonexistent
# `/guide/0` and was reported broken, which read as a wave of failures rather
# than as one missing character. Anchoring the dot between word characters is
# what keeps sentence-final punctuation ("see /guide/tensors.") out of the URL.
GUIDE_URL = re.compile(
    r"(?<![\w.-])(/guide(?:/[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)*/?)"
)


def tracked(patterns):
    out = subprocess.run(
        ["git", "ls-files", "-z"] + patterns,
        capture_output=True, text=True, check=True,
    ).stdout
    files = [p for p in out.split("\0") if p]
    return [
        p for p in files
        if not p.startswith(EXCLUDE_PREFIXES) and p not in EXCLUDE_FILES
    ]


def strip_fences(lines):
    """Return (content_lines, unbalanced) where content_lines is a list of
    (lineno, text) with fenced blocks removed.

    A fence closes only on the same marker character and at least the same run
    length, with no info string -- CommonMark's rule. Getting that wrong
    matters: a ``` line inside a ```` block is content, not a close.
    """
    content = []
    open_marker = None  # (char, length)
    for i, line in enumerate(lines, start=1):
        m = FENCE.match(line)
        if m:
            marker, info = m.group(2), m.group(3).strip()
            char, length = marker[0], len(marker)
            if open_marker is None:
                open_marker = (char, length)
                continue
            if char == open_marker[0] and length >= open_marker[1] and not info:
                open_marker = None
                continue
            # A different marker inside an open fence is just content.
            continue
        if open_marker is None:
            content.append((i, line))
    return content, open_marker is not None


def slugify(text):
    """GitHub's heading -> anchor transform.

    Rendered inline markup is removed before slugging (GitHub slugs the
    rendered text, not the source), then everything that is not a word
    character, whitespace or hyphen is dropped and spaces become hyphens.
    """
    s = text.strip()

    # Code-span contents are protected first: inside a code span `<cmd>` is
    # literal text, not an HTML tag. Stripping HTML before unwrapping code
    # spans deletes it -- which mis-slugs a heading like
    # "`fdl @cluster <cmd>` - multi-host fan-out" and reports its live anchor
    # as broken.
    spans = []

    def stash(m):
        spans.append(m.group(2))
        return f"\x00{len(spans) - 1}\x00"

    s = CODE_SPAN.sub(stash, s)
    s = re.sub(r"<[^>]+>", "", s)                      # inline HTML
    s = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", s)   # links -> label
    s = re.sub(r"!?\[([^\]]*)\]\[[^\]]*\]", r"\1", s)  # ref links -> label
    s = re.sub(r"\*{1,3}(?=\S)(.*?)(?<=\S)\*{1,3}", r"\1", s)
    # Underscore emphasis requires non-word delimiters (CommonMark's intra-word
    # rule). Without the guard, `reports_per_epoch` reads as emphasis around
    # "per" and loses its middle word -- and GitHub keeps underscores in slugs,
    # so that difference is the whole anchor.
    s = re.sub(r"(?<!\w)_{1,3}(?=\S)(.*?)(?<=\S)_{1,3}(?!\w)", r"\1", s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], s)

    s = s.lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    # Do NOT strip here. Dropping punctuation can leave whitespace behind, and
    # both renderers hyphenate it rather than trimming it: `## 4. Closures: `|| {}``
    # loses `|`/`{`/`}` and keeps TWO trailing spaces, so kramdown and
    # github-slugger both emit `4-closures--`. A `.strip()` at this point produced
    # `4-closures`, which matches neither — it would pass a link that is broken on
    # both surfaces and fail the one that works. (The heading text is already
    # trimmed at the top of this function; that part is correct.)
    return s.replace(" ", "-")


def anchors_of(path):
    """Every fragment the file offers: heading slugs (with GitHub's -1/-2
    duplicate suffixes) plus explicit kramdown/HTML ids."""
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    content, _ = strip_fences(lines)
    anchors, seen = set(), {}
    for _lineno, line in content:
        for explicit in HTML_ID.findall(line):
            anchors.add(explicit)
        m = HEADING.match(line)
        if not m:
            continue
        text = m.group(3)
        kram = KRAMDOWN_ID.search(text)
        if kram:
            anchors.add(kram.group(1))
            text = KRAMDOWN_ID.sub("", text)
        slug = slugify(text)
        if not slug:
            continue
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        anchors.add(slug if n == 0 else f"{slug}-{n}")
    return anchors


def links_of(content):
    """(lineno, target) for every relative link, image, reference definition
    and HTML href in the fence-stripped content."""
    found = []
    for lineno, line in content:
        for pattern in (INLINE_LINK, REF_DEF, HTML_HREF):
            for target in pattern.findall(line):
                target = target.strip()
                if not target or "{" in target or SKIP_TARGET.match(target):
                    continue
                found.append((lineno, target))
    return found


def published_permalinks(root):
    """Every guide URL that exists, read from the stub frontmatter that defines
    them. `/guide/` itself is the landing page.

    Stubs declare the BARE path (`/guide/tensors`); the site publishes each one
    under a channel segment as well (`/guide/main/tensors`,
    `/guide/0.8.x/tensors`), because the guide is versioned in the URL so links
    written against a release keep resolving. Both spellings are legitimate and
    mean different things: a bare link follows the current release, a
    channel-prefixed one is pinned. So both are accepted here, and the channel
    set is read off the published trees rather than hardcoded.
    """
    urls = {"/guide", "/guide/"}
    stub_dir = os.path.join(root, "site", "_stubs")
    if not os.path.isdir(stub_dir):
        return None

    guide_root = os.path.join(root, "site", "guide")
    channels = ["main"]
    if os.path.isdir(guide_root):
        channels += [d for d in os.listdir(guide_root)
                     if re.match(r"^\d+\.\d+\.x$", d)
                     and os.path.isdir(os.path.join(guide_root, d))]

    bare = []
    for name in os.listdir(stub_dir):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(stub_dir, name), encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"^permalink:\s*(\S+)", line)
                if m:
                    bare.append(m.group(1).rstrip("/"))
                    break

    for url in bare:
        urls.add(url)
        for ch in channels:
            urls.add(url.replace("/guide/", f"/guide/{ch}/", 1))
    for ch in channels:
        urls.add(f"/guide/{ch}")
    return urls


def check_guide_urls(root):
    """Check 4: hand-written /guide/... URLs across the whole repo."""
    urls = published_permalinks(root)
    if urls is None:
        return [], True

    # Every tracked text file, not just markdown: guide URLs live in blog
    # posts, site/index.html, a Jekyll layout, and the Rust `fdl init` scaffold.
    files = tracked([
        "*.md", "*.html", "*.rs", "*.yml", "*.yaml", "*.sh", "*.txt", "*.toml",
    ])
    broken = []
    for rel in files:
        # The generated guide pages are gitignored, but be explicit.
        if rel.startswith("site/guide/"):
            continue
        try:
            with open(os.path.join(root, rel), encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(lines, start=1):
            for url in GUIDE_URL.findall(line):
                if url.rstrip("/") not in urls:
                    broken.append((rel, lineno, url))
    return broken, False


def emit_fdl_refs(root):
    """Print every distinct `fdl <cmd>` command referenced in an inline code
    span in the user-facing docs, one per line, fence-stripped."""
    refs = set()
    for rel in tracked(["*.md"]):
        if not rel.startswith(FDL_REF_SCOPE) or rel.startswith(FDL_REF_EXCLUDE):
            continue
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            content, _ = strip_fences(fh.read().splitlines())
        for _lineno, line in content:
            refs.update(FDL_REF.findall(line))
    for cmd in sorted(refs):
        print(cmd)
    return 0


def main():
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    os.chdir(root)

    if "--fdl-refs" in sys.argv[1:]:
        return emit_fdl_refs(root)

    files = tracked(["*.md"])
    anchor_cache = {}

    def anchors_for(rel):
        if rel not in anchor_cache:
            anchor_cache[rel] = anchors_of(os.path.join(root, rel))
        return anchor_cache[rel]

    unbalanced, missing_files, missing_anchors = [], [], []

    for rel in files:
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        content, odd = strip_fences(lines)
        if odd:
            unbalanced.append(rel)

        src_dir = os.path.dirname(rel)
        for lineno, target in links_of(content):
            path_part, _, fragment = target.partition("#")
            fragment = fragment.strip()

            if not path_part:  # same-page `#anchor`
                if fragment and fragment not in (anchors_for(rel) or set()):
                    missing_anchors.append((rel, lineno, target))
                continue

            path_part = path_part.replace("%20", " ")
            resolved = os.path.normpath(os.path.join(src_dir, path_part))
            if resolved.startswith(".."):  # outside the repo; not ours to check
                continue
            if not os.path.exists(os.path.join(root, resolved)):
                missing_files.append((rel, lineno, target))
                continue
            if fragment and resolved.endswith(".md"):
                available = anchors_for(resolved)
                if available is not None and fragment not in available:
                    missing_anchors.append((rel, lineno, target))

    bad_urls, no_stubs = check_guide_urls(root)

    fail = 0
    if unbalanced:
        print("FAIL: unbalanced code fences (unterminated block):")
        for rel in unbalanced:
            print(f"  {rel}")
        fail = 1
    if missing_files:
        print("FAIL: link targets do not exist:")
        for rel, lineno, target in missing_files:
            print(f"  {rel}:{lineno}  ->  {target}")
        fail = 1
    if missing_anchors:
        print("FAIL: link anchors do not match any heading in the target:")
        for rel, lineno, target in missing_anchors:
            print(f"  {rel}:{lineno}  ->  {target}")
        fail = 1
    if bad_urls:
        print("FAIL: /guide/ URLs with no matching permalink in site/_stubs/:")
        for rel, lineno, url in bad_urls:
            print(f"  {rel}:{lineno}  ->  {url}")
        fail = 1
    if no_stubs:
        print("WARN: site/_stubs/ not found; skipping guide-URL check")

    if not fail:
        print(f"PASS: doc links clean ({len(files)} markdown files, "
              f"fence-aware headings, anchors and /guide/ URLs resolved)")
    return fail


if __name__ == "__main__":
    sys.exit(main())
