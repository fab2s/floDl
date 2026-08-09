#!/bin/sh
# The released version's guide is committed, or the release ships without docs.
#
# site/guide/<series>/ is what flodl.dev serves at /guide/<series>/ and what
# the bare /guide/... paths redirect to. It is generated once, at release, by
#   python3 site/build_guide.py --channel <series>
# and committed. Miss that step and two things happen quietly: the new version
# has no documentation on the site at all, and every bare /guide/ link keeps
# pointing at the PREVIOUS release, which still resolves and therefore never
# looks broken.
#
# That is the failure this gate exists for. A missing snapshot produces no
# error anywhere else in the release: the site builds, the links work, and the
# docs are simply a version behind.

set -u
cd "$(git rev-parse --show-toplevel)"

VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
if [ -z "$VERSION" ]; then
    echo "FAIL: cannot read version from Cargo.toml"
    exit 1
fi

# 0.8.0 -> 0.8.x. The guide is versioned per release SERIES: patch releases
# rarely move documentation, and a tree per patch would multiply permanent
# URLs for nothing.
SERIES=$(printf '%s' "$VERSION" | sed 's/\.[0-9][0-9]*$/.x/')
DIR="site/guide/$SERIES"

if [ ! -d "$DIR" ]; then
    echo "FAIL: $DIR is missing, so flodl.dev would carry no guide for $VERSION"
    echo "  python3 site/build_guide.py --channel $SERIES"
    echo "  git add $DIR site/_includes/sidebar-$SERIES.html site/guide/index.html"
    exit 1
fi

PAGES=$(find "$DIR" -name '*.md' -not -name 'README.md' | wc -l | tr -d ' ')
if [ "$PAGES" -eq 0 ]; then
    echo "FAIL: $DIR exists but holds no pages"
    exit 1
fi

# The sidebar freezes with the tree it belongs to; without it the version's
# pages render with no navigation at all (the layout includes it by channel).
if [ ! -f "site/_includes/sidebar-$SERIES.html" ]; then
    echo "FAIL: site/_includes/sidebar-$SERIES.html is missing"
    echo "  the $SERIES pages would render with no navigation"
    exit 1
fi

# Committed, not merely present: an untracked tree is one `git clean` from
# gone, and it would never reach the deployed site.
UNTRACKED=$(git ls-files --others --exclude-standard "$DIR" | head -1)
if [ -n "$UNTRACKED" ]; then
    echo "FAIL: $DIR has untracked files, e.g. $UNTRACKED"
    echo "  git add $DIR"
    exit 1
fi

echo "PASS: $DIR is committed with $PAGES pages, and its sidebar is present"
