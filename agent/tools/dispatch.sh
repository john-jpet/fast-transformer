#!/bin/sh
# Put one of our engine trees on a merged-team queue: dispatch.sh <mate|silver> <sha> "<message>"
# A merge commit keeps their history (their main is a parent), so the push is a fast-forward.
# Every push starts one official run on THAT team's queue; their result is visible only as
# their leaderboard best.
set -e
case "$1" in
  mate)   url=https://github.com/john-jpet/fast-transformer ;;
  silver) url=https://github.com/sivakovivan/silver-transformer ;;
  *) echo "usage: dispatch.sh <mate|silver> <sha> <message>" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/../.."
git fetch -q "$url" "+main:refs/remotes/$1/main"
merge=$(git commit-tree "$2^{tree}" -p "$2" -p "refs/remotes/$1/main" -m "$3")
git update-ref "refs/heads/dispatch-$1" "$merge"
git push "$url" "dispatch-$1:main"
