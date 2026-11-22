#!/usr/bin/env bash
# Formats project sources under src/ with clang-format.
# Never touches third_party/.
#
# Usage:
#   scripts/format.sh           format files in place
#   scripts/format.sh --check   change nothing, exit non-zero if any file is not formatted
#
# Override the binary with CLANG_FORMAT=/path/to/clang-format.
# Ubuntu: sudo apt install clang-format

set -euo pipefail

check=0
for arg in "$@"; do
    case "$arg" in
        --check) check=1 ;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *)
            echo "format.sh: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

clang_format="${CLANG_FORMAT:-clang-format}"
if ! command -v "$clang_format" >/dev/null 2>&1; then
    echo "format.sh: '$clang_format' not found (Ubuntu: sudo apt install clang-format)" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mapfile -d '' files < <(find src -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) -print0 | sort -z)

if [ "${#files[@]}" -eq 0 ]; then
    echo "format.sh: no source files found under src/"
    exit 0
fi

if [ "$check" -eq 1 ]; then
    if "$clang_format" --style=file --dry-run --Werror "${files[@]}"; then
        echo "format.sh: ${#files[@]} file(s) checked, all formatted"
    else
        echo "format.sh: some files are not formatted, run scripts/format.sh" >&2
        exit 1
    fi
else
    "$clang_format" --style=file -i "${files[@]}"
    echo "format.sh: ${#files[@]} file(s) formatted"
fi
