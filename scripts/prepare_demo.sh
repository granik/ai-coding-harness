#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TARGET="$ROOT/vendor/cosmic-python"
COMMIT="17215904806967ee80a71ce0c6f771932382ed0f"

if [ ! -d "$TARGET/.git" ]; then
    mkdir -p "$ROOT/vendor"
    git clone --depth 1 --branch chapter_04_service_layer \
        https://github.com/cosmicpython/code.git "$TARGET"
fi

ACTUAL=$(git -C "$TARGET" rev-parse HEAD)
if [ "$ACTUAL" != "$COMMIT" ]; then
    echo "Unexpected demo commit: $ACTUAL" >&2
    echo "Expected: $COMMIT" >&2
    exit 1
fi

echo "Demo repository ready at $TARGET ($ACTUAL)"

