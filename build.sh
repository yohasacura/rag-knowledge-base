#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Build self-contained packages for rag-knowledge-base
#
# Usage:
#   ./build.sh wheel    # Python wheel + sdist
#   ./build.sh exe      # Standalone executable (PyInstaller)
#   ./build.sh docker   # Docker container image
#   ./build.sh all      # All of the above
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

step()  { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
ok()    { printf '    \033[32mOK: %s\033[0m\n' "$1"; }
err()   { printf '    \033[31mERROR: %s\033[0m\n' "$1"; }

# ── Wheel + sdist ────────────────────────────────────────────
build_wheel() {
    step "Building wheel + sdist"
    pip install --quiet build
    rm -rf "$SCRIPT_DIR/dist"
    (cd "$SCRIPT_DIR" && python -m build)
    ok "Packages created in dist/"
    ls -lh "$SCRIPT_DIR/dist/"
}

# ── PyInstaller executable ───────────────────────────────────
build_exe() {
    step "Building standalone executable (PyInstaller)"
    pip install --quiet pyinstaller

    # Install lightweight package alternatives to minimize bundle size
    step "Installing lightweight dependencies for bundling"

    # CPU-only torch (~150 MB instead of ~3.8 GB with CUDA)
    pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps --quiet
    ok "torch CPU-only installed"

    # Headless OpenCV (~40 MB instead of ~109 MB — no GUI/ffmpeg)
    pip install opencv-python-headless --force-reinstall --quiet
    ok "opencv-python-headless installed"

    # Pre-download models so they get bundled into the executable
    step "Downloading ML models for bundling"
    (cd "$SCRIPT_DIR" && python -c "from rag_kb.models import download_models; download_models()")
    ok "Models downloaded to models/"

    (cd "$SCRIPT_DIR" && pyinstaller rag-kb.spec --noconfirm)
    ok "Executable created in dist/rag-kb/"
    echo "    Run with: ./dist/rag-kb/rag-kb --help"
}

# ── Docker image ─────────────────────────────────────────────
build_docker() {
    step "Building Docker image"
    if ! command -v docker &>/dev/null; then
        err "Docker is not installed or not in PATH"
        return 1
    fi
    (cd "$SCRIPT_DIR" && docker build -t rag-kb:latest .)
    ok "Docker image built: rag-kb:latest"
    echo "    Run with: docker run --rm -it -v /path/to/docs:/data rag-kb serve"
}

# ── Main ─────────────────────────────────────────────────────
TARGET="${1:-all}"

echo "╔══════════════════════════════════════════╗"
echo "║   rag-knowledge-base  Package Builder    ║"
echo "╚══════════════════════════════════════════╝"

case "$TARGET" in
    wheel)  build_wheel ;;
    exe)    build_exe ;;
    docker) build_docker ;;
    all)    build_wheel; build_exe; build_docker ;;
    *)      echo "Usage: $0 {wheel|exe|docker|all}"; exit 1 ;;
esac

echo -e "\nDone!"
