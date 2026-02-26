<#
.SYNOPSIS
    Build rag-kb standalone (one-folder) executable using Nuitka.

.DESCRIPTION
    Produces a one-folder distribution using Nuitka that bundles:
      - Python interpreter & stdlib
      - All runtime dependencies (as bytecode — NOT compiled to C)
      - Only rag_kb is compiled to native C for speed
      - Pre-downloaded ML models

    This avoids the extremely long compile times from trying to
    C-compile heavy packages like torch, transformers, chromadb, etc.

.EXAMPLE
    .\build-nuitka.ps1
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Write-Err($msg)  { Write-Host "    ERROR: $msg" -ForegroundColor Red }

Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Yellow
Write-Host "║   rag-knowledge-base  Nuitka Builder     ║" -ForegroundColor Yellow
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Yellow

# ── Prerequisites ─────────────────────────────────────────────
Write-Step "Checking prerequisites"
pip install --quiet nuitka ordered-set zstandard
Write-Ok "Nuitka installed"

# CPU-only torch (smaller bundle)
Write-Step "Installing lightweight dependencies for bundling"
pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps --quiet
Write-Ok "torch CPU-only installed"
pip install opencv-python-headless --force-reinstall --quiet
Write-Ok "opencv-python-headless installed"

# Pre-download models
Write-Step "Downloading ML models for bundling"
Push-Location $ProjectRoot
python -c "from rag_kb.models import download_models; download_models()"
Write-Ok "Models downloaded to models/"
Pop-Location

# ── Build ─────────────────────────────────────────────────────
Write-Step "Building with Nuitka (standalone one-folder mode)"

# Strategy: compile ONLY rag_kb to C. Everything else is included
# as Python bytecode via --include-package. This keeps the build
# fast (~5 min) while still producing a standalone distribution.

$nuitkaArgs = @(
    # ── Mode: one-folder standalone ──
    "--standalone"
    "--output-dir=dist-nuitka"
    "--output-filename=rag-kb.exe"
    "--windows-console-mode=force"

    # ── Compile only our code to C; third-party stays as bytecode ──
    "--follow-import-to=rag_kb"

    # ── Force-include packages (stays as bytecode since not in --follow-import-to) ──
    "--include-package=rag_kb"
    "--include-package=rag_kb.parsers"
    "--include-package=sentence_transformers"
    "--include-package=transformers"
    "--include-package=tokenizers"
    "--include-package=huggingface_hub"
    "--include-package=safetensors"
    "--include-package=chromadb"
    "--include-package=mcp"
    "--include-package=rich"
    "--include-package=pydantic"
    "--include-package=pydantic_settings"
    "--include-package=pydantic_core"
    "--include-package=watchdog"
    "--include-package=yaml"
    "--include-package=PIL"
    "--include-package=pypdf"
    "--include-package=docx"
    "--include-package=pptx"
    "--include-package=bs4"
    "--include-package=lxml"
    "--include-package=openpyxl"
    "--include-package=striprtf"
    "--include-package=rapidocr"
    "--include-package=numpy"
    "--include-package=rank_bm25"
    "--include-package=xxhash"
    "--include-package=torch"
    "--include-package=certifi"
    "--include-package=charset_normalizer"
    "--include-package=requests"
    "--include-package=urllib3"
    "--include-package=pygments"
    "--include-package=click"
    "--include-package=orjson"
    "--include-package=regex"

    # PySide6 — only what we use
    "--include-module=PySide6"
    "--include-module=PySide6.QtCore"
    "--include-module=PySide6.QtGui"
    "--include-module=PySide6.QtWidgets"

    # Watchdog Windows observer
    "--include-module=watchdog.observers.read_directory_changes"

    # ── Include data files ──
    "--include-data-dir=models=models"
    "--include-package-data=sentence_transformers"
    "--include-package-data=transformers"
    "--include-package-data=chromadb"
    "--include-package-data=pydantic"
    "--include-package-data=rich"
    "--include-package-data=rapidocr"
    "--include-package-data=certifi"

    # ── Exclude heavy unused modules ──
    "--nofollow-import-to=tkinter"
    "--nofollow-import-to=unittest"
    "--nofollow-import-to=test"
    "--nofollow-import-to=pytest"
    "--nofollow-import-to=matplotlib"
    "--nofollow-import-to=scipy"
    "--nofollow-import-to=pandas"
    "--nofollow-import-to=sklearn"
    "--nofollow-import-to=IPython"
    "--nofollow-import-to=notebook"
    "--nofollow-import-to=jupyter"
    "--nofollow-import-to=gradio"

    # Heavy torch subpackages we don't need
    "--nofollow-import-to=torch.distributed"
    "--nofollow-import-to=torch.testing"
    "--nofollow-import-to=torch._dynamo"
    "--nofollow-import-to=torch._inductor"
    "--nofollow-import-to=torch._export"
    "--nofollow-import-to=torch.onnx"
    "--nofollow-import-to=torch.utils.tensorboard"
    "--nofollow-import-to=torch._functorch"
    "--nofollow-import-to=torch.sparse"
    "--nofollow-import-to=torch.ao"
    "--nofollow-import-to=torch.package"
    "--nofollow-import-to=torch.fx"
    "--nofollow-import-to=torch.profiler"
    "--nofollow-import-to=torch.optim"
    "--nofollow-import-to=torch.jit"

    # Unused PySide6 modules
    "--nofollow-import-to=PySide6.Qt3DAnimation"
    "--nofollow-import-to=PySide6.Qt3DCore"
    "--nofollow-import-to=PySide6.QtBluetooth"
    "--nofollow-import-to=PySide6.QtCharts"
    "--nofollow-import-to=PySide6.QtMultimedia"
    "--nofollow-import-to=PySide6.QtNetwork"
    "--nofollow-import-to=PySide6.QtOpenGL"
    "--nofollow-import-to=PySide6.QtQml"
    "--nofollow-import-to=PySide6.QtQuick"
    "--nofollow-import-to=PySide6.QtSvg"
    "--nofollow-import-to=PySide6.QtWebEngine"
    "--nofollow-import-to=PySide6.QtWebSockets"

    # Misc unused
    "--nofollow-import-to=chromadb.server"
    "--nofollow-import-to=transformers.cli"

    # ── Misc settings ──
    "--assume-yes-for-downloads"
    "--no-pyi-file"

    # Entry point
    "src/rag_kb/cli.py"
)

Push-Location $ProjectRoot
try {
    $env:PYTHONPATH = "src"
    python -m nuitka @nuitkaArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Nuitka build failed with exit code $LASTEXITCODE"
        exit 1
    }
    Write-Ok "Executable created in dist-nuitka/cli.dist/"
    Write-Host "    Run with: .\dist-nuitka\cli.dist\rag-kb.exe --help"
}
finally {
    Pop-Location
}

Write-Host "`nDone!" -ForegroundColor Green
