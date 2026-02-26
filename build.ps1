<#
.SYNOPSIS
    Build self-contained packages for rag-knowledge-base.

.DESCRIPTION
    Supports three build targets:
      - wheel   : Python wheel + sdist (pip-installable)
      - exe     : Standalone executable via PyInstaller (no Python needed)
      - docker  : Docker container image

.EXAMPLE
    .\build.ps1 wheel
    .\build.ps1 exe
    .\build.ps1 docker
    .\build.ps1 all
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet("wheel", "exe", "docker", "all")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Write-Err($msg)  { Write-Host "    ERROR: $msg" -ForegroundColor Red }

# ── Wheel + sdist ────────────────────────────────────────────
function Build-Wheel {
    Write-Step "Building wheel + sdist"

    # Ensure build tool is installed
    pip install --quiet build

    # Clean previous builds
    if (Test-Path "$ProjectRoot\dist") {
        Remove-Item -Recurse -Force "$ProjectRoot\dist"
    }

    Push-Location $ProjectRoot
    try {
        python -m build
        Write-Ok "Packages created in dist/"
        Get-ChildItem dist/ | ForEach-Object { Write-Host "    $_" }
    }
    finally {
        Pop-Location
    }
}

# ── PyInstaller executable ───────────────────────────────────
function Build-Exe {
    Write-Step "Building standalone executable (PyInstaller)"

    pip install --quiet pyinstaller

    # Install lightweight package alternatives to minimize bundle size
    Write-Step "Installing lightweight dependencies for bundling"

    # CPU-only torch (~150 MB instead of ~3.8 GB with CUDA)
    pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps --quiet
    Write-Ok "torch CPU-only installed"

    # Headless OpenCV (~40 MB instead of ~109 MB — no GUI/ffmpeg)
    pip install opencv-python-headless --force-reinstall --quiet
    Write-Ok "opencv-python-headless installed"

    # Pre-download models so they get bundled into the executable
    Write-Step "Downloading ML models for bundling"
    Push-Location $ProjectRoot
    try {
        python -c "from rag_kb.models import download_models; download_models()"
        Write-Ok "Models downloaded to models/"

        pyinstaller rag-kb.spec --noconfirm
        Write-Ok "Executable created in dist/rag-kb/"
        Write-Host "    Run with: .\dist\rag-kb\rag-kb.exe --help"
    }
    finally {
        Pop-Location
    }
}

# ── Docker image ─────────────────────────────────────────────
function Build-Docker {
    Write-Step "Building Docker image"

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Err "Docker is not installed or not in PATH"
        return
    }

    Push-Location $ProjectRoot
    try {
        docker build -t rag-kb:latest .
        Write-Ok "Docker image built: rag-kb:latest"
        Write-Host "    Run with: docker run --rm -it -v /path/to/docs:/data rag-kb serve"
    }
    finally {
        Pop-Location
    }
}

# ── Main ─────────────────────────────────────────────────────
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Yellow
Write-Host "║   rag-knowledge-base  Package Builder    ║" -ForegroundColor Yellow
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Yellow

switch ($Target) {
    "wheel"  { Build-Wheel }
    "exe"    { Build-Exe }
    "docker" { Build-Docker }
    "all"    {
        Build-Wheel
        Build-Exe
        Build-Docker
    }
}

Write-Host "`nDone!" -ForegroundColor Green
