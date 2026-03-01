"""Default skip patterns for technical / non-content files and directories.

This module provides a single source of truth for paths that should be
excluded from indexing and file-watching.  The patterns cover:

  - **Version control** — ``.git``, ``.hg``, ``.svn``, …
  - **Package managers** — ``node_modules``, ``vendor``, ``__pypackages__``,
    ``bower_components``, ``.npm``, ``.yarn``, …
  - **Build & compile outputs** — ``dist``, ``build``, ``out``, ``target``,
    ``__pycache__``, ``*.egg-info``, …
  - **Virtual environments** — ``venv``, ``.venv``, ``.env``, ``.tox``, …
  - **IDE / editor state** — ``.idea``, ``.vscode``, ``.vs``, …
  - **OS junk** — ``.DS_Store``, ``Thumbs.db``, ``desktop.ini``, …
  - **Test & coverage caches** — ``.pytest_cache``, ``.mypy_cache``,
    ``.coverage``, ``htmlcov``, …
  - **Config / lock / secrets** — ``.env`` files, ``package-lock.json``,
    ``poetry.lock``, ``*.pem``, ``*.key``, …
  - **Logs & temp** — ``*.log``, ``*.tmp``, ``*.swp``, …
  - **Binary / media junk** — ``*.min.js``, ``*.map``, ``*.wasm``, …

Two public helpers are exposed:

* :func:`is_skipped_dir`  — fast ``O(1)`` check for directory **names**
* :func:`is_skipped_file` — fast ``O(1)`` check for file **names** plus
  an ``O(n)`` suffix scan for wildcard patterns (n = number of suffix
  patterns, typically < 30)
"""

from __future__ import annotations

# ── directories to skip (matched against the directory *name*, case-insensitive) ──

SKIP_DIRS: frozenset[str] = frozenset(
    {
        # Version control
        ".git",
        ".hg",
        ".svn",
        ".bzr",
        "_darcs",
        ".fossil",
        # Python
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        ".nox",
        ".eggs",
        ".ipynb_checkpoints",
        "site-packages",
        # Virtual environments
        "venv",
        ".venv",
        "env",
        ".env",
        ".virtualenvs",
        "__pypackages__",
        ".pixi",
        ".conda",
        # Node / JS / TS
        "node_modules",
        ".npm",
        ".yarn",
        ".pnpm-store",
        "bower_components",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".parcel-cache",
        ".turbo",
        # Build / distribution outputs
        "dist",
        "build",
        "out",
        "target",
        "bin",
        "obj",
        "_build",
        "cmake-build-debug",
        "cmake-build-release",
        # Rust
        # (target/ already listed above)
        # Go
        "vendor",  # also PHP Composer
        # Java / Gradle / Maven
        ".gradle",
        ".m2",
        # .NET / C#
        "packages",
        # IDE / Editor
        ".idea",
        ".vscode",
        ".vs",
        ".eclipse",
        ".settings",
        ".project",
        # Coverage & profiling
        "htmlcov",
        "coverage",
        ".nyc_output",
        # Docker
        ".docker",
        # Terraform
        ".terraform",
        # Miscellaneous caches
        ".cache",
        ".sass-cache",
        ".eslintcache",
        ".stylelintcache",
        ".angular",
        ".webpack",
    }
)


# ── files to skip (exact name match, case-insensitive) ──

SKIP_FILES: frozenset[str] = frozenset(
    {
        # OS junk
        ".ds_store",
        "thumbs.db",
        "desktop.ini",
        "icon\r",  # macOS Icon? file
        # Git
        ".gitignore",
        ".gitattributes",
        ".gitmodules",
        ".gitkeep",
        ".git-blame-ignore-revs",
        # Editor / IDE
        ".editorconfig",
        # Python packaging
        "setup.cfg",
        "manifest.in",
        "pip-log.txt",
        "pip-delete-this-directory.txt",
        # Node / JS locks & configs
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
        ".npmrc",
        ".yarnrc",
        ".yarnrc.yml",
        ".nvmrc",
        ".node-version",
        ".browserslistrc",
        # Rust
        "cargo.lock",
        # Go
        "go.sum",
        # Ruby
        "gemfile.lock",
        # PHP
        "composer.lock",
        # Python
        "poetry.lock",
        "pdm.lock",
        "pipfile.lock",
        "uv.lock",
        # .NET
        "packages.lock.json",
        # Coverage data files
        ".coverage",
        "coverage.xml",
        "coverage.json",
        ".lcov",
        # Docker
        ".dockerignore",
        # Environment / secrets (avoid indexing secrets!)
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
        ".env.test",
        ".env.staging",
        ".env.example",
        ".flaskenv",
        # Misc config that isn't useful content
        ".eslintrc",
        ".eslintrc.js",
        ".eslintrc.json",
        ".eslintrc.yml",
        ".prettierrc",
        ".prettierrc.js",
        ".prettierrc.json",
        ".prettierrc.yml",
        ".prettierignore",
        ".stylelintrc",
        ".stylelintrc.json",
        ".babelrc",
        ".babelrc.js",
        ".postcssrc",
        ".commitlintrc",
        ".commitlintrc.yml",
        ".huskyrc",
        ".lintstagedrc",
        ".pylintrc",
        ".flake8",
        ".isort.cfg",
        ".mypy.ini",
        ".bandit",
        ".pre-commit-config.yaml",
        ".rubocop.yml",
        ".scalafmt.conf",
        ".clang-format",
        ".clang-tidy",
        ".rustfmt.toml",
        "rustfmt.toml",
        "pyrightconfig.json",
        # Terraform
        ".terraform.lock.hcl",
    }
)


# ── file suffixes to skip (matched against the lowercased file name) ──

SKIP_SUFFIXES: tuple[str, ...] = (
    # Lock / compiled Python
    ".pyc",
    ".pyo",
    ".pyd",
    # Compiled / object files
    ".o",
    ".obj",
    ".a",
    ".lib",
    ".so",
    ".dll",
    ".dylib",
    ".class",
    ".jar",
    ".war",
    ".ear",
    # Archives (not parseable content)
    ".zip",  # note: epub/docx/odt use zip internally but have their own extensions
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".tgz",
    ".tar.gz",
    # Executables / binaries
    ".exe",
    ".msi",
    ".app",
    ".bin",
    ".wasm",
    # Minified / generated JS/CSS
    ".min.js",
    ".min.css",
    ".map",  # source maps
    ".chunk.js",
    ".bundle.js",
    # Images (unless you want OCR — handled by image_parser separately)
    # Skipping common image formats that aren't documents
    ".ico",
    ".icns",
    ".cur",
    ".ani",
    # Fonts
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    # Media
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".wav",
    ".flac",
    ".ogg",
    ".webm",
    ".m4a",
    ".aac",
    # Database files
    ".sqlite",
    ".sqlite3",
    ".db",
    ".db-shm",
    ".db-wal",
    ".mdb",
    ".ldb",
    # Keys / certs (avoid indexing secrets!)
    ".pem",
    ".key",
    ".crt",
    ".cer",
    ".p12",
    ".pfx",
    ".jks",
    # Temp / swap / backup
    ".tmp",
    ".temp",
    ".swp",
    ".swo",
    ".swn",
    ".bak",
    ".orig",
    ".rej",
    "~",  # Emacs backup files (e.g. foo.py~)
    # Python eggs
    ".egg-info",
    ".egg",
    ".whl",
    ".dist-info",
    # Misc generated
    ".tfstate",
    ".tfstate.backup",
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_skipped_dir(dirname: str) -> bool:
    """Return True if *dirname* (bare name, not full path) should be skipped.

    The check is case-insensitive and ``O(1)`` via frozenset lookup.
    Also skips any directory whose name starts with ``.`` and is in the set,
    or any ``*.egg-info`` directory.
    """
    lower = dirname.lower()
    if lower in SKIP_DIRS:
        return True
    # Catch *.egg-info directories (e.g. mypackage.egg-info)
    return bool(lower.endswith(".egg-info"))


def is_skipped_file(filename: str) -> bool:
    """Return True if *filename* (bare name, not full path) should be skipped.

    Performs an ``O(1)`` exact-name check, then an ``O(n)`` suffix scan
    for wildcard patterns.
    """
    lower = filename.lower()
    if lower in SKIP_FILES:
        return True
    return any(lower.endswith(suffix) for suffix in SKIP_SUFFIXES)


def is_skipped_path(path: str) -> bool:
    """Return True if any component of *path* matches the skip rules.

    This is the most general check — it splits the path and tests each
    directory component with :func:`is_skipped_dir` and the final
    filename with :func:`is_skipped_file`.  Suitable for the file
    watcher where you receive a full path.
    """
    parts = _split_path(path)
    if not parts:
        return False
    # Check all directory components
    for part in parts[:-1]:
        if is_skipped_dir(part):
            return True
    # Check the filename
    return is_skipped_file(parts[-1])


def _split_path(path: str) -> list[str]:
    """Split a file path into components, handling both / and \\\\ separators."""
    # Normalise to forward slashes, then split
    normalised = path.replace("\\", "/")
    return [p for p in normalised.split("/") if p]
