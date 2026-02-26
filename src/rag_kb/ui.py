"""Cross-platform desktop UI for managing RAG knowledge bases (PySide6)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Protocol, cast

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QFont, QIcon, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QStackedWidget,
    QListWidget,
    QListWidgetItem,
    QFrame,
    QScrollArea,
    QFileDialog,
    QMessageBox,
    QHeaderView,
    QProgressBar,
    QGroupBox,
    QGridLayout,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
)

from rag_kb.config import (
    AppSettings,
    RagRegistry,
    safe_display_path,
    CONFIG_PATH,
    DATA_DIR,
)
from rag_kb.embedder import embed_query
from rag_kb.indexer import Indexer, IndexingState
from rag_kb.search import bm25_search, hybrid_fuse_scores, mmr_diversify, rerank_cross_encoder
from rag_kb.sharing import export_rag, import_rag
from rag_kb.vector_store import SearchResult, VectorStore
from rag_kb.watcher import FolderWatcher

logger = logging.getLogger(__name__)


class _Refreshable(Protocol):
    def refresh(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

_STYLESHEET = """
/* ===== Main Window ===== */
QMainWindow {
    background-color: #14181f;
}

QWidget {
    color: #e4e8ef;
    font-size: 13px;
}

/* ===== Sidebar ===== */
QWidget#sidebar {
    background-color: #11151c;
}

QLabel#sidebar-title {
    color: #e8ecf0;
    font-size: 15px;
    font-weight: bold;
    padding: 18px 16px 4px 16px;
}

QLabel#sidebar-subtitle {
    color: #96a7bb;
    font-size: 11px;
    padding: 0px 16px 14px 16px;
}

QListWidget#nav-list {
    background-color: transparent;
    border: none;
    color: #9daec1;
    font-size: 13px;
    outline: none;
}

QListWidget#nav-list::item {
    padding: 11px 18px;
    border-left: 3px solid transparent;
    margin: 1px 0px;
}

QListWidget#nav-list::item:selected {
    background-color: rgba(103, 211, 255, 0.18);
    color: #e8ecf0;
    border-left: 3px solid #67d3ff;
    font-weight: bold;
}

QListWidget#nav-list::item:hover:!selected {
    background-color: rgba(255, 255, 255, 0.05);
    color: #b0c4d8;
}

/* ===== Page titles ===== */
QLabel#page-title {
    font-size: 22px;
    font-weight: bold;
    color: #e8ecf0;
    padding: 2px 0px;
}

QLabel#page-subtitle {
    font-size: 13px;
    color: #9daec1;
    padding-bottom: 8px;
}

/* ===== Stat cards ===== */
QFrame#stat-card {
    background-color: #1d232e;
    border: 1px solid #313949;
    border-radius: 10px;
}

QLabel#stat-value {
    font-size: 28px;
    font-weight: bold;
    color: #67d3ff;
}

QLabel#stat-label {
    font-size: 12px;
    color: #9daec1;
}

/* ===== Buttons ===== */
QPushButton#primary-btn {
    background-color: #2f79c6;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton#primary-btn:hover {
    background-color: #3f8cdd;
}

QPushButton#primary-btn:pressed {
    background-color: #2368ab;
}

QPushButton#primary-btn:disabled {
    background-color: #3a4450;
    color: #5a6a78;
}

QPushButton#secondary-btn {
    background-color: #1d232e;
    color: #d5deea;
    border: 1px solid #3e4a5f;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 13px;
}

QPushButton#secondary-btn:hover {
    background-color: #353a44;
    border-color: #4a5568;
}

QPushButton#danger-btn {
    background-color: #c0392b;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton#danger-btn:hover {
    background-color: #e04a3a;
}

QPushButton#danger-btn:disabled {
    background-color: #3a2a28;
    color: #6a4a48;
}

/* ===== Input fields ===== */
QLineEdit, QPlainTextEdit {
    border: 1px solid #3e4a5f;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 13px;
    background-color: #1d232e;
    color: #e4e8ef;
    selection-background-color: #2f79c6;
    selection-color: #ffffff;
    placeholder-text-color: #8fa2b8;
}

QLineEdit:focus, QPlainTextEdit:focus {
    border-color: #67d3ff;
    outline: none;
}

/* ===== ComboBox (fully styled with working dropdown) ===== */
QComboBox {
    border: 1px solid #3e4a5f;
    border-radius: 6px;
    padding: 7px 10px;
    padding-right: 30px;
    font-size: 13px;
    background-color: #1d232e;
    color: #e4e8ef;
    min-height: 18px;
}

QComboBox:hover {
    border-color: #4a5568;
}

QComboBox:focus, QComboBox:on {
    border-color: #67d3ff;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 28px;
    border: none;
    border-left: 1px solid #3a4050;
    background-color: transparent;
}

QComboBox::down-arrow {
    width: 10px;
    height: 10px;
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #788a9a;
    margin-right: 6px;
}

QComboBox::down-arrow:hover {
    border-top-color: #4fc3f7;
}

QComboBox QAbstractItemView {
    background-color: #1d232e;
    border: 1px solid #3e4a5f;
    color: #e4e8ef;
    selection-background-color: #2f79c6;
    selection-color: #ffffff;
    padding: 4px;
    outline: none;
}

QComboBox QAbstractItemView::item {
    padding: 6px 10px;
    min-height: 24px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #2a3540;
}

/* ===== SpinBox ===== */
QSpinBox, QDoubleSpinBox {
    border: 1px solid #3e4a5f;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 13px;
    background-color: #1d232e;
    color: #e4e8ef;
    selection-background-color: #2f79c6;
    selection-color: #ffffff;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #67d3ff;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border: none;
    border-left: 1px solid #3a4050;
    border-bottom: 1px solid #3a4050;
    background-color: #2a2e36;
    border-top-right-radius: 5px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: #353a44;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border: none;
    border-left: 1px solid #3a4050;
    background-color: #2a2e36;
    border-bottom-right-radius: 5px;
}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #353a44;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 8px;
    height: 8px;
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #788a9a;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 8px;
    height: 8px;
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #788a9a;
}

/* ===== Tables ===== */
QTableWidget {
    background-color: #1d232e;
    border: 1px solid #313949;
    border-radius: 8px;
    gridline-color: #2e3440;
    font-size: 13px;
    color: #e4e8ef;
    alternate-background-color: #252930;
}

QTableWidget::item {
    padding: 6px 10px;
}

QTableWidget::item:selected {
    background-color: #285d89;
    color: #e8ecf0;
}

QHeaderView::section {
    background-color: #1a1d23;
    color: #788a9a;
    padding: 8px 10px;
    border: none;
    border-bottom: 2px solid #2e3440;
    font-weight: bold;
    font-size: 12px;
}

/* ===== Group boxes ===== */
QGroupBox {
    font-size: 14px;
    font-weight: bold;
    color: #4fc3f7;
    border: 1px solid #2e3440;
    border-radius: 8px;
    margin-top: 14px;
    padding: 20px 14px 14px 14px;
    background-color: #22262e;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
}

/* ===== Result cards ===== */
QFrame#result-card {
    background-color: #1d232e;
    border: 1px solid #313949;
    border-left: 4px solid #2f79c6;
    border-radius: 0px 8px 8px 0px;
}

/* ===== Scroll area ===== */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #1a1d23;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #3a4050;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #4a5568;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #1a1d23;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: #3a4050;
    border-radius: 5px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #4a5568;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ===== Status bar ===== */
QStatusBar {
    background-color: #12151a;
    color: #788a9a;
    font-size: 12px;
    padding: 2px 8px;
    border-top: 1px solid #2e3440;
}

/* ===== Progress bar ===== */
QProgressBar {
    border: 1px solid #2e3440;
    border-radius: 4px;
    text-align: center;
    font-size: 11px;
    background-color: #22262e;
    color: #d4d8e0;
    max-height: 18px;
}

QProgressBar::chunk {
    background-color: #2f79c6;
    border-radius: 3px;
}

/* ===== Details panel ===== */
QFrame#details-panel {
    background-color: #1d232e;
    border: 1px solid #313949;
    border-radius: 10px;
}

QLabel#detail-key {
    color: #9daec1;
    font-size: 12px;
}

QLabel#detail-val {
    color: #d4d8e0;
    font-size: 13px;
    font-weight: bold;
}

/* ===== Folder list ===== */
QListWidget#folder-list {
    background-color: #1d232e;
    border: 1px solid #3e4a5f;
    border-radius: 6px;
    font-size: 13px;
    color: #d4d8e0;
}

QListWidget#folder-list::item:selected {
    background-color: #1e4a6e;
    color: #e8ecf0;
}

/* ===== Config text ===== */
QPlainTextEdit#config-text {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 13px;
    background-color: #1a1d23;
    border: 1px solid #2e3440;
    border-radius: 8px;
    padding: 10px;
    color: #c0c8d4;
}

/* ===== Checkboxes ===== */
QCheckBox {
    font-size: 13px;
    spacing: 8px;
    color: #d4d8e0;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #3e4a5f;
    background-color: #1d232e;
}

QCheckBox::indicator:checked {
    background-color: #2f79c6;
    border-color: #2f79c6;
}

QCheckBox::indicator:hover {
    border-color: #67d3ff;
}

/* ===== Slider ===== */
QSlider::groove:horizontal {
    height: 6px;
    background: #2e3440;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -5px 0;
    background: #2f79c6;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #67d3ff;
}

QSlider::sub-page:horizontal {
    background: #2f79c6;
    border-radius: 3px;
}

QPushButton:focus, QLineEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QListWidget:focus,
QTableWidget:focus, QPlainTextEdit:focus {
    border-color: #67d3ff;
}

/* ===== Splitter ===== */
QSplitter::handle {
    background-color: #2e3440;
}

/* ===== Config section header ===== */
QLabel#config-section {
    font-size: 14px;
    font-weight: bold;
    color: #4fc3f7;
    padding: 10px 0px 2px 0px;
}

QLabel#config-hint {
    font-size: 11px;
    color: #5a6a78;
    padding: 0px 0px 2px 0px;
}

/* ===== Success/warning labels ===== */
QLabel#success-msg {
    color: #66bb6a;
    font-weight: bold;
    font-size: 13px;
    padding: 4px 0px;
}

QLabel#error-msg {
    color: #ef5350;
    font-weight: bold;
    font-size: 13px;
    padding: 4px 0px;
}

/* ===== Message boxes ===== */
QMessageBox {
    background-color: #22262e;
}

QMessageBox QLabel {
    color: #d4d8e0;
}

QMessageBox QPushButton {
    background-color: #2a2e36;
    color: #d4d8e0;
    border: 1px solid #3a4050;
    border-radius: 6px;
    padding: 6px 18px;
    min-width: 70px;
}

QMessageBox QPushButton:hover {
    background-color: #353a44;
}

/* ===== Tooltips ===== */
QToolTip {
    background-color: #2a2e36;
    color: #d4d8e0;
    border: 1px solid #3a4050;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
}

/* ===== Status badges (Manage page) ===== */
QLabel#badge-active {
    background-color: #1b5e20;
    color: #a5d6a7;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}

QLabel#badge-local {
    background-color: #1a237e;
    color: #9fa8da;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}

QLabel#badge-imported {
    background-color: #4e342e;
    color: #bcaaa4;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}

QLabel#badge-detached {
    background-color: #4a3728;
    color: #ffcc80;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}

/* ===== Detail panel (Manage page) ===== */
QFrame#rag-detail-panel {
    background-color: #1d232e;
    border: 1px solid #313949;
    border-radius: 10px;
}

QLabel#detail-section-title {
    font-size: 13px;
    font-weight: bold;
    color: #4fc3f7;
    padding: 2px 0px;
}

QLabel#change-badge-new {
    background-color: #1b3a2a;
    color: #66bb6a;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}

QLabel#change-badge-modified {
    background-color: #3e2f1e;
    color: #ffa726;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}

QLabel#change-badge-removed {
    background-color: #3b1c1c;
    color: #ef5350;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}

QLabel#change-badge-ok {
    background-color: #1b3a2a;
    color: #66bb6a;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: bold;
}

QListWidget#change-file-list {
    background-color: #151b24;
    border: 1px solid #313949;
    border-radius: 4px;
    color: #b0bec5;
    font-size: 11px;
    padding: 2px;
}

/* ===== Dialog ===== */
QDialog {
    background-color: #1d232e;
}

QDialog QLabel {
    color: #d4d8e0;
}

QDialogButtonBox QPushButton {
    background-color: #2a2e36;
    color: #d4d8e0;
    border: 1px solid #3a4050;
    border-radius: 6px;
    padding: 6px 18px;
    min-width: 70px;
}

QDialogButtonBox QPushButton:hover {
    background-color: #353a44;
}
"""


# ---------------------------------------------------------------------------
# Shared state (per-process)
# ---------------------------------------------------------------------------

_settings: AppSettings | None = None
_registry: RagRegistry | None = None
_watcher: FolderWatcher | None = None
_last_index_state: IndexingState | None = None
_store_cache: dict[str, VectorStore] = {}  # normalised db_path → VectorStore

EMBEDDING_MODEL_OPTIONS: list[str] = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
]

RERANKER_MODEL_OPTIONS: list[str] = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/ms-marco-electra-base",
    "BAAI/bge-reranker-base",
]


def _init() -> tuple[AppSettings, RagRegistry]:
    global _settings, _registry
    if _settings is None:
        _settings = AppSettings.load()
    if _registry is None:
        _registry = RagRegistry()
    return _settings, _registry


def _get_store(entry) -> VectorStore | None:
    """Return a **cached** VectorStore for the given RAG entry.

    Re-uses the same ChromaDB PersistentClient for a given db_path so
    that only one set of memory-mapped HNSW files is ever open.
    Callers should **not** call ``store.close()`` — the cache owns the
    lifecycle.  Use ``_close_store(db_path)`` to explicitly release a
    store (e.g. before deleting the RAG directory).
    """
    if entry is None:
        return None
    import os
    norm = os.path.normcase(os.path.normpath(entry.db_path))
    if norm in _store_cache:
        cached = _store_cache[norm]
        # Guard against a previously force-closed store still in cache
        if cached._client is not None:
            return cached
        else:
            del _store_cache[norm]
    s, _ = _init()
    store = VectorStore(
        entry.db_path,
        hnsw_ef_construction=s.hnsw_ef_construction,
        hnsw_m=s.hnsw_m,
    )
    _store_cache[norm] = store
    return store


def _close_store(db_path: str, *, force: bool = False) -> None:
    """Close and remove a cached VectorStore for *db_path*.

    Parameters
    ----------
    force : bool
        Passed to ``VectorStore.close()`` — when *True*, also stops the
        internal ChromaDB system to release SQLite / HNSW file locks.
    """
    import os
    norm = os.path.normcase(os.path.normpath(db_path))
    store = _store_cache.pop(norm, None)
    if store is not None:
        store.close(force=force)


def _source_folders() -> list[str]:
    _, registry = _init()
    active = registry.get_active()
    return active.source_folders if active else []


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------


class IndexWorker(QThread):
    """Run indexing in a background thread to keep the UI responsive."""

    finished = Signal(object)        # IndexingState
    progress = Signal(str)           # progress message (legacy)
    detail_progress = Signal(dict)   # {phase, pct, processed, total, file}
    error = Signal(str)

    def __init__(
        self, entry, registry, settings, full: bool = False, parent=None
    ):
        super().__init__(parent)
        self.entry = entry
        self.registry = registry
        self.settings = settings
        self.full = full

    def run(self):  # noqa: D401
        try:
            def on_progress(state):
                pct = int(state.progress * 100)
                self.detail_progress.emit({
                    "phase": state.status,
                    "pct": pct,
                    "processed": state.processed_files,
                    "total": state.total_files,
                    "file": state.current_file,
                })
                if state.current_file:
                    self.progress.emit(
                        f"[{state.processed_files}/{state.total_files}] "
                        f"{state.current_file}"
                    )

            indexer = Indexer(
                self.entry, self.registry, self.settings, on_progress=on_progress
            )
            state = indexer.index(full=self.full)
            self.finished.emit(state)
        except Exception as exc:
            self.error.emit(str(exc))


class SearchWorker(QThread):
    """Run search in a background thread."""

    finished = Signal(list)    # list[SearchResult]
    error = Signal(str)

    def __init__(
        self,
        query: str,
        n_results: int,
        active_entry,
        settings: AppSettings,
        parent=None,
    ):
        super().__init__(parent)
        self.query = query
        self.n_results = n_results
        self.active_entry = active_entry
        self.settings = settings

    def run(self):  # noqa: D401
        store: VectorStore | None = None
        try:
            store = _get_store(self.active_entry)
            if store is None or store.count() == 0:
                self.finished.emit([])
                return

            n = self.n_results
            fetch_k = max(n * 4, 20)
            qemb = embed_query(
                self.query, model_name=self.active_entry.embedding_model
            )
            vec_results = store.search(
                qemb,
                n_results=fetch_k,
                min_score=self.settings.min_score_threshold,
                include_embeddings=self.settings.mmr_enabled,
            )

            if not vec_results:
                self.finished.emit([])
                return

            # Hybrid BM25 fusion
            if self.settings.hybrid_search_enabled:
                try:
                    all_ids, all_texts, all_metas = store.get_all_documents()
                    if all_texts:
                        bm25_hits = bm25_search(
                            self.query, all_texts, all_ids, top_k=fetch_k
                        )
                        vec_scores = {
                            r.source_file + "::chunk_" + str(r.chunk_index): r.score
                            for r in vec_results
                        }
                        bm25_scores = {
                            hit_id: score for hit_id, score in bm25_hits
                        }
                        fused = hybrid_fuse_scores(
                            vec_scores,
                            bm25_scores,
                            alpha=self.settings.hybrid_search_alpha,
                        )

                        all_results_map = {
                            r.source_file + "::chunk_" + str(r.chunk_index): r
                            for r in vec_results
                        }
                        id_to_idx = {
                            doc_id: i for i, doc_id in enumerate(all_ids)
                        }
                        for hit_id, _ in bm25_hits:
                            if (
                                hit_id not in all_results_map
                                and hit_id in id_to_idx
                            ):
                                idx = id_to_idx[hit_id]
                                meta = (
                                    all_metas[idx]
                                    if idx < len(all_metas)
                                    else {}
                                )
                                all_results_map[hit_id] = SearchResult(
                                    text=all_texts[idx],
                                    source_file=meta.get("source_file", ""),
                                    chunk_index=int(
                                        meta.get("chunk_index", 0)
                                    ),
                                    score=0.0,
                                    metadata=meta,
                                )
                        for key, score in fused.items():
                            if key in all_results_map:
                                all_results_map[key].score = score
                        vec_results = sorted(
                            all_results_map.values(),
                            key=lambda r: r.score,
                            reverse=True,
                        )[:fetch_k]
                except Exception as exc:
                    logger.warning("BM25 hybrid search failed: %s", exc)

            # Backfill embeddings for BM25-only hits when MMR is needed
            if self.settings.mmr_enabled:
                missing = [r for r in vec_results if r.embedding is None]
                if missing:
                    try:
                        ids_needed = [
                            r.source_file + "::chunk_" + str(r.chunk_index)
                            for r in missing
                        ]
                        emb_map = store.get_embeddings_by_ids(ids_needed)
                        for r, doc_id in zip(missing, ids_needed):
                            if doc_id in emb_map:
                                r.embedding = emb_map[doc_id]
                    except Exception as exc:
                        logger.warning("Failed to backfill embeddings for MMR: %s", exc)

            # Cross-encoder reranking
            if self.settings.reranking_enabled:
                try:
                    texts_for_rerank = [r.text for r in vec_results]
                    scores_for_rerank = [r.score for r in vec_results]
                    reranked = rerank_cross_encoder(
                        self.query,
                        texts_for_rerank,
                        scores_for_rerank,
                        model_name=self.settings.reranker_model,
                    )
                    reordered = [vec_results[idx] for idx, _ in reranked]
                    for res, (_, new_score) in zip(reordered, reranked):
                        res.score = new_score
                    vec_results = reordered
                except Exception as exc:
                    logger.warning("Cross-encoder reranking failed: %s", exc)

            # MMR diversity
            if self.settings.mmr_enabled and len(vec_results) > n:
                try:
                    import numpy as np
                    doc_embeddings = np.array(
                        [r.embedding for r in vec_results if r.embedding is not None],
                        dtype=np.float32,
                    )
                    if len(doc_embeddings) == len(vec_results):
                        mmr_indices = mmr_diversify(
                            query_embedding=qemb,
                            doc_embeddings=doc_embeddings,
                            scores=[r.score for r in vec_results],
                            lambda_mult=self.settings.mmr_lambda,
                            top_n=n,
                        )
                        vec_results = [vec_results[i] for i in mmr_indices]
                    else:
                        logger.warning("MMR: some embeddings missing, falling back to top-N")
                        vec_results = vec_results[:n]
                except Exception as exc:
                    logger.warning("MMR diversity failed, falling back to top-N: %s", exc)
                    vec_results = vec_results[:n]
            else:
                vec_results = vec_results[:n]

            # Apply minimum score threshold after all score transformations
            min_thr = self.settings.min_score_threshold
            if min_thr > 0:
                vec_results = [r for r in vec_results if r.score >= min_thr]

            self.finished.emit(vec_results)
        except Exception as exc:
            self.error.emit(str(exc))


class FileChangeWorker(QThread):
    """Scan source folders and compare against the file manifest."""

    finished = Signal(dict)  # {new: list[str], modified: list[str], removed: list[str]}
    error = Signal(str)

    def __init__(
        self,
        source_folders: list[str],
        supported_extensions: list[str],
        db_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self.source_folders = source_folders
        self.supported_extensions = set(supported_extensions)
        self.db_path = db_path

    def run(self):  # noqa: D401
        import os
        from rag_kb.file_manifest import FileManifest

        try:
            # Discover current source files
            source_files: list[str] = []
            for folder in self.source_folders:
                folder_path = Path(folder)
                if not folder_path.exists():
                    continue
                for root, _dirs, filenames in os.walk(folder_path):
                    for fname in filenames:
                        fp = Path(root) / fname
                        if fp.suffix.lower() in self.supported_extensions:
                            source_files.append(str(fp))

            source_set = set(source_files)

            # Open manifest read-only
            manifest_db = os.path.join(self.db_path, "file_manifest.db")
            if not os.path.exists(manifest_db):
                # No manifest yet — everything is new
                self.finished.emit({
                    "new": sorted(source_files),
                    "modified": [],
                    "removed": [],
                })
                return

            manifest = FileManifest(manifest_db)
            indexed_set = manifest.all_paths()

            new_files = sorted(source_set - indexed_set)
            removed_files = sorted(indexed_set - source_set)

            # Check for modifications among files present in both sets
            modified_files: list[str] = []
            for fp in sorted(source_set & indexed_set):
                if manifest.is_changed(fp):
                    modified_files.append(fp)

            manifest.close()

            self.finished.emit({
                "new": new_files,
                "modified": modified_files,
                "removed": removed_files,
            })
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------


class StatCard(QFrame):
    """A styled card showing a large stat value and a small label."""

    def __init__(self, value: str, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("stat-card")
        self.setMinimumHeight(90)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(12, 14, 12, 14)

        self.value_label = QLabel(value)
        self.value_label.setObjectName("stat-value")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)

        self.text_label = QLabel(label)
        self.text_label.setObjectName("stat-label")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.text_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class ResultCard(QFrame):
    """A styled card for a single search result."""

    def __init__(
        self,
        index: int,
        source: str,
        score: float,
        text: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("result-card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # Header row: rank, source, score
        header = QHBoxLayout()
        num_label = QLabel(f"#{index}")
        num_label.setStyleSheet("font-weight: bold; color: #d4d8e0; font-size: 14px;")
        header.addWidget(num_label)

        source_label = QLabel(source)
        source_label.setStyleSheet("color: #788a9a; font-size: 12px;")
        source_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        header.addWidget(source_label)

        pct = int(score * 100)
        color = "#66bb6a" if pct >= 60 else "#ffa726" if pct >= 35 else "#ef5350"
        score_label = QLabel(f"{pct}%")
        score_label.setStyleSheet(
            f"font-weight: bold; color: {color}; font-size: 14px;"
        )
        header.addWidget(score_label)
        layout.addLayout(header)

        # Score bar
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(pct)
        bar.setTextVisible(False)
        bar.setFixedHeight(4)
        bar.setStyleSheet(
            f"QProgressBar {{ background-color: #313949; border: none; border-radius: 2px; }}"
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}"
        )
        layout.addWidget(bar)

        # Text content
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        text_label.setStyleSheet(
            "font-size: 13px; color: #c0c8d4; padding: 4px 0px;"
        )
        layout.addWidget(text_label)


class PlaceholderLabel(QLabel):
    """Centered placeholder message for empty states."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("color: #5a6a78; font-size: 14px; padding: 40px;")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


class DashboardPage(QWidget):
    """Overview dashboard with stat cards and active RAG info."""

    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(10)

        title = QLabel("Dashboard")
        title.setObjectName("page-title")
        layout.addWidget(title)

        subtitle = QLabel("Overview of your RAG knowledge bases")
        subtitle.setObjectName("page-subtitle")
        layout.addWidget(subtitle)

        # Stat cards row
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(14)
        self.card_rags = StatCard("0", "Knowledge Bases")
        self.card_files = StatCard("0", "Indexed Files")
        self.card_chunks = StatCard("0", "Text Chunks")
        self.card_active = StatCard("\u2014", "Active RAG")
        cards_layout.addWidget(self.card_rags)
        cards_layout.addWidget(self.card_files)
        cards_layout.addWidget(self.card_chunks)
        cards_layout.addWidget(self.card_active)
        layout.addLayout(cards_layout)

        # Active RAG details panel
        self.details_frame = QFrame()
        self.details_frame.setObjectName("details-panel")
        dl = QGridLayout(self.details_frame)
        dl.setContentsMargins(18, 16, 18, 16)
        dl.setHorizontalSpacing(20)
        dl.setVerticalSpacing(8)

        for row, label_text in enumerate(
            ["Type", "Embedding Model", "Watcher", "Last Indexed"]
        ):
            key = QLabel(label_text)
            key.setObjectName("detail-key")
            dl.addWidget(key, row, 0)

        self.detail_type = QLabel("\u2014")
        self.detail_type.setObjectName("detail-val")
        self.detail_model = QLabel("\u2014")
        self.detail_model.setObjectName("detail-val")
        self.detail_watcher = QLabel("\u2014")
        self.detail_watcher.setObjectName("detail-val")
        self.detail_last_indexed = QLabel("\u2014")
        self.detail_last_indexed.setObjectName("detail-val")

        dl.addWidget(self.detail_type, 0, 1)
        dl.addWidget(self.detail_model, 1, 1)
        dl.addWidget(self.detail_watcher, 2, 1)
        dl.addWidget(self.detail_last_indexed, 3, 1)

        dl.setColumnStretch(1, 1)
        layout.addWidget(self.details_frame)

        layout.addStretch()

        # Auto-refresh timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(10_000)

        self.refresh()

    def refresh(self) -> None:
        store: VectorStore | None = None
        try:
            settings, registry = _init()
            active = registry.get_active()
            rags = registry.list_rags()

            self.card_rags.set_value(str(len(rags)))

            if active:
                store = _get_store(active)
                stats = store.get_stats() if store else None
                self.card_files.set_value(
                    str(stats.total_files if stats else 0)
                )
                self.card_chunks.set_value(
                    str(stats.total_chunks if stats else 0)
                )
                self.card_active.set_value(active.name)

                rag_type = "Imported" if active.is_imported else "Local"
                if active.detached:
                    rag_type += " (Detached)"
                self.detail_type.setText(rag_type)
                self.detail_model.setText(active.embedding_model)

                global _watcher
                watcher_status = (
                    "Running"
                    if _watcher and _watcher.is_running
                    else "Stopped"
                )
                self.detail_watcher.setText(watcher_status)

                global _last_index_state
                if _last_index_state and _last_index_state.last_indexed:
                    self.detail_last_indexed.setText(
                        _last_index_state.last_indexed[:19].replace("T", " ")
                    )
                else:
                    self.detail_last_indexed.setText("\u2014")

                self.details_frame.setVisible(True)
            else:
                self.card_files.set_value("0")
                self.card_chunks.set_value("0")
                self.card_active.set_value("\u2014")
                self.details_frame.setVisible(False)
        except Exception as exc:
            # During reindexing the ChromaDB collection may be temporarily
            # deleted.  Avoid noisy ERROR logs for this expected situation.
            exc_name = type(exc).__name__
            if "NotFoundError" in exc_name or "does not exist" in str(exc):
                logger.debug("Dashboard refresh skipped (collection unavailable — likely reindexing)")
            else:
                logger.exception("Dashboard refresh failed")


class SearchPage(QWidget):
    """Full-text + vector search interface."""

    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main_window = main_window
        self._search_worker: SearchWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(10)

        title = QLabel("Search")
        title.setObjectName("page-title")
        layout.addWidget(title)

        subtitle = QLabel("Search your indexed knowledge base")
        subtitle.setObjectName("page-subtitle")
        layout.addWidget(subtitle)

        # Search bar
        search_bar = QHBoxLayout()
        search_bar.setSpacing(8)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("What are you looking for?")
        self.query_input.returnPressed.connect(self._do_search)
        search_bar.addWidget(self.query_input)

        self.n_results = QSpinBox()
        self.n_results.setRange(1, 50)
        self.n_results.setValue(5)
        self.n_results.setPrefix("Top ")
        self.n_results.setSuffix(" results")
        self.n_results.setFixedWidth(150)
        search_bar.addWidget(self.n_results)

        self.search_btn = QPushButton("Search")
        self.search_btn.setObjectName("primary-btn")
        self.search_btn.setFixedWidth(100)
        self.search_btn.clicked.connect(self._do_search)
        search_bar.addWidget(self.search_btn)

        layout.addLayout(search_bar)

        # Results scroll area
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_layout.setSpacing(8)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_scroll.setWidget(self.results_container)
        layout.addWidget(self.results_scroll)

        # Initial placeholder
        self._show_placeholder("Enter a query and press Search.")

    def _show_placeholder(self, text: str) -> None:
        self._clear_results()
        self.results_layout.addWidget(PlaceholderLabel(text))

    def _clear_results(self) -> None:
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child is not None:
                widget = child.widget()
                if widget is not None:
                    widget.deleteLater()

    def _do_search(self) -> None:
        query = self.query_input.text().strip()
        if not query:
            self._show_placeholder("Enter a search query.")
            return

        settings, registry = _init()
        active = registry.get_active()
        if active is None:
            self._show_placeholder(
                "No active RAG database. Create or import one first."
            )
            return

        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching\u2026")
        self._show_placeholder("Searching\u2026")

        self._search_worker = SearchWorker(
            query=query,
            n_results=self.n_results.value(),
            active_entry=active,
            settings=settings,
            parent=self,
        )
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.error.connect(self._on_search_error)
        self._search_worker.start()

    def _on_search_finished(self, results: list) -> None:
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search")
        self._clear_results()

        if not results:
            self._show_placeholder("No matching results.")
            return

        folders = _source_folders()
        for i, r in enumerate(results, 1):
            display_src = safe_display_path(r.source_file, folders)
            card = ResultCard(i, display_src, r.score, r.text)
            self.results_layout.addWidget(card)

        self.main_window.show_status(
            f"Found {len(results)} result(s)."
        )

    def _on_search_error(self, msg: str) -> None:
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search")
        self._show_placeholder(f"Search error: {msg}")
        self.main_window.show_status(f"Search failed: {msg}")

    def refresh(self) -> None:
        pass  # nothing to auto-refresh


# ---------------------------------------------------------------------------
# Create RAG Dialog
# ---------------------------------------------------------------------------


class CreateRagDialog(QDialog):
    """Modal dialog for creating a new RAG knowledge base."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Knowledge Base")
        self.setMinimumWidth(480)
        self.result_name: str = ""
        self.result_desc: str = ""
        self.result_folders: list[str] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 20, 24, 20)

        # Header
        header = QLabel("Create a New Knowledge Base")
        header.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #e8ecf0; padding-bottom: 4px;"
        )
        layout.addWidget(header)

        desc_label = QLabel(
            "Give your knowledge base a name, optional description, "
            "and add the folders containing your documents."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #9daec1; font-size: 12px; padding-bottom: 8px;")
        layout.addWidget(desc_label)

        # Form
        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. project-docs, research-papers")
        form.addRow("Name:", self.name_edit)

        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("Brief description (optional)")
        form.addRow("Description:", self.desc_edit)
        layout.addLayout(form)

        # Source folders
        folder_header = QLabel("Source Folders")
        folder_header.setObjectName("detail-section-title")
        layout.addWidget(folder_header)

        self.folder_list = QListWidget()
        self.folder_list.setObjectName("folder-list")
        self.folder_list.setMinimumHeight(80)
        self.folder_list.setMaximumHeight(140)
        layout.addWidget(self.folder_list)

        folder_btns = QHBoxLayout()
        folder_btns.setSpacing(8)
        add_btn = QPushButton("+ Add Folder\u2026")
        add_btn.setObjectName("secondary-btn")
        add_btn.clicked.connect(self._add_folder)
        folder_btns.addWidget(add_btn)

        remove_btn = QPushButton("\u2212 Remove Selected")
        remove_btn.setObjectName("secondary-btn")
        remove_btn.clicked.connect(self._remove_folder)
        folder_btns.addWidget(remove_btn)
        folder_btns.addStretch()
        layout.addLayout(folder_btns)

        layout.addSpacing(8)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondary-btn")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        create_btn = QPushButton("Create Knowledge Base")
        create_btn.setObjectName("primary-btn")
        create_btn.clicked.connect(self._on_create)
        btn_layout.addWidget(create_btn)
        layout.addLayout(btn_layout)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Source Folder", "", QFileDialog.Option.ShowDirsOnly,
        )
        if folder:
            resolved = str(Path(folder).resolve())
            existing = {
                self.folder_list.item(i).text()
                for i in range(self.folder_list.count())
            }
            if resolved not in existing:
                self.folder_list.addItem(resolved)

    def _remove_folder(self) -> None:
        row = self.folder_list.currentRow()
        if row >= 0:
            self.folder_list.takeItem(row)

    def _on_create(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Name is required.")
            return
        self.result_name = name
        self.result_desc = self.desc_edit.text().strip()
        self.result_folders = [
            self.folder_list.item(i).text()
            for i in range(self.folder_list.count())
        ]
        self.accept()


# ---------------------------------------------------------------------------
# Manage Page
# ---------------------------------------------------------------------------


class ManagePage(QWidget):
    """CRUD operations for RAG knowledge bases."""

    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main_window = main_window
        self._selected_name: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(10)

        # --- Header row: title + create button ---
        header_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title = QLabel("Manage")
        title.setObjectName("page-title")
        title_col.addWidget(title)
        subtitle = QLabel("Create, configure, and organise your knowledge bases")
        subtitle.setObjectName("page-subtitle")
        title_col.addWidget(subtitle)
        header_row.addLayout(title_col)
        header_row.addStretch()

        create_btn = QPushButton("+ New Knowledge Base")
        create_btn.setObjectName("primary-btn")
        create_btn.setFixedHeight(36)
        create_btn.clicked.connect(self._open_create_dialog)
        header_row.addWidget(create_btn, alignment=Qt.AlignmentFlag.AlignTop)

        import_btn = QPushButton("Import .rag\u2026")
        import_btn.setObjectName("secondary-btn")
        import_btn.setFixedHeight(36)
        import_btn.setToolTip("Import a knowledge base from a shared .rag file")
        import_btn.clicked.connect(self._do_import)
        header_row.addWidget(import_btn, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_row)

        # --- RAG table ---
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Status", "Description", "Files / Chunks", "Created"]
        )
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(140)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        layout.addWidget(self.table, stretch=1)

        # --- Placeholder shown when no RAG is selected ---
        self.placeholder = PlaceholderLabel(
            "Select a knowledge base above to view details and actions"
        )

        # --- Detail panel (hidden until a row is selected) ---
        self.detail_panel = QFrame()
        self.detail_panel.setObjectName("rag-detail-panel")
        dp_layout = QVBoxLayout(self.detail_panel)
        dp_layout.setContentsMargins(20, 16, 20, 16)
        dp_layout.setSpacing(10)

        # Detail header: name + badges
        detail_header = QHBoxLayout()
        self.detail_name = QLabel()
        self.detail_name.setStyleSheet(
            "font-size: 17px; font-weight: bold; color: #e8ecf0;"
        )
        detail_header.addWidget(self.detail_name)
        self.badge_active = QLabel("ACTIVE")
        self.badge_active.setObjectName("badge-active")
        detail_header.addWidget(self.badge_active)
        self.badge_type = QLabel()
        detail_header.addWidget(self.badge_type)
        detail_header.addStretch()
        dp_layout.addLayout(detail_header)

        # Description
        self.detail_desc = QLabel()
        self.detail_desc.setWordWrap(True)
        self.detail_desc.setStyleSheet("color: #9daec1; font-size: 12px;")
        dp_layout.addWidget(self.detail_desc)

        # Two-column body: properties | source folders
        body = QHBoxLayout()
        body.setSpacing(20)

        # Left column: properties
        props_col = QVBoxLayout()
        props_title = QLabel("Properties")
        props_title.setObjectName("detail-section-title")
        props_col.addWidget(props_title)

        props_grid = QGridLayout()
        props_grid.setHorizontalSpacing(16)
        props_grid.setVerticalSpacing(6)
        for row, label in enumerate(
            ["Embedding Model", "Files Indexed", "Chunks", "Created"]
        ):
            key = QLabel(label)
            key.setObjectName("detail-key")
            props_grid.addWidget(key, row, 0)
        self.prop_model = QLabel("\u2014")
        self.prop_model.setObjectName("detail-val")
        self.prop_files = QLabel("\u2014")
        self.prop_files.setObjectName("detail-val")
        self.prop_chunks = QLabel("\u2014")
        self.prop_chunks.setObjectName("detail-val")
        self.prop_created = QLabel("\u2014")
        self.prop_created.setObjectName("detail-val")
        props_grid.addWidget(self.prop_model, 0, 1)
        props_grid.addWidget(self.prop_files, 1, 1)
        props_grid.addWidget(self.prop_chunks, 2, 1)
        props_grid.addWidget(self.prop_created, 3, 1)
        props_grid.setColumnStretch(1, 1)
        props_col.addLayout(props_grid)
        props_col.addStretch()
        body.addLayout(props_col)

        # Right column: source folders
        folders_col = QVBoxLayout()
        folders_title = QLabel("Source Folders")
        folders_title.setObjectName("detail-section-title")
        folders_col.addWidget(folders_title)

        self.selected_folder_list = QListWidget()
        self.selected_folder_list.setObjectName("folder-list")
        self.selected_folder_list.setMinimumHeight(80)
        folders_col.addWidget(self.selected_folder_list)

        folder_btns = QHBoxLayout()
        folder_btns.setSpacing(8)
        self.add_folder_btn = QPushButton("+ Add\u2026")
        self.add_folder_btn.setObjectName("secondary-btn")
        self.add_folder_btn.clicked.connect(self._add_selected_folder)
        folder_btns.addWidget(self.add_folder_btn)
        self.remove_folder_btn = QPushButton("\u2212 Remove")
        self.remove_folder_btn.setObjectName("secondary-btn")
        self.remove_folder_btn.setEnabled(False)
        self.remove_folder_btn.clicked.connect(self._remove_selected_folder)
        folder_btns.addWidget(self.remove_folder_btn)
        self.save_folders_btn = QPushButton("Save Changes")
        self.save_folders_btn.setObjectName("primary-btn")
        self.save_folders_btn.setEnabled(False)
        self.save_folders_btn.clicked.connect(self._save_selected_folders)
        folder_btns.addWidget(self.save_folders_btn)
        folder_btns.addStretch()
        folders_col.addLayout(folder_btns)
        body.addLayout(folders_col)

        # Track folder selection and list changes
        self.selected_folder_list.currentRowChanged.connect(
            self._on_folder_selection_changed
        )
        self.selected_folder_list.model().rowsInserted.connect(
            self._on_folders_modified
        )
        self.selected_folder_list.model().rowsRemoved.connect(
            self._on_folders_modified
        )

        dp_layout.addLayout(body)

        # Action bar
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313949;")
        dp_layout.addWidget(sep)

        action_bar = QHBoxLayout()
        action_bar.setSpacing(10)

        self.activate_btn = QPushButton("Set as Active")
        self.activate_btn.setObjectName("primary-btn")
        self.activate_btn.setToolTip("Make this the active knowledge base for searches")
        self.activate_btn.clicked.connect(self._set_active)
        action_bar.addWidget(self.activate_btn)

        self.export_btn = QPushButton("Export .rag\u2026")
        self.export_btn.setObjectName("secondary-btn")
        self.export_btn.setToolTip("Export this knowledge base as a shareable .rag file")
        self.export_btn.clicked.connect(self._do_export)
        action_bar.addWidget(self.export_btn)

        self.detach_btn = QPushButton("Detach")
        self.detach_btn.setObjectName("secondary-btn")
        self.detach_btn.setToolTip(
            "Detach from source files \u2014 makes the knowledge base read-only "
            "so you can safely delete source files"
        )
        self.detach_btn.clicked.connect(self._detach_rag)
        action_bar.addWidget(self.detach_btn)

        self.attach_btn = QPushButton("Re-attach")
        self.attach_btn.setObjectName("secondary-btn")
        self.attach_btn.setToolTip("Re-attach to source files and re-enable indexing")
        self.attach_btn.clicked.connect(self._attach_rag)
        action_bar.addWidget(self.attach_btn)

        action_bar.addStretch()

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setObjectName("danger-btn")
        self.delete_btn.setToolTip("Permanently delete this knowledge base and all its data")
        self.delete_btn.clicked.connect(self._delete_rag)
        action_bar.addWidget(self.delete_btn)

        dp_layout.addLayout(action_bar)

        # Stack: placeholder vs detail panel
        self.detail_stack = QStackedWidget()
        self.detail_stack.addWidget(self.placeholder)    # index 0
        self.detail_stack.addWidget(self.detail_panel)    # index 1
        self.detail_stack.setCurrentIndex(0)
        layout.addWidget(self.detail_stack)

        self.refresh()

    # ------------------------------------------------------------------
    # Table selection
    # ------------------------------------------------------------------

    def _on_table_selection(self) -> None:
        """Show the detail panel for the selected RAG."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self._selected_name = None
            self.detail_stack.setCurrentIndex(0)
            return
        name_item = self.table.item(rows[0].row(), 0)
        if not name_item:
            return
        name = name_item.data(Qt.ItemDataRole.UserRole)
        if not name:
            name = name_item.text().strip()
        self._selected_name = name
        self._populate_detail(name)
        self.detail_stack.setCurrentIndex(1)

    def _populate_detail(self, name: str) -> None:
        """Fill the detail panel with data for *name*."""
        try:
            _, registry = _init()
            entry = registry.get_rag(name)
            active_name = registry.get_active_name()
        except KeyError:
            return

        is_active = name == active_name
        self.detail_name.setText(entry.name)
        self.badge_active.setVisible(is_active)

        # Type badge
        if entry.detached:
            self.badge_type.setText("DETACHED")
            self.badge_type.setObjectName("badge-detached")
        elif entry.is_imported:
            self.badge_type.setText("IMPORTED")
            self.badge_type.setObjectName("badge-imported")
        else:
            self.badge_type.setText("LOCAL")
            self.badge_type.setObjectName("badge-local")
        # Force style refresh after objectName change
        self.badge_type.style().unpolish(self.badge_type)
        self.badge_type.style().polish(self.badge_type)

        self.detail_desc.setText(entry.description or "No description.")
        self.detail_desc.setVisible(True)

        # Properties
        self.prop_model.setText(entry.embedding_model)
        self.prop_files.setText(str(entry.file_count))
        self.prop_chunks.setText(str(entry.chunk_count))
        self.prop_created.setText(
            entry.created_at[:10] if entry.created_at else "\u2014"
        )

        # Source folders
        self.selected_folder_list.clear()
        for folder in entry.source_folders:
            self.selected_folder_list.addItem(folder)

        # Enable / disable contextual buttons
        self.activate_btn.setEnabled(not is_active)
        self.activate_btn.setText(
            "\u2713 Active" if is_active else "Set as Active"
        )
        self.detach_btn.setVisible(not entry.detached)
        self.attach_btn.setVisible(entry.detached)

        # Disable folder editing for detached RAGs; Save starts disabled
        # until the user actually changes the folder list.
        folders_editable = not entry.detached
        self.add_folder_btn.setEnabled(folders_editable)
        self.remove_folder_btn.setEnabled(
            folders_editable
            and self.selected_folder_list.currentRow() >= 0
        )
        # Save is only enabled when the folder list differs from the
        # persisted state — reset on every populate.
        self.save_folders_btn.setEnabled(False)
        # Store the original folder list so we can detect changes.
        self._original_folders: list[str] = list(entry.source_folders)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def _open_create_dialog(self) -> None:
        dlg = CreateRagDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            try:
                settings, registry = _init()
                registry.create_rag(
                    dlg.result_name, dlg.result_desc, dlg.result_folders
                )
                self.main_window.show_status(
                    f"Created '{dlg.result_name}'"
                )
                self.refresh()
                self.main_window.on_data_changed()
            except Exception as e:
                QMessageBox.critical(self, "Create Error", str(e))

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def _do_export(self) -> None:
        name = self._selected()
        if not name:
            return
        path, _ = QFileDialog.getSaveFileName(
            self.window(), "Export RAG", f"{name}.rag", "RAG files (*.rag)"
        )
        if not path:
            return
        try:
            _, registry = _init()
            result = export_rag(registry, name, path)
            QMessageBox.information(
                self, "Export", f"Exported to:\n{result}"
            )
            self.main_window.show_status(f"Exported '{name}'")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _do_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window(), "Import RAG", "", "RAG files (*.rag)"
        )
        if not path:
            return
        try:
            _, registry = _init()
            imported = import_rag(registry, path)
            QMessageBox.information(
                self, "Import", f"Imported as '{imported}'"
            )
            self.main_window.show_status(f"Imported '{imported}'")
            self.refresh()
            self.main_window.on_data_changed()
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _selected(self) -> str | None:
        return self._selected_name

    def _set_active(self) -> None:
        name = self._selected()
        if not name:
            return
        try:
            settings, registry = _init()
            registry.set_active(name)
            self._sync_watcher_for_active(registry, settings)
            self.main_window.show_status(f"Active RAG set to '{name}'")
            self.refresh()
            self.main_window.on_data_changed()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _delete_rag(self) -> None:
        name = self._selected()
        if not name:
            return
        reply = QMessageBox.question(
            self,
            "Delete Knowledge Base",
            f"Are you sure you want to permanently delete '{name}'?\n\n"
            "This will remove all indexed data and cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            # Stop the dashboard auto-refresh timer so it doesn't re-open
            # a VectorStore while we're trying to release the file lock.
            dashboard = self.main_window.dashboard_page
            dashboard._timer.stop()

            global _watcher
            if _watcher:
                _watcher.stop()
                _watcher = None

            # Force-close the cached VectorStore (and its ChromaDB client)
            # so the HNSW memory-mapped .bin files are released on Windows.
            _, registry = _init()
            db_path = None
            try:
                entry = registry.get_rag(name)
                db_path = entry.db_path
                _close_store(db_path, force=True)
            except Exception:
                pass

            # Also sweep for orphaned ChromaDB clients just in case
            if db_path:
                from rag_kb.config import _close_chroma_for_path
                try:
                    _close_chroma_for_path(db_path)
                except Exception:
                    pass

            import gc
            gc.collect()

            # delete_rag() removes the registry entry first (so the RAG
            # disappears immediately) then attempts file cleanup.  If the
            # files are still locked it schedules deferred cleanup.
            fully_cleaned = registry.delete_rag(name)

            # Auto-select the active RAG (or first remaining) after deletion
            new_active = registry.get_active_name()
            remaining = registry.list_rags()
            if new_active:
                self._selected_name = new_active
            elif remaining:
                self._selected_name = remaining[0].name
            else:
                self._selected_name = None

            if fully_cleaned:
                self.main_window.show_status(f"Deleted '{name}'")
            else:
                self.main_window.show_status(
                    f"Deleted '{name}' \u2014 data files are locked by another "
                    f"process (e.g. MCP server) and will be cleaned up on "
                    f"next launch.",
                    10_000,
                )
            self.refresh()
            self.main_window.on_data_changed()
        except Exception as e:
            QMessageBox.critical(self, "Delete Error", str(e))
        finally:
            # Always restart the dashboard timer
            dashboard = self.main_window.dashboard_page
            dashboard._timer.start(10_000)

    def _detach_rag(self) -> None:
        name = self._selected()
        if not name:
            return
        try:
            _, registry = _init()
            entry = registry.get_rag(name)
            entry.detached = True
            registry.update_rag(entry)

            global _watcher
            if _watcher:
                _watcher.stop()
                _watcher = None

            self.main_window.show_status(
                f"Detached '{name}' \u2014 source files can be safely deleted."
            )
            self.refresh()
            self.main_window.on_data_changed()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _attach_rag(self) -> None:
        name = self._selected()
        if not name:
            return
        try:
            settings, registry = _init()
            entry = registry.get_rag(name)
            entry.detached = False
            registry.update_rag(entry)
            self._sync_watcher_for_active(registry, settings)
            self.main_window.show_status(
                f"Re-attached '{name}' \u2014 indexing re-enabled."
            )
            self.refresh()
            self.main_window.on_data_changed()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ------------------------------------------------------------------
    # Source folder management
    # ------------------------------------------------------------------

    def _on_folder_selection_changed(self, row: int) -> None:
        """Enable 'Remove' button only when a folder is selected."""
        name = self._selected()
        if not name:
            return
        try:
            _, registry = _init()
            entry = registry.get_rag(name)
            editable = not entry.detached
        except KeyError:
            editable = False
        self.remove_folder_btn.setEnabled(editable and row >= 0)

    def _on_folders_modified(self) -> None:
        """Enable 'Save Changes' when the folder list differs from the persisted state."""
        current = [
            self.selected_folder_list.item(i).text()
            for i in range(self.selected_folder_list.count())
        ]
        orig = getattr(self, "_original_folders", None) or []
        self.save_folders_btn.setEnabled(current != orig)

    def _add_selected_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self.window(), "Select Source Folder",
            "", QFileDialog.Option.ShowDirsOnly,
        )
        if not folder:
            return
        resolved = str(Path(folder).resolve())
        existing = {
            self.selected_folder_list.item(i).text()
            for i in range(self.selected_folder_list.count())
        }
        if resolved not in existing:
            self.selected_folder_list.addItem(resolved)

    def _remove_selected_folder(self) -> None:
        row = self.selected_folder_list.currentRow()
        if row >= 0:
            self.selected_folder_list.takeItem(row)

    def _save_selected_folders(self) -> None:
        name = self._selected()
        if not name:
            QMessageBox.warning(self, "Folders", "Select a RAG first.")
            return

        folders = [
            self.selected_folder_list.item(i).text()
            for i in range(self.selected_folder_list.count())
        ]
        try:
            settings, registry = _init()
            entry = registry.get_rag(name)
            entry.source_folders = folders
            registry.update_rag(entry)
            self._sync_watcher_for_active(registry, settings)
            self.main_window.show_status(
                f"Updated source folders for '{name}'."
            )
            self.refresh()
            self.main_window.on_data_changed()
        except Exception as exc:
            QMessageBox.critical(self, "Folders Error", str(exc))

    # ------------------------------------------------------------------
    # Watcher sync
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_watcher_for_active(registry, settings) -> None:
        global _watcher
        if _watcher:
            _watcher.stop()
            _watcher = None

        active = registry.get_active()
        if (
            active
            and not active.detached
            and active.source_folders
        ):
            _watcher = FolderWatcher(active, registry, settings)
            _watcher.start()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        try:
            _, registry = _init()
            rags = registry.list_rags()
            active = registry.get_active_name()

            prev_selected = self._selected_name

            # Update table
            self.table.blockSignals(True)
            self.table.setRowCount(len(rags))
            select_row: int | None = None
            for row, r in enumerate(rags):
                is_act = r.name == active
                # Name
                name_item = QTableWidgetItem(r.name)
                name_item.setData(Qt.ItemDataRole.UserRole, r.name)
                if is_act:
                    font = name_item.font()
                    font.setBold(True)
                    name_item.setFont(font)
                self.table.setItem(row, 0, name_item)

                # Status
                if is_act:
                    status_text = "\u25cf Active"
                elif r.detached:
                    status_text = "Detached"
                elif r.is_imported:
                    status_text = "Imported"
                else:
                    status_text = "Local"
                status_item = QTableWidgetItem(status_text)
                if is_act:
                    status_item.setForeground(QColor("#a5d6a7"))
                    font = status_item.font()
                    font.setBold(True)
                    status_item.setFont(font)
                elif r.detached:
                    status_item.setForeground(QColor("#ffcc80"))
                self.table.setItem(row, 1, status_item)

                # Description
                self.table.setItem(row, 2, QTableWidgetItem(r.description))

                # Files / Chunks
                fc_text = f"{r.file_count} / {r.chunk_count}"
                self.table.setItem(row, 3, QTableWidgetItem(fc_text))

                # Created
                created = r.created_at[:10] if r.created_at else ""
                self.table.setItem(row, 4, QTableWidgetItem(created))

                if r.name == prev_selected:
                    select_row = row

            self.table.blockSignals(False)

            # Restore selection
            if select_row is not None:
                self.table.selectRow(select_row)
                self._populate_detail(prev_selected)  # type: ignore[arg-type]
                self.detail_stack.setCurrentIndex(1)
            else:
                # No matching row (RAG was deleted or list is empty)
                self._selected_name = None
                self.detail_stack.setCurrentIndex(0)

        except Exception:
            logger.exception("Manage page refresh failed")


class IndexingPage(QWidget):
    """Index documents and view indexed files."""

    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main_window = main_window
        self._index_worker: IndexWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(10)

        title = QLabel("Indexing")
        title.setObjectName("page-title")
        layout.addWidget(title)

        subtitle = QLabel("Index documents into the active knowledge base")
        subtitle.setObjectName("page-subtitle")
        layout.addWidget(subtitle)

        # --- Active RAG info panel ---
        rag_info_frame = QFrame()
        rag_info_frame.setObjectName("details-panel")
        rag_info_layout = QVBoxLayout(rag_info_frame)
        rag_info_layout.setContentsMargins(16, 10, 16, 10)
        rag_info_layout.setSpacing(4)

        rag_info_header = QHBoxLayout()
        rag_info_title = QLabel("Active Knowledge Base")
        rag_info_title.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #78909c;"
        )
        rag_info_header.addWidget(rag_info_title)
        rag_info_header.addStretch()
        rag_info_layout.addLayout(rag_info_header)

        self.active_rag_name = QLabel("\u2014")
        self.active_rag_name.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #e8ecf0;"
        )
        rag_info_layout.addWidget(self.active_rag_name)

        self.active_rag_folders = QLabel("")
        self.active_rag_folders.setStyleSheet(
            "font-size: 12px; color: #9daec1;"
        )
        self.active_rag_folders.setWordWrap(True)
        rag_info_layout.addWidget(self.active_rag_folders)

        self.no_rag_warning = QLabel("")
        self.no_rag_warning.setStyleSheet(
            "font-size: 12px; color: #ef5350; font-weight: bold;"
        )
        self.no_rag_warning.setVisible(False)
        rag_info_layout.addWidget(self.no_rag_warning)

        layout.addWidget(rag_info_frame)

        # Buttons
        btn_layout = QHBoxLayout()
        self.incr_btn = QPushButton("Incremental Index")
        self.incr_btn.setObjectName("primary-btn")
        self.incr_btn.clicked.connect(lambda: self._do_index(False))
        btn_layout.addWidget(self.incr_btn)

        self.full_btn = QPushButton("Full Reindex")
        self.full_btn.setObjectName("secondary-btn")
        self.full_btn.clicked.connect(lambda: self._do_index(True))
        btn_layout.addWidget(self.full_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # --- Source file changes section ---
        changes_frame = QFrame()
        changes_frame.setObjectName("details-panel")
        changes_layout = QVBoxLayout(changes_frame)
        changes_layout.setContentsMargins(16, 12, 16, 12)
        changes_layout.setSpacing(6)

        changes_header = QHBoxLayout()
        changes_title = QLabel("Source File Changes")
        changes_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #4fc3f7;"
        )
        changes_header.addWidget(changes_title)
        changes_header.addStretch()

        self.scan_btn = QPushButton("Check for Changes")
        self.scan_btn.setObjectName("secondary-btn")
        self.scan_btn.setStyleSheet(
            "padding: 4px 14px; font-size: 12px;"
        )
        self.scan_btn.clicked.connect(self._scan_file_changes)
        changes_header.addWidget(self.scan_btn)
        changes_layout.addLayout(changes_header)

        # Summary row: badges + status
        summary_row = QHBoxLayout()
        summary_row.setSpacing(8)

        self.change_new_badge = QLabel()
        self.change_new_badge.setObjectName("change-badge-new")
        self.change_new_badge.setVisible(False)
        summary_row.addWidget(self.change_new_badge)

        self.change_modified_badge = QLabel()
        self.change_modified_badge.setObjectName("change-badge-modified")
        self.change_modified_badge.setVisible(False)
        summary_row.addWidget(self.change_modified_badge)

        self.change_removed_badge = QLabel()
        self.change_removed_badge.setObjectName("change-badge-removed")
        self.change_removed_badge.setVisible(False)
        summary_row.addWidget(self.change_removed_badge)

        self.change_ok_badge = QLabel("\u2713 No pending changes")
        self.change_ok_badge.setObjectName("change-badge-ok")
        self.change_ok_badge.setVisible(False)
        summary_row.addWidget(self.change_ok_badge)

        self.scan_status_label = QLabel(
            "Click \u201cCheck for Changes\u201d to compare source folders with indexed files."
        )
        self.scan_status_label.setStyleSheet("color: #78909c; font-size: 11px;")
        self.scan_status_label.setWordWrap(True)
        summary_row.addWidget(self.scan_status_label, stretch=1)

        summary_row.addStretch()
        changes_layout.addLayout(summary_row)

        # File list — shows new / modified / removed files
        self.change_file_list = QListWidget()
        self.change_file_list.setObjectName("change-file-list")
        self.change_file_list.setMaximumHeight(150)
        self.change_file_list.setVisible(False)
        changes_layout.addWidget(self.change_file_list)

        layout.addWidget(changes_frame)

        self._file_change_worker: FileChangeWorker | None = None

        # Progress section
        progress_frame = QFrame()
        progress_frame.setObjectName("details-panel")
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(16, 12, 16, 12)
        progress_layout.setSpacing(6)

        # Phase + counter row
        phase_row = QHBoxLayout()
        self.phase_label = QLabel("")
        self.phase_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #4fc3f7; "
            "background-color: #1a3a4a; border-radius: 4px; padding: 2px 10px;"
        )
        self.phase_label.setVisible(False)
        phase_row.addWidget(self.phase_label)

        self.counter_label = QLabel("")
        self.counter_label.setStyleSheet("font-size: 12px; color: #9daec1;")
        phase_row.addStretch()
        phase_row.addWidget(self.counter_label)

        self.pct_label = QLabel("")
        self.pct_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #67d3ff;"
        )
        phase_row.addWidget(self.pct_label)
        progress_layout.addLayout(phase_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #313949; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background-color: #2f79c6; border-radius: 3px; }"
        )
        progress_layout.addWidget(self.progress_bar)

        # Current file
        self.file_label = QLabel("")
        self.file_label.setStyleSheet("font-size: 12px; color: #788a9a;")
        self.file_label.setWordWrap(True)
        progress_layout.addWidget(self.file_label)

        # Status / result message
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 13px; color: #d4d8e0;")
        progress_layout.addWidget(self.status_label)

        self.progress_frame = progress_frame
        layout.addWidget(progress_frame)

        # Files table
        files_label = QLabel("Indexed Files")
        files_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #4fc3f7; "
            "padding-top: 10px;"
        )
        layout.addWidget(files_label)

        self.files_table = QTableWidget(0, 2)
        self.files_table.setHorizontalHeaderLabels(["File", "Chunks"])
        self.files_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.files_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.files_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.files_table.verticalHeader().setVisible(False)
        self.files_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        layout.addWidget(self.files_table)

        self.refresh()

    def _do_index(self, full: bool) -> None:
        settings, registry = _init()
        active = registry.get_active()
        if active is None:
            QMessageBox.warning(
                self, "Indexing", "No active RAG. Create or select one first."
            )
            return
        if active.detached:
            QMessageBox.warning(
                self,
                "Indexing",
                "RAG is detached (read-only). Re-attach it first.",
            )
            return
        if not active.source_folders:
            QMessageBox.warning(
                self,
                "Indexing",
                f"'{active.name}' has no source folders configured.\n\n"
                "Go to Manage \u2192 select this RAG \u2192 add Source Folders "
                "\u2192 Save Changes, then try again.",
            )
            return

        logger.info(
            "Starting %s index for '%s' with %d source folder(s): %s",
            "full" if full else "incremental",
            active.name,
            len(active.source_folders),
            active.source_folders,
        )

        self.incr_btn.setEnabled(False)
        self.full_btn.setEnabled(False)
        self._reset_progress()
        self.status_label.setText("")

        # Evict the cached VectorStore *before* the worker starts so that
        # dashboard refreshes during indexing don't hit a stale ChromaDB
        # collection reference (the Indexer creates its own VectorStore
        # which may delete & recreate the collection on full reindex).
        import os as _os
        _norm = _os.path.normcase(_os.path.normpath(active.db_path))
        _old_store = _store_cache.pop(_norm, None)
        if _old_store is not None:
            try:
                _old_store._client = None
            except Exception:
                pass

        self._index_worker = IndexWorker(
            entry=active,
            registry=registry,
            settings=settings,
            full=full,
            parent=self,
        )
        self._index_worker.progress.connect(self._on_index_progress)
        self._index_worker.detail_progress.connect(self._on_detail_progress)
        self._index_worker.finished.connect(self._on_index_finished)
        self._index_worker.error.connect(self._on_index_error)
        self._index_worker.start()

    def _reset_progress(self) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.phase_label.setVisible(False)
        self.counter_label.setText("")
        self.pct_label.setText("")
        self.file_label.setText("")

    _PHASE_LABELS = {
        "scanning": "Scanning",
        "parsing": "Parsing",
        "embedding": "Embedding",
        "done": "Done",
    }

    def _on_detail_progress(self, info: dict) -> None:
        phase = info.get("phase", "")
        pct = info.get("pct", 0)
        processed = info.get("processed", 0)
        total = info.get("total", 0)
        current_file = info.get("file", "")

        # Phase badge
        label = self._PHASE_LABELS.get(phase, phase.capitalize()) or ""
        self.phase_label.setText(label)
        self.phase_label.setVisible(True)

        # Special styling per phase
        if phase == "embedding":
            self.phase_label.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #ffcc80; "
                "background-color: #3a3020; border-radius: 4px; padding: 2px 10px;"
            )
        elif phase == "scanning":
            self.phase_label.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #a5d6a7; "
                "background-color: #1a3a2a; border-radius: 4px; padding: 2px 10px;"
            )
        else:
            self.phase_label.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #4fc3f7; "
                "background-color: #1a3a4a; border-radius: 4px; padding: 2px 10px;"
            )

        # Counter
        if total > 0:
            self.counter_label.setText(f"{processed} / {total} files")
        else:
            self.counter_label.setText("")

        # Percentage and progress bar
        if phase == "scanning":
            # Scanning is indeterminate
            self.progress_bar.setRange(0, 0)
            self.pct_label.setText("")
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(pct)
            self.pct_label.setText(f"{pct}%")

        # Current file
        if current_file:
            folders = self._index_worker.entry.source_folders if self._index_worker else []
            self.file_label.setText(safe_display_path(current_file, folders))
        else:
            self.file_label.setText("")

    def _on_index_progress(self, msg: str) -> None:
        # Keep for backward compat — detail_progress handles the UI now
        pass

    def _on_index_finished(self, state: IndexingState) -> None:
        global _last_index_state
        _last_index_state = state

        # The Indexer created its own VectorStore and (on full reindex)
        # dropped & recreated the ChromaDB collection.  The cached store
        # still holds a reference to the *old* collection UUID, so evict
        # it.  The next _get_store() call will create a fresh handle.
        _, registry = _init()
        active = registry.get_active()
        if active:
            import os
            norm = os.path.normcase(os.path.normpath(active.db_path))
            old = _store_cache.pop(norm, None)
            if old is not None:
                try:
                    old._client = None  # drop ref; don't force-close
                except Exception:
                    pass

        self.incr_btn.setEnabled(True)
        self.full_btn.setEnabled(True)

        # Final progress state
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.pct_label.setText("100%")
        self.phase_label.setText("Done")
        self.phase_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #a5d6a7; "
            "background-color: #1a3a2a; border-radius: 4px; padding: 2px 10px;"
        )
        self.file_label.setText("")
        self.counter_label.setText(
            f"{state.processed_files} / {state.total_files} files"
            if state.total_files else ""
        )

        msg = (
            f"Indexed {state.processed_files} files \u2192 "
            f"{state.total_chunks} chunks in {state.duration_seconds}s"
        )
        if state.errors:
            msg += f"  \u2022  {len(state.errors)} error(s)"
        self.status_label.setText(msg)
        self.status_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #a5d6a7;"
        )
        self.main_window.show_status(msg)
        self.refresh()
        self.main_window.on_data_changed()

    def _on_index_error(self, msg: str) -> None:
        self.incr_btn.setEnabled(True)
        self.full_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.phase_label.setText("Error")
        self.phase_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #ef5350; "
            "background-color: #3a1a1a; border-radius: 4px; padding: 2px 10px;"
        )
        self.file_label.setText("")
        self.status_label.setText(f"Error: {msg}")
        self.status_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #ef5350;"
        )
        QMessageBox.critical(self, "Indexing Error", msg)

    # ------------------------------------------------------------------
    # File change scanning
    # ------------------------------------------------------------------

    def _scan_file_changes(self) -> None:
        """Scan source folders and compare against indexed manifest."""
        try:
            settings, registry = _init()
            active = registry.get_active()
        except Exception:
            return

        if active is None:
            self.scan_status_label.setText("No active RAG selected.")
            return
        if not active.source_folders:
            self.scan_status_label.setText("No source folders configured.")
            return

        # Disable button while scanning
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning\u2026")
        self.scan_status_label.setText("Scanning source folders\u2026")
        self._reset_change_badges()

        self._file_change_worker = FileChangeWorker(
            source_folders=active.source_folders,
            supported_extensions=settings.supported_extensions,
            db_path=active.db_path,
            parent=self,
        )
        self._file_change_worker.finished.connect(self._on_scan_finished)
        self._file_change_worker.error.connect(self._on_scan_error)
        self._file_change_worker.start()

    def _on_scan_finished(self, result: dict) -> None:
        """Handle completed file change scan."""
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Check for Changes")

        new_files = result.get("new", [])
        modified_files = result.get("modified", [])
        removed_files = result.get("removed", [])

        has_changes = bool(new_files or modified_files or removed_files)

        if has_changes:
            if new_files:
                self.change_new_badge.setText(f"+ {len(new_files)} new")
                self.change_new_badge.setVisible(True)
            if modified_files:
                self.change_modified_badge.setText(
                    f"\u25cb {len(modified_files)} modified"
                )
                self.change_modified_badge.setVisible(True)
            if removed_files:
                self.change_removed_badge.setText(
                    f"\u2212 {len(removed_files)} removed"
                )
                self.change_removed_badge.setVisible(True)

            self.change_ok_badge.setVisible(False)
            self.scan_status_label.setText("")

            # Populate file list grouped by change type
            self.change_file_list.clear()
            active_folders: list[str] = []
            try:
                _, registry = _init()
                active = registry.get_active()
                if active:
                    active_folders = active.source_folders
            except Exception:
                pass

            for fp in new_files:
                display = safe_display_path(fp, active_folders)
                item = QListWidgetItem(f"  + {display}")
                item.setToolTip(fp)
                item.setForeground(QColor("#66bb6a"))
                self.change_file_list.addItem(item)
            for fp in modified_files:
                display = safe_display_path(fp, active_folders)
                item = QListWidgetItem(f"  \u25cb {display}")
                item.setToolTip(fp)
                item.setForeground(QColor("#ffa726"))
                self.change_file_list.addItem(item)
            for fp in removed_files:
                display = safe_display_path(fp, active_folders)
                item = QListWidgetItem(f"  \u2212 {display}")
                item.setToolTip(fp)
                item.setForeground(QColor("#ef5350"))
                self.change_file_list.addItem(item)
            self.change_file_list.setVisible(True)
        else:
            self.change_ok_badge.setVisible(True)
            self.scan_status_label.setText("")
            self.change_file_list.setVisible(False)

    def _on_scan_error(self, msg: str) -> None:
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Check for Changes")
        self.scan_status_label.setText(f"Scan error: {msg}")

    def _reset_change_badges(self) -> None:
        """Hide all change badges and file list."""
        self.change_new_badge.setVisible(False)
        self.change_modified_badge.setVisible(False)
        self.change_removed_badge.setVisible(False)
        self.change_ok_badge.setVisible(False)
        self.change_file_list.setVisible(False)
        self.change_file_list.clear()

    def refresh(self) -> None:
        store: VectorStore | None = None
        try:
            _, registry = _init()
            active = registry.get_active()

            # Update active RAG info panel
            if active is None:
                self.active_rag_name.setText("\u2014 No active knowledge base")
                self.active_rag_folders.setText("")
                self.no_rag_warning.setText(
                    "Create or select a knowledge base on the Manage page."
                )
                self.no_rag_warning.setVisible(True)
                self.incr_btn.setEnabled(False)
                self.full_btn.setEnabled(False)
                self.scan_btn.setEnabled(False)
                self.files_table.setRowCount(0)
                self._reset_change_badges()
                return

            self.active_rag_name.setText(active.name)
            self.no_rag_warning.setVisible(False)
            self.incr_btn.setEnabled(not active.detached)
            self.full_btn.setEnabled(not active.detached)
            self.scan_btn.setEnabled(
                not active.detached and bool(active.source_folders)
            )

            if active.source_folders:
                folder_lines = []
                for f in active.source_folders:
                    folder_lines.append(f"  \u2022 {f}")
                self.active_rag_folders.setText(
                    f"Source folders ({len(active.source_folders)}):\n"
                    + "\n".join(folder_lines)
                )
            else:
                self.active_rag_folders.setText(
                    "No source folders configured \u2014 "
                    "add them on the Manage page."
                )
                self.active_rag_folders.setStyleSheet(
                    "font-size: 12px; color: #ffcc80;"
                )

            if active.detached:
                self.no_rag_warning.setText(
                    "This RAG is detached (read-only). "
                    "Re-attach it on the Manage page to enable indexing."
                )
                self.no_rag_warning.setVisible(True)

            # Reset change badges when switching RAGs
            self._reset_change_badges()
            self.scan_status_label.setText(
                "Click \u201cCheck for Changes\u201d to compare "
                "source folders with indexed files."
            )

            store = _get_store(active)
            if store is None:
                self.files_table.setRowCount(0)
                return

            stats = store.get_stats()
            folders = active.source_folders
            files = stats.files if stats else []

            self.files_table.setRowCount(len(files))
            for row, src in enumerate(files):
                chunks = store.get_by_source(src)
                self.files_table.setItem(
                    row, 0,
                    QTableWidgetItem(safe_display_path(src, folders)),
                )
                self.files_table.setItem(
                    row, 1, QTableWidgetItem(str(len(chunks)))
                )
        except Exception:
            logger.exception("Indexing page refresh failed")


class ConfigPage(QWidget):
    """Interactive configuration editor."""

    def __init__(self, main_window: "MainWindow"):
        super().__init__()
        self.main_window = main_window
        self._loading = False  # guard against feedback loops

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(10)

        title = QLabel("Configuration")
        title.setObjectName("page-title")
        outer.addWidget(title)

        self.path_label = QLabel()
        self.path_label.setObjectName("page-subtitle")
        outer.addWidget(self.path_label)

        # Scrollable area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 12, 4)

        # === Embedding & Chunking ===
        self._add_section(layout, "Embedding & Chunking")

        self.embedding_model = QComboBox()
        self.embedding_model.setEditable(False)
        self.embedding_model.addItems(EMBEDDING_MODEL_OPTIONS)
        self._add_field(layout, "Embedding model", self.embedding_model,
                "Select sentence-transformers model")

        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(64, 8192)
        self.chunk_size.setSingleStep(64)
        self.chunk_size.setSuffix(" tokens")
        self._add_field(layout, "Chunk size", self.chunk_size,
                        "Max tokens per text chunk")

        self.chunk_overlap = QSpinBox()
        self.chunk_overlap.setRange(0, 2048)
        self.chunk_overlap.setSingleStep(16)
        self.chunk_overlap.setSuffix(" tokens")
        self._add_field(layout, "Chunk overlap", self.chunk_overlap,
                        "Overlap between consecutive chunks")

        # === Search Quality ===
        self._add_section(layout, "Search Quality")

        self.reranking_enabled = QCheckBox("Enable cross-encoder reranking")
        layout.addWidget(self.reranking_enabled)

        self.reranker_model = QComboBox()
        self.reranker_model.setEditable(False)
        self.reranker_model.addItems(RERANKER_MODEL_OPTIONS)
        self._add_field(layout, "Reranker model", self.reranker_model,
                "Select cross-encoder model for reranking")

        self.hybrid_search_enabled = QCheckBox("Enable hybrid search (vector + BM25)")
        layout.addWidget(self.hybrid_search_enabled)

        # Alpha slider + spin
        alpha_row = QHBoxLayout()
        alpha_row.setSpacing(10)
        alpha_lbl = QLabel("Hybrid alpha:")
        alpha_lbl.setMinimumWidth(130)
        alpha_row.addWidget(alpha_lbl)
        self.hybrid_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.hybrid_alpha_slider.setRange(0, 100)
        self.hybrid_alpha_slider.setTickInterval(10)
        alpha_row.addWidget(self.hybrid_alpha_slider)
        self.hybrid_alpha_spin = QDoubleSpinBox()
        self.hybrid_alpha_spin.setRange(0.0, 1.0)
        self.hybrid_alpha_spin.setSingleStep(0.05)
        self.hybrid_alpha_spin.setDecimals(2)
        self.hybrid_alpha_spin.setFixedWidth(80)
        alpha_row.addWidget(self.hybrid_alpha_spin)
        layout.addLayout(alpha_row)
        hint = QLabel("0 = all BM25, 1 = all vector")
        hint.setObjectName("config-hint")
        layout.addWidget(hint)

        # Sync slider <-> spin
        self.hybrid_alpha_slider.valueChanged.connect(
            lambda v: self.hybrid_alpha_spin.setValue(v / 100.0)
            if not self._loading else None
        )
        self.hybrid_alpha_spin.valueChanged.connect(
            lambda v: self.hybrid_alpha_slider.setValue(int(v * 100))
            if not self._loading else None
        )

        # Min score
        self.min_score_threshold = QDoubleSpinBox()
        self.min_score_threshold.setRange(0.0, 1.0)
        self.min_score_threshold.setSingleStep(0.05)
        self.min_score_threshold.setDecimals(2)
        self._add_field(layout, "Minimum score threshold",
                        self.min_score_threshold,
                        "Results below this score are discarded")

        self.mmr_enabled = QCheckBox("Enable MMR diversity filtering")
        layout.addWidget(self.mmr_enabled)

        # MMR lambda slider + spin
        mmr_row = QHBoxLayout()
        mmr_row.setSpacing(10)
        mmr_lbl = QLabel("MMR lambda:")
        mmr_lbl.setMinimumWidth(130)
        mmr_row.addWidget(mmr_lbl)
        self.mmr_lambda_slider = QSlider(Qt.Orientation.Horizontal)
        self.mmr_lambda_slider.setRange(0, 100)
        self.mmr_lambda_slider.setTickInterval(10)
        mmr_row.addWidget(self.mmr_lambda_slider)
        self.mmr_lambda_spin = QDoubleSpinBox()
        self.mmr_lambda_spin.setRange(0.0, 1.0)
        self.mmr_lambda_spin.setSingleStep(0.05)
        self.mmr_lambda_spin.setDecimals(2)
        self.mmr_lambda_spin.setFixedWidth(80)
        mmr_row.addWidget(self.mmr_lambda_spin)
        layout.addLayout(mmr_row)
        mmr_hint = QLabel("0 = max diversity, 1 = max relevance")
        mmr_hint.setObjectName("config-hint")
        layout.addWidget(mmr_hint)

        # Sync slider <-> spin for MMR lambda
        self.mmr_lambda_slider.valueChanged.connect(
            lambda v: self.mmr_lambda_spin.setValue(v / 100.0)
            if not self._loading else None
        )
        self.mmr_lambda_spin.valueChanged.connect(
            lambda v: self.mmr_lambda_slider.setValue(int(v * 100))
            if not self._loading else None
        )

        # === Indexing Performance ===
        self._add_section(layout, "Indexing Performance")

        self.indexing_workers = QSpinBox()
        self.indexing_workers.setRange(1, 32)
        self._add_field(layout, "Parallel workers", self.indexing_workers,
                        "Number of parallel file-parsing workers")

        self.embedding_batch_size = QSpinBox()
        self.embedding_batch_size.setRange(1, 4096)
        self.embedding_batch_size.setSingleStep(32)
        self._add_field(layout, "Embedding batch size",
                        self.embedding_batch_size,
                        "Texts per encode() call")

        # === ChromaDB HNSW Tuning ===
        self._add_section(layout, "ChromaDB HNSW Tuning")

        self.hnsw_ef = QSpinBox()
        self.hnsw_ef.setRange(16, 1024)
        self._add_field(layout, "EF construction", self.hnsw_ef,
                        "Higher = better recall, slower build")

        self.hnsw_m = QSpinBox()
        self.hnsw_m.setRange(4, 128)
        self._add_field(layout, "M (connections)", self.hnsw_m,
                        "Higher = better recall, more memory")

        # === MCP Server ===
        self._add_section(layout, "MCP Server")

        self.host = QLineEdit()
        self._add_field(layout, "Host", self.host, "Bind address for MCP HTTP server")

        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self._add_field(layout, "Port", self.port, "Port for MCP HTTP server")

        layout.addStretch()
        scroll.setWidget(scroll_widget)
        outer.addWidget(scroll)

        # === Bottom buttons ===
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.save_btn = QPushButton("Save Configuration")
        self.save_btn.setObjectName("primary-btn")
        self.save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(self.save_btn)

        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setObjectName("secondary-btn")
        self.reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(self.reset_btn)

        btn_row.addStretch()

        self.save_msg = QLabel("")
        self.save_msg.setObjectName("success-msg")
        btn_row.addWidget(self.save_msg)

        outer.addLayout(btn_row)

        self.refresh()

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _add_section(layout: QVBoxLayout, text: str) -> None:
        lbl = QLabel(text)
        lbl.setObjectName("config-section")
        layout.addWidget(lbl)

    @staticmethod
    def _add_field(
        layout: QVBoxLayout,
        label: str,
        widget: QWidget,
        hint: str = "",
    ) -> None:
        row = QHBoxLayout()
        row.setSpacing(10)
        lbl = QLabel(label + ":")
        lbl.setMinimumWidth(180)
        lbl.setStyleSheet("font-size: 13px; color: #c0c8d4;")
        row.addWidget(lbl)
        widget.setMinimumWidth(200)
        row.addWidget(widget)
        row.addStretch()
        layout.addLayout(row)
        if hint:
            h = QLabel(hint)
            h.setObjectName("config-hint")
            h.setContentsMargins(185, 0, 0, 0)
            layout.addWidget(h)

    @staticmethod
    def _set_combo_to_value(combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx < 0:
            combo.addItem(value)
            idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # --- actions ---------------------------------------------------------

    def _save_config(self) -> None:
        try:
            global _settings
            settings, _ = _init()

            settings.embedding_model = self.embedding_model.currentText().strip()
            settings.chunk_size = self.chunk_size.value()
            settings.chunk_overlap = self.chunk_overlap.value()
            settings.reranking_enabled = self.reranking_enabled.isChecked()
            settings.reranker_model = self.reranker_model.currentText().strip()
            settings.hybrid_search_enabled = self.hybrid_search_enabled.isChecked()
            settings.hybrid_search_alpha = round(self.hybrid_alpha_spin.value(), 2)
            settings.min_score_threshold = round(self.min_score_threshold.value(), 2)
            settings.mmr_enabled = self.mmr_enabled.isChecked()
            settings.mmr_lambda = round(self.mmr_lambda_spin.value(), 2)
            settings.indexing_workers = self.indexing_workers.value()
            settings.embedding_batch_size = self.embedding_batch_size.value()
            settings.hnsw_ef_construction = self.hnsw_ef.value()
            settings.hnsw_m = self.hnsw_m.value()
            settings.host = self.host.text().strip() or "127.0.0.1"
            settings.port = self.port.value()

            settings.save()
            _settings = settings  # update cached reference

            self.save_msg.setObjectName("success-msg")
            self.save_msg.setStyleSheet("color: #66bb6a; font-weight: bold;")
            self.save_msg.setText("Configuration saved.")
            self.main_window.show_status("Configuration saved.")

            # Clear message after 4 seconds
            QTimer.singleShot(4000, lambda: self.save_msg.setText(""))
        except Exception as e:
            self.save_msg.setObjectName("error-msg")
            self.save_msg.setStyleSheet("color: #ef5350; font-weight: bold;")
            self.save_msg.setText(f"Error: {e}")

    def _reset_defaults(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset Configuration",
            "Reset all settings to defaults?\n\nThis will overwrite your config.yaml.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        global _settings
        defaults = AppSettings()
        defaults.save()
        _settings = defaults
        self.refresh()
        self.save_msg.setStyleSheet("color: #66bb6a; font-weight: bold;")
        self.save_msg.setText("Reset to defaults.")
        self.main_window.show_status("Configuration reset to defaults.")
        QTimer.singleShot(4000, lambda: self.save_msg.setText(""))

    # --- refresh ---------------------------------------------------------

    def refresh(self) -> None:
        try:
            self._loading = True
            settings, _ = _init()

            self.path_label.setText(
                f"Data directory: {DATA_DIR}  |  Config file: {CONFIG_PATH}"
            )

            self._set_combo_to_value(self.embedding_model, settings.embedding_model)
            self.chunk_size.setValue(settings.chunk_size)
            self.chunk_overlap.setValue(settings.chunk_overlap)

            self.reranking_enabled.setChecked(settings.reranking_enabled)
            self._set_combo_to_value(self.reranker_model, settings.reranker_model)
            self.hybrid_search_enabled.setChecked(settings.hybrid_search_enabled)
            self.hybrid_alpha_slider.setValue(int(settings.hybrid_search_alpha * 100))
            self.hybrid_alpha_spin.setValue(settings.hybrid_search_alpha)
            self.min_score_threshold.setValue(settings.min_score_threshold)

            self.mmr_enabled.setChecked(settings.mmr_enabled)
            self.mmr_lambda_slider.setValue(int(settings.mmr_lambda * 100))
            self.mmr_lambda_spin.setValue(settings.mmr_lambda)

            self.indexing_workers.setValue(settings.indexing_workers)
            self.embedding_batch_size.setValue(settings.embedding_batch_size)

            self.hnsw_ef.setValue(settings.hnsw_ef_construction)
            self.hnsw_m.setValue(settings.hnsw_m)

            self.host.setText(settings.host)
            self.port.setValue(settings.port)

            self._loading = False
        except Exception:
            self._loading = False
            logger.exception("Config page refresh failed")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Application main window with sidebar navigation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Knowledge Base")
        self.setMinimumSize(960, 640)
        self.resize(1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self._build_sidebar()
        main_layout.addWidget(self.sidebar)

        # Stacked content area
        self.pages = QStackedWidget()
        self.pages.setObjectName("content-area")
        main_layout.addWidget(self.pages)

        # Create pages
        self.dashboard_page = DashboardPage(self)
        self.search_page = SearchPage(self)
        self.manage_page = ManagePage(self)
        self.indexing_page = IndexingPage(self)
        self.config_page = ConfigPage(self)

        self.pages.addWidget(self.dashboard_page)
        self.pages.addWidget(self.search_page)
        self.pages.addWidget(self.manage_page)
        self.pages.addWidget(self.indexing_page)
        self.pages.addWidget(self.config_page)

        # Status bar
        self.statusBar().showMessage("Ready")

        # Default to dashboard
        self.nav_list.setCurrentRow(0)

    def _build_sidebar(self) -> None:
        self.sidebar = QWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(220)

        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title = QLabel("\U0001f4da RAG Knowledge Base")
        title.setObjectName("sidebar-title")
        layout.addWidget(title)

        subtitle = QLabel("Manage your knowledge bases")
        subtitle.setObjectName("sidebar-subtitle")
        layout.addWidget(subtitle)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("nav-list")

        nav_items = [
            "\U0001f4ca  Dashboard",
            "\U0001f50d  Search",
            "\u2699\ufe0f  Manage",
            "\U0001f4c4  Indexing",
            "\U0001f6e0\ufe0f  Config",
        ]
        for text in nav_items:
            item = QListWidgetItem(text)
            item.setSizeHint(QSize(0, 44))
            self.nav_list.addItem(item)

        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        layout.addWidget(self.nav_list)

        layout.addStretch()

        version = QLabel("v1.0.0")
        version.setObjectName("sidebar-subtitle")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

    def _on_nav_changed(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        page = self.pages.currentWidget()
        refresh = getattr(page, "refresh", None)
        if callable(refresh):
            cast(_Refreshable, page).refresh()

    def show_status(self, message: str, timeout: int = 5000) -> None:
        self.statusBar().showMessage(message, timeout)

    def on_data_changed(self) -> None:
        """Refresh all pages after a data-changing operation."""
        self.dashboard_page.refresh()
        # Refresh manage page table
        if hasattr(self, "manage_page"):
            self.manage_page.refresh()
        # Refresh indexing page (active RAG info panel, file list, etc.)
        if hasattr(self, "indexing_page"):
            self.indexing_page.refresh()
        # Refresh search page
        if hasattr(self, "search_page"):
            self.search_page.refresh()

    def closeEvent(self, event) -> None:
        global _watcher, _store_cache
        if _watcher:
            _watcher.stop()
            _watcher = None
        # Close all cached VectorStores
        for store in _store_cache.values():
            try:
                store.close(force=True)
            except Exception:
                pass
        _store_cache.clear()
        # Try to clean up any deferred directory deletions now that
        # all ChromaDB clients are closed.
        try:
            import gc
            gc.collect()
            _, registry = _init()
            registry._run_pending_cleanups()
        except Exception:
            pass
        event.accept()


# ---------------------------------------------------------------------------
# Launch function
# ---------------------------------------------------------------------------


def launch_ui(**kwargs) -> None:
    """Launch the PySide6 desktop application."""
    # Windows taskbar grouping
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "ragkb.desktop.1.0"
        )
    except Exception:
        pass

    _init()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    elif not isinstance(app, QApplication):
        raise RuntimeError("Active Qt application is not a QApplication instance")

    app = cast(QApplication, app)

    app.setApplicationName("RAG Knowledge Base")
    app_palette = app.palette()
    app_palette.setColor(QPalette.ColorRole.Window, QColor("#14181f"))
    app_palette.setColor(QPalette.ColorRole.WindowText, QColor("#e4e8ef"))
    app_palette.setColor(QPalette.ColorRole.Base, QColor("#1d232e"))
    app_palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#242b37"))
    app_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1d232e"))
    app_palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#e4e8ef"))
    app_palette.setColor(QPalette.ColorRole.Text, QColor("#e4e8ef"))
    app_palette.setColor(QPalette.ColorRole.Button, QColor("#1d232e"))
    app_palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e4e8ef"))
    app_palette.setColor(QPalette.ColorRole.Highlight, QColor("#2f79c6"))
    app_palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(app_palette)

    app.setStyle("Fusion")
    app.setStyleSheet(_STYLESHEET)

    window = MainWindow()
    window.show()

    app.exec()
