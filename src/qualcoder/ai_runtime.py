"""Background loading support for the optional AI runtime."""

import logging
import traceback

from PyQt6 import QtCore

from .helpers import Message

logger = logging.getLogger(__name__)

AI_NOT_STARTED = "not_started"
AI_LOADING = "loading"
AI_INITIALIZING = "initializing"
AI_READY = "ready"
AI_FAILED = "failed"


class AiImportThread(QtCore.QThread):
    """Import expensive AI modules without blocking the GUI event loop."""

    loaded = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def run(self) -> None:
        """Load the complete AI module graph in this worker thread."""

        try:
            from . import ai_llm

            ai_llm.load_ai_runtime_dependencies()
            # Preload the dialogs as well. Their Qt widgets are only constructed
            # later, in the main-thread slot connected to ``loaded``.
            from . import ai_chat  # noqa: F401
            from . import ai_prompt_library  # noqa: F401
        except Exception:
            error_text = traceback.format_exc()
            logger.exception("Could not load the AI runtime")
            self.failed.emit(error_text)
            return
        self.loaded.emit()


def ai_runtime_ready(app) -> bool:
    """Return whether the application's AI runtime can be used."""

    return getattr(app, "ai_runtime_state", AI_NOT_STARTED) == AI_READY


def show_ai_runtime_not_ready(app, title: str = "AI") -> None:
    """Tell the user to retry later when background loading is incomplete."""

    state = getattr(app, "ai_runtime_state", AI_NOT_STARTED)
    if state == AI_FAILED:
        text = _("The AI components could not be loaded. Please restart QualCoder or check the log for details.")
    else:
        text = _("The AI components are still loading in the background. Please retry in a moment.")
    Message(app, title, text, "Information").exec()
