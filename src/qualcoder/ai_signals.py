"""Lightweight signals shared by coding views and the AI chat."""

from PyQt6.QtCore import QObject, pyqtSignal


class AIChatSignalEmitter(QObject):
    """Relay requests from coding views to the AI chat."""

    newTextChatSignal = pyqtSignal(int, str, str, int, object)


# Construct the QObject on the GUI thread during normal application imports.
ai_chat_signal_emitter = AIChatSignalEmitter()
