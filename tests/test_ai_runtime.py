import subprocess
import sys
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from qualcoder.ai_runtime import AI_FAILED, AI_LOADING, AI_READY, ai_runtime_ready
from qualcoder.ai_runtime import show_ai_runtime_not_ready


class TestAiRuntime(TestCase):
    """Tests for the non-blocking AI startup boundary."""

    def test_ready_state(self):
        self.assertTrue(ai_runtime_ready(SimpleNamespace(ai_runtime_state=AI_READY)))
        self.assertFalse(ai_runtime_ready(SimpleNamespace(ai_runtime_state=AI_LOADING)))

    def test_loading_message_asks_user_to_retry(self):
        app = SimpleNamespace(ai_runtime_state=AI_LOADING)
        with patch("qualcoder.ai_runtime.Message") as message_class:
            show_ai_runtime_not_ready(app, "AI Agent")
        self.assertIn("retry", message_class.call_args.args[2].lower())
        message_class.return_value.exec.assert_called_once_with()

    def test_failure_message_does_not_claim_loading(self):
        app = SimpleNamespace(ai_runtime_state=AI_FAILED)
        with patch("qualcoder.ai_runtime.Message") as message_class:
            show_ai_runtime_not_ready(app, "AI Agent")
        self.assertIn("could not be loaded", message_class.call_args.args[2].lower())

    def test_lightweight_ai_llm_import_excludes_model_stack(self):
        code = (
            "import sys; import qualcoder.ai_llm; "
            "assert 'torch' not in sys.modules; "
            "assert 'sentence_transformers' not in sys.modules; "
            "assert 'langchain_openai' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
