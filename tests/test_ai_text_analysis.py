import json
from types import MethodType, SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from qualcoder.ai_chat import DialogAIChat
from qualcoder.ai_mcp_server import AiMcpServer


class TestAiTextAnalysis(TestCase):
    """Check selection limits and complete MCP loading before model invocation."""

    def setUp(self) -> None:
        translation = patch('qualcoder.ai_chat._', lambda text: text, create=True)
        translation.start()
        self.addCleanup(translation.stop)
        readiness = patch('qualcoder.ai_chat.ensure_ai_ready', return_value=True)
        readiness.start()
        self.addCleanup(readiness.stop)
        warning_patch = patch('qualcoder.ai_chat.Message')
        self.warning = warning_patch.start()
        self.addCleanup(warning_patch.stop)

        self.chat = MagicMock()
        self.chat.app.project_name = 'Test project'
        self.chat.app.settings = {'ai_enable': 'True'}
        self.chat.app.ai.large_llm_context_window = 20000
        self.chat.app.ai.is_current_run_canceled.return_value = False
        self.chat.app.ai._exception_summary.side_effect = str
        self.chat.current_chat_idx = 0
        self.prompt = SimpleNamespace(name='text-analysis/test', scope='project')
        self.chat._resolve_text_analysis_prompt_payload.return_value = self.prompt
        self.chat.agent_prompts_catalog.find_prompt_variant.return_value = self.prompt
        self.server = AiMcpServer.__new__(AiMcpServer)
        self.server._fetchone = MagicMock()
        self.server._fetch_file_attributes = MagicMock(return_value={})
        self.server.describe_status_event = MagicMock(return_value='Reading')
        self.chat.ai_mcp_server = self.server

        for method_name in ('_text_analysis_document_uri', '_compact_mcp_result_content'):
            setattr(self.chat, method_name, MethodType(getattr(DialogAIChat, method_name), self.chat))
        self.chat._mcp_base_system_prompt.return_value = 'Base instructions'
        self.chat._begin_ai_change_set.return_value = ''
        self.chat._empty_methodology_gate.return_value = {}
        self.chat._text_analysis_bootstrap_contract.return_value = 'Analyze the selected passage'
        self.chat._mcp_reflection_system_prompt.return_value = 'Reflect'
        self.chat._build_mcp_combined_system_prompt.side_effect = lambda text: text
        self.chat._merge_methodology_gate.return_value = {}
        self.chat._short_reflection_next_step_note.return_value = ''
        self.chat._is_methodology_gate_blocking.return_value = False
        self.chat._snapshot_methodology_gate.return_value = {}
        self.chat._normalize_progress_note.side_effect = lambda text, **kwargs: text
        self.chat._mcp_final_answer_phase_prompts.return_value = ('Final instructions', 'Answer now')
        self.chat._parse_mcp_reflection_response.return_value = {
            'action': 'final_answer',
            'reflection_summary': '',
            'answer_brief': '',
            'methodology_gate': {},
            'next_step_note': '',
            'continue_deferred_calls': False,
            'revised_calls': [],
            'proposed_next_calls': [],
            'user_decision_required': False,
            'decision_question': '',
            'decision_context': '',
        }
        self.chat._run_mcp_request.side_effect = self._read_resource

    def _read_resource(self, method: str, params: dict) -> tuple[dict, dict]:
        if method != 'resources/read':
            return {}, {'result': {}}
        uri = params['uri']
        _, window = self.server._parse_read_window(uri)
        payload = self.server._read_document(7, window)
        return {}, {'result': {'contents': [{'uri': uri, 'text': json.dumps(payload)}]}}

    @staticmethod
    def _loaded_passages(messages: list) -> list[dict]:
        passages = []
        for message in messages:
            if not message.content.startswith('MCP result:\n'):
                continue
            result_payload = json.loads(message.content.removeprefix('MCP result:\n'))
            if result_payload['method'] == 'resources/read':
                passages.extend(item['payload'] for item in result_payload['contents'])
        return passages

    def test_selection_uses_model_budget_and_rejects_excess_before_starting(self) -> None:
        # Half of 20,000 tokens at four characters per token is 40,000 characters.
        for text_length in (12000, 12001, 40000, 40001):
            with self.subTest(text_length=text_length):
                self.chat.reset_mock()
                self.warning.reset_mock()

                DialogAIChat.new_text_chat(self.chat, 7, 'Interview', 'x' * text_length, 137, self.prompt)

                if text_length <= 40000:
                    self.warning.assert_not_called()
                    self.chat._start_mcp_agent_worker.assert_called_once()
                    bootstrap_spec = self.chat._start_mcp_agent_worker.call_args.args[3]
                    self.assertEqual(text_length, bootstrap_spec['text_length'])
                else:
                    self.warning.return_value.exec.assert_called_once()
                    self.assertIn('too long', self.warning.call_args.args[2])
                    self.chat.new_chat.assert_not_called()
                    self.chat.main_window.ai_go_chat.assert_not_called()
                    self.chat._start_mcp_agent_worker.assert_not_called()

    def test_bootstrap_loads_only_the_complete_selection_before_model_invocation(self) -> None:
        for text_length in (37, 12000, 12001, 24000, 24001, 60000):
            with self.subTest(text_length=text_length):
                self.chat.reset_mock()
                self.server._fetchone.reset_mock()
                start_pos = 137
                selected_text = ('Interview: Grüß dich!\nAntwort: 😀\n' * text_length)[:text_length]
                fulltext = 'p' * start_pos + selected_text + 'Outside the selection'
                self.server._fetchone.return_value = (7, 'Interview', '', 'coder', fulltext)

                def reflect(messages: list, **kwargs) -> dict:
                    passages = self._loaded_passages(messages)
                    self.assertEqual(selected_text, ''.join(passage['text'] for passage in passages))
                    return {'action': 'final_answer'}

                self.chat._invoke_json_llm_with_step_timeout.side_effect = reflect
                result = DialogAIChat._mcp_text_analysis_worker(self.chat, [], 0, {
                    'doc_id': 7, 'doc_name': 'Interview', 'start_pos': start_pos,
                    'text_length': text_length, 'prompt_name': self.prompt.name,
                    'prompt_scope': self.prompt.scope,
                })

                self.assertNotIn('error', result)
                self.chat._invoke_json_llm_with_step_timeout.assert_called_once()
                passages = self._loaded_passages(result['stream_messages'])
                self.assertEqual(selected_text, ''.join(passage['text'] for passage in passages))
                self.assertEqual((text_length + 11999) // 12000, len(passages))
                next_start = start_pos
                for passage in passages:
                    self.assertEqual(next_start, passage['start'])
                    self.assertGreater(passage['length'], 0)
                    self.assertLessEqual(passage['length'], 12000)
                    next_start += passage['length']
                self.assertEqual(start_pos + text_length, next_start)

    def test_cancellation_between_chunks_prevents_analysis_of_partial_text(self) -> None:
        self.server._fetchone.return_value = (7, 'Interview', '', 'coder', 'x' * 24000)
        self.chat.app.ai.is_current_run_canceled.side_effect = [False] * 4 + [True]

        result = DialogAIChat._mcp_text_analysis_worker(self.chat, [], 0, {
            'doc_id': 7, 'doc_name': 'Interview', 'start_pos': 0,
            'text_length': 24000, 'prompt_name': self.prompt.name,
            'prompt_scope': self.prompt.scope,
        })

        self.assertTrue(result['canceled'])
        self.server._fetchone.assert_called_once()
        self.chat._invoke_json_llm_with_step_timeout.assert_not_called()
        self.assertEqual([], result['stream_messages'])
