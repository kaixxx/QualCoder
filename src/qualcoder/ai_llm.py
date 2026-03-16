# -*- coding: utf-8 -*-

"""
This file is part of QualCoder.

QualCoder is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.

QualCoder is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with QualCoder.
If not, see <https://www.gnu.org/licenses/>.

Author: Kai Droege (kaixxx)
https://github.com/ccbogel/QualCoder
https://qualcoder.wordpress.com/
https://qualcoder-org.github.io/
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import traceback
from datetime import datetime
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui
import qtawesome as qta

from openai import OpenAI, BadRequestError
from .ai_prompts import PromptItem
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.globals import set_llm_cache  # Unused
from langchain_community.cache import InMemoryCache  # Unused
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage  # Unused
from langchain_core.messages.system import SystemMessage
from langchain_core.documents.base import Document  # Unused
from .ai_async_worker import Worker
from .ai_vectorstore import AiVectorstore
from .helpers import Message
from .select_items import DialogSelectItems
from .confirm_delete import DialogConfirmDelete
from .error_dlg import qt_exception_hook
from .html_parser import html_to_text
import json_repair
import asyncio
import configparser
from Bio.Align import PairwiseAligner
from pydantic import ValidationError
import re

max_memo_length = 1500  # Maximum length of the memo send to the AI

path = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
        

class MyCustomSyncHandler(BaseCallbackHandler):
    def __init__(self, ai_llm):
        self.ai_llm = ai_llm
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.ai_llm.ai_async_progress_count += 1
    

def extract_ai_memo(memo: str) -> str:
    """In any memo, any text after the mark '#####' are considered as personal notes that will not be send to the AI.
    This function extracts the text before this mark (or all text if no marking is found) 

    Args:
        memo (str): memo text

    Returns:
        str: shortened memo for AI
    """
    mark = memo.find('#####')
    if mark > -1:
        return memo[0:mark]
    else:
        return memo
    
def get_available_models(app, api_base: str, api_key: str) -> list:
    """Queries the API and returns a list of all AI models available from this provider."""
    msg = None
    if app is not None:
        msg = Message(app, _('AI Models'), _('Loading list of available AI models...'))
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.NoButton)
        msg.show()
        QtWidgets.QApplication.processEvents()
    try:        
        if api_base == '':
            api_base = None
        client = OpenAI(api_key=api_key, base_url=api_base)
        response = client.models.list(timeout=4.0)
        model_dict = response.model_dump().get('data', [])
        model_list = sorted([model['id'] for model in model_dict])
    finally:
        if msg is not None:
            msg.deleteLater()
    return model_list

def get_default_ai_models():
    ini_string = """
[ai_model_OpenAI GPT5.2 reasoning]
desc = Powerful model from OpenAI, with internal reasoning, for complex tasks.
	You need an API-key from OpenAI and have paid for credits in your account.
	OpenAI will charge a small amount for every use.
access_info_url = https://platform.openai.com/api-keys
large_model = gpt-5.2
large_model_context_window = 1000000
fast_model = gpt-5-mini
fast_model_context_window = 128000
reasoning_effort = medium
api_base = 
api_key = 

[ai_model_OpenAI GPT5.2 no reasoning]
desc = Powerful model from OpenAI, no reasoning, faster and cheaper.
	You need an API-key from OpenAI and have paid for credits in your account.
	OpenAI will charge a small amount for every use.
access_info_url = https://platform.openai.com/api-keys
large_model = gpt-5.2
large_model_context_window = 1000000
fast_model = gpt-5-mini
fast_model_context_window = 128000
reasoning_effort = low
api_base = 
api_key = 

[ai_model_Blablador]
desc = Free and open source models, excellent privacy, but not as powerful 
	as the commercial offerings. Blablador runs on a server of the Helmholtz 
	Society, a large non-profit research organization in Germany. To gain 
	access and get an API-key, you have to identify yourself once with your
	university, ORCID, GitHub, or Google account.
access_info_url = https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/
large_model = alias-large
large_model_context_window = 128000
fast_model = alias-fast
fast_model_context_window = 32000
reasoning_effort = default
api_base = https://api.helmholtz-blablador.fz-juelich.de/v1/
api_key = 

[ai_model_Blablador Huge]
desc = The largest and most powerful model currently running on Blablador. 
    Availability might change.
	Blablador is free to use and runs on a server of the Helmholtz Society,
	a large non-profit research organization in Germany. To gain
	access and get an API-key, you have to identify yourself once with your
	university, ORCID, GitHub, or Google account.
access_info_url = https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/
large_model = alias-huge
large_model_context_window = 128000
fast_model = alias-fast
fast_model_context_window = 128000
reasoning_effort = default
api_base = https://api.helmholtz-blablador.fz-juelich.de/v1/
api_key = 

[ai_model_Anthropic Claude Sonnet 4.5]
desc = Claude is a family of high quality models from Anthropic.
	You need an API-key from Anthropic and credits in your account.
	Anthropic will charge a small amount for every use.
access_info_url = https://console.anthropic.com/settings/keys
large_model = claude-sonnet-4-5
large_model_context_window = 200000
fast_model = claude-sonnet-4-5
fast_model_context_window = 200000
reasoning_effort = medium
api_base = https://api.anthropic.com/v1/
api_key = 

[ai_model_Google Gemini]
desc = Google offers several free and paid models on their servers.
	Select one in the Advanced AI options below.
	You need an API-key from Google.
access_info_url = https://ai.google.dev/gemini-api/docs
large_model = gemini-2.5-flash
large_model_context_window = 1000000
fast_model = gemini-2.5-flash
fast_model_context_window = 1000000
reasoning_effort = default
api_base = https://generativelanguage.googleapis.com/v1beta/openai/
api_key = 

[ai_model_Deepseek Chat V3]
desc = Deepseek is a high quality Chinese chat model.
	You will need an an API-key from Deepseek and have payed credits in your account.
	Deepseek will charge a small amount for every use.
access_info_url = https://platform.deepseek.com/api_keys
large_model = deepseek-chat
large_model_context_window = 64000
fast_model = deepseek-chat
fast_model_context_window = 64000
reasoning_effort = default
api_base = https://api.deepseek.com
api_key = 

[ai_model_Mistral]
desc = Mistral AI offers high-performance, open-source and proprietary language models, 
    prioritizing transparency, privacy, and ethical AI for researchers and developers.
access_info_url = https://mistral.ai
large_model = mistral-large-latest
large_model_context_window = 128000
fast_model = mistral-small-latest
fast_model_context_window = 128000
reasoning_effort = default
api_base = https://api.mistral.ai/v1
api_key =

[ai_model_OpenRouter]
desc = OpenRouter is a unified interface to access many different AI language
	models, both free and paid. You need an API-key from OpenRouter.
	Select a model in the Advanced AI Options below.
access_info_url = https://openrouter.ai/
large_model = deepseek/deepseek-chat:free
large_model_context_window = 64000
fast_model = deepseek/deepseek-chat:free
fast_model_context_window = 64000
reasoning_effort = default
api_base = https://openrouter.ai/api/v1
api_key = 

[ai_model_Ollama local AI]
desc = Ollama is an open source server that lets you run LLMs locally on
	your computer. To use it in QualCoder, you must have Ollama set up and
	running first. Use the Advanced AI Options below to select between your
	locally installed models.
access_info_url = https://ollama.com
large_model = 
large_model_context_window = 32000
fast_model = 
fast_model_context_window = 32000
reasoning_effort = default
api_base = http://localhost:11434/v1/
api_key = <no API key needed>

    """
    
    config = configparser.ConfigParser()
    config.read_string(ini_string)
    ai_models = []
    for section in config.sections():
        if section.startswith('ai_model_'):
            model = {
                'name': section[9:],
                'desc': config[section].get('desc', ''),
                'access_info_url': config[section].get('access_info_url', ''),
                'large_model': config[section].get('large_model', ''),
                'large_model_context_window': config[section].get('large_model_context_window', '32768'),
                'fast_model': config[section].get('fast_model', ''),
                'fast_model_context_window': config[section].get('fast_model_context_window', '32768'),
                'reasoning_effort': config[section].get('reasoning_effort', ''),
                'api_base': config[section].get('api_base', ''),
                'api_key': config[section].get('api_key', '')
            }
            ai_models.append(model)
    return ai_models

def add_new_ai_model(current_models: list, new_name: str) -> tuple[list, int]:
    """Adds a new AI profile to the list, sets some default values,
    and returns the extended list as well as the index of the new model 

    Args:
        current_models (list): AI profiles
        new_name (str): the name for the new profile

    Returns:
        tuple[list, int]: extended list, index of new profile
    """
    new_model = {
        'name': new_name,
        'desc': '',
        'access_info_url': '',
        'large_model': '',
        'large_model_context_window': '32768',
        'fast_model': '',
        'fast_model_context_window': '32768',
        'reasoning_effort': 'default',
        'api_base': '',
        'api_key': ''
    }
    current_models.append(new_model)
    return current_models, len(current_models) - 1

def update_ai_models(current_models: list, current_model_index: int) -> tuple[list, int]:
    """Update the AI model definitions, and add new models from the default set

    Args:
        current_models (list): the current list from config.ini
        current_model_index (int): the index of the currently selected model

    Returns:
        tuple[list, int]: updated list and current model index 
    """
    if current_model_index < 0 or current_model_index > (len(current_models) - 1):
        current_model_index = 0

    # add new models from the default model list
    default_models = get_default_ai_models()
    current_models_names = {model['name'] for model in current_models}
    for model in default_models:
        if not model['name'] in current_models_names:
            if model['name'] == 'OpenAI GPT5.1 reasoning': # insert this at the top, because it is the current default model
                current_models.insert(0, model)
                if current_model_index >= 0:
                    current_model_index += 1
            elif model['name'] == 'OpenAI GPT5.1 no reasoning' and len(current_models) > 1: # insert this at the second position
                current_models.insert(1, model)
                if current_model_index >= 1:
                    current_model_index += 1        
            else:
                current_models.append(model)
                
    # Blablador: update config (api base, model alias)
    curr_model = current_models[current_model_index]
    if curr_model['api_base'] == 'https://helmholtz-blablador.fz-juelich.de:8000/v1' and curr_model['large_model'] != 'alias-large':
        msg = _('You are using the "Blablador" service on an old server that will soon be disabled. '
                'Your configuration will be updated automatically. Please test if the AI access still works as expected. '
                'You might need to change to a different AI model in the settings dialog under "Advanced AI Settings".')
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle(_('AI Setup'))
        msg_box.setText(msg)
        msg_box.exec()
    for model in current_models:
        if model['api_base'] == 'https://helmholtz-blablador.fz-juelich.de:8000/v1':
            model['api_base'] = 'https://api.helmholtz-blablador.fz-juelich.de/v1/'
        if model['large_model'] == 'alias-llama3-huge': # this alias is no longer available
            model['large_model'] = 'alias-huge'
        if model['fast_model'] == 'alias-llama3-huge':
            model['fast_model'] = 'alias-huge'
    
    # add parameter "reasoning_effort"    
    for model in current_models:
        if model['reasoning_effort'] == '':
            if model['large_model'].lower().find('gpt-5') > -1 or \
                    model['large_model'].lower().find('o4') > -1 or \
                    model['large_model'].lower().find('o3') > -1 or \
                    model['large_model'].lower().find('o1') > -1 or \
                    model['large_model'].lower().find('gpt-oss') > -1 or \
                    model['large_model'].lower().find('qwen3') > -1 or \
                    model['large_model'].lower().find('opus') > -1 or \
                    model['large_model'].lower().find('sonnet') > -1 or \
                    model['large_model'].lower().find('grok-4') > -1:
                model['reasoning_effort'] = 'medium'
            else:
                model['reasoning_effort'] = 'default'
    
        # Correct an error in the QualCoder 3.8 release, where reasoning effort was set to medium for GPT-4.1:            
        if model['large_model'].lower().find('gpt-4.1') > -1: 
            model['reasoning_effort'] = 'default'
    
    return current_models, current_model_index

def strip_think_blocks(text: str) -> str:
    """
    Removes <think>...</think> blocks from an LLM response.
    If the closing </think> is missing (e.g., during streaming),
    removes everything from <think> to the end of the text.
    """
    # Case 1: Remove complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Case 2: Remove unfinished <think> blocks (no closing tag yet)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    return text.strip()

def ai_quote_search(quote: str, original: str) -> tuple[int, int]:
    """Searches the quote in the original text using the Smith-Waterman algorithm.
    This also tolerates gaps up to complete sentences in the cited text or other 
    minor differences in the exact wording.
    The "PairwiseAligner" is normally used to find partial overlaps in DNA-strings.
    Returns -1, -1 if no match is found.
    """
    
    # try finding an exact match first
    start_idx = original.find(quote)
    if start_idx > -1:
        return start_idx, start_idx + len(quote)
    
    # no exact match found, use the PairwiseAligner to find also partial matches
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 2         # score for each matched char
    aligner.mismatch_score = -1     # penalty for mismatched chars (errors)
    aligner.open_gap_score = -0.5   # penalty for opening a gap (left out chars)
    aligner.extend_gap_score = -0.1 # penalty for gap continuation

    alignments = aligner.align(original.lower(), quote.lower())
    try:
        best = next(alignments)
    except StopIteration: 
        # No alignments were found
        return -1, -1
    except Exception as e:
        # any other error (could be a memory issue)
        logger.error(e)
        return -1, -1

    orig_spans = best.aligned[0]
    if not len(orig_spans):
        return -1, -1

    # combine all matched blocks from the first start to the last end:
    start_idx = int(orig_spans[0][0])
    end_idx = int(orig_spans[-1][1])
        
    # only accept a match if it reaches 80% of the max score -> prevents false positives
    max_score = len(quote) * aligner.match_score
    score_fraction = best.score / max_score
    if score_fraction > 0.8:
        return start_idx, end_idx
    else:
        return -1, -1

class AiLLM():
    """ This manages the communication between qualcoder, the vectorstore 
    and the LLM (large language model, e.g. GPT-4)."""
    app = None
    parent_text_edit = None
    main_window = None
    threadpool: QtCore.QThreadPool = None
    ai_async_is_canceled = False
    ai_async_is_finished = False
    ai_async_is_errored = False
    ai_async_progress_msg = ''
    ai_async_progress_count = -1
    ai_async_progress_max = -1
    _status = ''   
    large_llm = None
    fast_llm = None
    large_llm_context_window = 128000
    fast_llm_context_window = 16385
    ai_streaming_output = ''
    sources_collection = 'qualcoder'  # name of the vectorstore collection for source documents
    ai_log_logger = None
    ai_log_handler = None
    ai_log_path = ''
    ai_log_seq = 0
    ai_change_history = None
    
    def __init__(self, app, parent_text_edit):
        self.app = app
        self.parent_text_edit = parent_text_edit
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount(1)
        self.sources_vectorstore = AiVectorstore(self.app, self.parent_text_edit, self.sources_collection)
        self.ai_change_history = []  # Session-scoped AI write operations for undo

    def _ai_log_target_path(self) -> str:
        """Return the file path for the dedicated AI communication log."""

        confighome = str(getattr(self.app, 'confighome', '')).strip()
        if confighome != '':
            log_dir = confighome
        else:
            log_dir = os.path.expanduser('~')
        return os.path.join(log_dir, 'ai.log')

    def _ensure_ai_log_logger(self):
        """Set up or refresh the dedicated rotating AI logger."""

        if self.ai_log_logger is None:
            self.ai_log_logger = logging.getLogger('qualcoder.ai_log')
            self.ai_log_logger.setLevel(logging.INFO)
            self.ai_log_logger.propagate = False
        if self.ai_log_handler is None and len(self.ai_log_logger.handlers) > 0:
            # Cleanup stale handlers from previous AiLLM instances in the same process.
            for stale_handler in list(self.ai_log_logger.handlers):
                try:
                    self.ai_log_logger.removeHandler(stale_handler)
                except Exception:
                    pass
                try:
                    stale_handler.close()
                except Exception:
                    pass

        desired_path = self._ai_log_target_path()
        if self.ai_log_handler is not None and self.ai_log_path == desired_path:
            return

        if self.ai_log_handler is not None:
            try:
                self.ai_log_logger.removeHandler(self.ai_log_handler)
            except Exception:
                pass
            try:
                self.ai_log_handler.close()
            except Exception:
                pass
            self.ai_log_handler = None

        os.makedirs(os.path.dirname(desired_path), exist_ok=True)
        handler = RotatingFileHandler(
            desired_path,
            maxBytes=500000,
            backupCount=2,
            encoding='utf-8',
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        self.ai_log_logger.addHandler(handler)
        self.ai_log_handler = handler
        self.ai_log_path = desired_path

    def _next_ai_log_id(self) -> int:
        self.ai_log_seq += 1
        return self.ai_log_seq

    def _llm_name_for_log(self, llm) -> str:
        for attr in ('model_name', 'model', 'deployment_name', 'azure_deployment'):
            try:
                value = str(getattr(llm, attr, '')).strip()
            except Exception:
                value = ''
            if value != '':
                return value
        model_idx = int(self.app.settings.get('ai_model_index', '-1'))
        if 0 <= model_idx < len(self.app.ai_models):
            return str(self.app.ai_models[model_idx].get('name', '')).strip()
        return 'unknown'

    def _message_role_and_text_for_log(self, msg) -> tuple[str, str]:
        if isinstance(msg, SystemMessage):
            return 'system', str(msg.content)
        if isinstance(msg, HumanMessage):
            return 'user', str(msg.content)
        if isinstance(msg, AIMessage):
            return 'assistant', str(msg.content)
        if isinstance(msg, dict):
            role = str(msg.get('role', 'message')).strip() or 'message'
            content = msg.get('content', '')
            return role, str(content)
        if hasattr(msg, 'content'):
            role = msg.__class__.__name__.replace('Message', '').lower() or 'message'
            return role, str(getattr(msg, 'content', ''))
        if isinstance(msg, str):
            return 'message', msg
        return 'message', str(msg)

    def _safe_to_text(self, value: object) -> str:
        """Best-effort conversion that never raises during error handling/logging."""

        try:
            return str(value)
        except Exception:
            try:
                return repr(value)
            except Exception:
                return '<unprintable>'

    def _write_ai_log(self, text: str):
        self._ensure_ai_log_logger()
        self.ai_log_logger.info(text)

    def log_llm_request(self, llm, messages, context: str = '') -> int:
        """Log one outgoing LLM request with readable message contents."""

        req_id = self._next_ai_log_id()
        model_name = self._llm_name_for_log(llm)
        header = f'[#{req_id}] REQUEST model="{model_name}"'
        if context.strip() != '':
            header += f' context="{context.strip()}"'
        lines = [header]

        if isinstance(messages, (list, tuple)):
            msg_list = list(messages)
        else:
            msg_list = [messages]
        for idx, msg in enumerate(msg_list, start=1):
            role, content = self._message_role_and_text_for_log(msg)
            lines.append(f'[{idx}] {role}:')
            content_lines = str(content).splitlines()
            if len(content_lines) == 0:
                lines.append('  ')
            else:
                for line in content_lines:
                    lines.append('  ' + line)
        self._write_ai_log('\n'.join(lines))
        return req_id

    def log_llm_response(self, req_id: int, llm, response_text, context: str = ''):
        """Log one final incoming LLM response."""

        model_name = self._llm_name_for_log(llm)
        header = f'[#{req_id}] RESPONSE model="{model_name}"'
        if context.strip() != '':
            header += f' context="{context.strip()}"'
        lines = [header, 'assistant:']
        for line in self._safe_to_text(response_text if response_text is not None else '').splitlines():
            lines.append('  ' + line)
        if len(lines) == 2:
            lines.append('  ')
        self._write_ai_log('\n'.join(lines))

    def log_llm_error(self, req_id: int, llm, err: Exception, context: str = ''):
        """Log one LLM error for traceability."""

        model_name = self._llm_name_for_log(llm)
        error_text = self._safe_to_text(err)
        line = f'[#{req_id}] ERROR model="{model_name}"'
        if context.strip() != '':
            line += f' context="{context.strip()}"'
        line += f' {err.__class__.__name__}: {error_text}'
        self._write_ai_log(line)

    def invoke_with_logging(self, llm, messages, response_format=None, config=None, context: str = '',
                            fallback_without_response_format: bool = False,
                            fallback_exceptions=(Exception,)):
        """Invoke an LLM and log request/response in the dedicated AI log."""

        req_id = self.log_llm_request(llm, messages, context=context)
        try:
            invoke_kwargs = {}
            if response_format is not None:
                invoke_kwargs['response_format'] = response_format
            if config is not None:
                invoke_kwargs['config'] = config
            res = llm.invoke(messages, **invoke_kwargs)
        except Exception as err:
            may_fallback = (
                fallback_without_response_format
                and response_format is not None
                and isinstance(err, fallback_exceptions)
            )
            if not may_fallback:
                self.log_llm_error(req_id, llm, err, context=context)
                raise
            try:
                fallback_kwargs = {}
                if config is not None:
                    fallback_kwargs['config'] = config
                res = llm.invoke(messages, **fallback_kwargs)
            except Exception as err2:
                self.log_llm_error(req_id, llm, err2, context=context)
                raise
        self.log_llm_response(req_id, llm, getattr(res, 'content', ''), context=context)
        return res

    def _ensure_ai_change_history(self):
        if not isinstance(self.ai_change_history, list):
            self.ai_change_history = []
        return self.ai_change_history

    def has_undoable_ai_changes(self) -> bool:
        """Return whether the current session contains AI changes that can be undone."""

        history = self._ensure_ai_change_history()
        for item in history:
            if not isinstance(item, dict):
                continue
            operations = item.get("operations", None)
            if isinstance(operations, list) and len(operations) > 0:
                return True
        return False

    def _ensure_ai_change_set(self, history, set_id: str):
        for item in history:
            if not isinstance(item, dict):
                continue
            if str(item.get("id", "")).strip() == set_id:
                ops = item.get("operations", None)
                if not isinstance(ops, list):
                    item["operations"] = []
                if "name" not in item or str(item.get("name", "")).strip() == "":
                    item["name"] = "" # _("AI changes")
                if "created_at" not in item:
                    item["created_at"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                return item
        new_set = {
            "id": set_id,
            "name": "",  # _("AI changes")
            "created_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "operations": [],
        }
        history.insert(0, new_set)
        return new_set

    def begin_ai_change_set(self, messages, chat_idx: int) -> str:
        """Create one session-scoped AI change set for the current chat turn."""

        history = self._ensure_ai_change_history()
        now = datetime.now().astimezone()
        set_id = f'ai-run-{now.strftime("%Y%m%d%H%M%S%f")}-{chat_idx}'
        title = "" # _("AI changes")
        title += "[" + now.strftime("%H:%M:%S") + "]"

        change_set = {
            "id": set_id,
            "name": title,
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "operations": [],
        }
        history.insert(0, change_set)
        return set_id

    def _short_change_label(self, text: str, max_len: int = 36) -> str:
        txt = " ".join(str(text if text is not None else "").split()).strip()
        if len(txt) <= max_len:
            return txt
        return txt[: max_len - 3] + "..."

    def _lookup_code_name(self, cid: int) -> str:
        if cid <= 0:
            return ""
        try:
            cur = self.app.conn.cursor()
            cur.execute("SELECT name FROM code_name WHERE cid=?", (cid,))
            row = cur.fetchone()
            if row is None:
                return ""
            return str(row[0] if row[0] is not None else "").strip()
        except Exception:
            return ""

    def _lookup_source_name(self, fid: int) -> str:
        if fid <= 0:
            return ""
        try:
            cur = self.app.conn.cursor()
            cur.execute("SELECT name FROM source WHERE id=?", (fid,))
            row = cur.fetchone()
            if row is None:
                return ""
            return str(row[0] if row[0] is not None else "").strip()
        except Exception:
            return ""

    def _format_name_line(self, prefix: str, names: list, max_items: int = 5) -> str:
        if not isinstance(names, list) or len(names) == 0:
            return ""
        trimmed = [n for n in names if str(n).strip() != ""]
        if len(trimmed) == 0:
            return ""
        shown = trimmed[:max_items]
        line = prefix + ", ".join(shown)
        remaining = len(trimmed) - len(shown)
        if remaining > 0:
            line += _(" (+{0} more)").format(remaining)
        return line

    def _format_ai_change_age(self, created_at: str) -> str:
        """Return a relative age label for one stored AI change set."""

        created_text = str(created_at if created_at is not None else "").strip()
        if created_text == "":
            return ""

        try:
            now = datetime.now().astimezone()
            created_dt = datetime.strptime(created_text, "%Y-%m-%d %H:%M:%S")
            created_dt = created_dt.replace(tzinfo=now.tzinfo)
        except Exception:
            return ""

        if created_dt > now:
            created_dt = now

        day_diff = (now.date() - created_dt.date()).days
        if day_diff > 0:
            if day_diff == 1:
                return _("1 day ago")
            return _("{0} days ago").format(day_diff)

        seconds_diff = max(0, int((now - created_dt).total_seconds()))
        minute_count = seconds_diff // 60
        if minute_count < 60:
            if minute_count == 1:
                return _("1 minute ago")
            return _("{0} minutes ago").format(minute_count)

        hour_count = minute_count // 60
        if hour_count == 1:
            return _("1 hour ago")
        return _("{0} hours ago").format(hour_count)

    def _blend_color(self, base: QtGui.QColor, overlay: QtGui.QColor, amount: float) -> QtGui.QColor:
        """Blend palette colors to create readable separators in light and dark themes."""

        factor = max(0.0, min(1.0, float(amount)))
        red = int(round(base.red() * (1.0 - factor) + overlay.red() * factor))
        green = int(round(base.green() * (1.0 - factor) + overlay.green() * factor))
        blue = int(round(base.blue() * (1.0 - factor) + overlay.blue() * factor))
        return QtGui.QColor(red, green, blue)

    def _ai_change_list_stylesheet(self, list_view: QtWidgets.QListView) -> str:
        palette = list_view.palette()
        base_color = palette.color(QtGui.QPalette.ColorRole.Base)
        text_color = palette.color(QtGui.QPalette.ColorRole.Text)
        blend_amount = 0.42 if base_color.lightness() < 128 else 0.28
        separator_color = self._blend_color(base_color, text_color, blend_amount)
        return (
            "QListView::item {"
            f"border-bottom: 1px solid {separator_color.name()};"
            "}"
        )

    def _refresh_ai_change_set_name(
            self,
            change_set: dict,
            allow_db_lookup: bool = False,
            use_relative_time: bool = False) -> None:
        """Update one change-set title from recorded operations."""

        if not isinstance(change_set, dict):
            return
        operations = change_set.get("operations", None)
        if not isinstance(operations, list) or len(operations) == 0:
            return

        category_count = 0
        code_count = 0
        coding_count = 0
        category_names = []
        code_names = []
        coding_targets = {}
        seen_category_names = set()
        seen_code_names = set()

        for op in operations:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("type", "")).strip()
            if op_type == "create_category":
                category_count += 1
                cat_name = self._short_change_label(op.get("name", ""))
                if cat_name != "" and cat_name not in seen_category_names:
                    seen_category_names.add(cat_name)
                    category_names.append(cat_name)
            elif op_type == "create_code":
                code_count += 1
                code_name = self._short_change_label(op.get("name", ""))
                if code_name != "" and code_name not in seen_code_names:
                    seen_code_names.add(code_name)
                    code_names.append(code_name)
            elif op_type == "create_coding_text":
                coding_count += 1
                cid = int(op.get("cid", -1))
                fid = int(op.get("fid", -1))
                code_name = self._short_change_label(op.get("code_name", ""))
                source_name = self._short_change_label(op.get("source_name", ""))
                if allow_db_lookup and code_name == "":
                    code_name = self._short_change_label(self._lookup_code_name(cid))
                if allow_db_lookup and source_name == "":
                    source_name = self._short_change_label(self._lookup_source_name(fid))
                if code_name == "":
                    if cid > 0:
                        code_name = _("Code") + " #" + str(cid)
                    else:
                        code_name = _("Code")
                if source_name == "":
                    if fid > 0:
                        source_name = _("Document") + " #" + str(fid)
                    else:
                        source_name = _("Document")
                key = code_name + " @ " + source_name
                coding_targets[key] = int(coding_targets.get(key, 0)) + 1

        parts = []
        if category_count > 0:
            parts.append(str(category_count) + " " + _("category(ies)"))
        if code_count > 0:
            parts.append(str(code_count) + " " + _("code(s)"))
        if coding_count > 0:
            parts.append(str(coding_count) + " " + _("text coding(s)"))
        if len(parts) == 0:
            return

        created_at = str(change_set.get("created_at", "")).strip()
        time_label = created_at
        if use_relative_time:
            relative_time = self._format_ai_change_age(created_at)
            if relative_time != "":
                time_label = relative_time
        elif len(created_at) >= 19:
            time_label = created_at[11:19] # extract HH:MM:SS from timestamp

        title = "" # _("AI changes")
        if time_label != "":
            if use_relative_time:
                title += time_label
            else:
                title += "[" + time_label + "]"
        first_line = title if title != "" else ", ".join(parts)
        lines = [first_line]
        categories_line = self._format_name_line(_("Categories: "), category_names)
        if categories_line != "":
            lines.append(categories_line)
        codes_line = self._format_name_line(_("Codes: "), code_names)
        if codes_line != "":
            lines.append(codes_line)
        if len(coding_targets) > 0:
            shown_targets = []
            for label, count in coding_targets.items():
                if int(count) > 1:
                    shown_targets.append(label + " (" + str(int(count)) + ")")
                else:
                    shown_targets.append(label)
            codings_line = self._format_name_line(_("Codings: "), shown_targets)
            if codings_line != "":
                lines.append(codings_line)
        change_set["name"] = "\n".join(lines)

    def discard_empty_ai_change_set(self, set_id: str) -> None:
        """Remove a pre-created AI change set if no write operation was recorded."""

        normalized_id = str(set_id).strip()
        if normalized_id == "":
            return
        history = self._ensure_ai_change_history()
        for idx, item in enumerate(history):
            if not isinstance(item, dict):
                continue
            if str(item.get("id", "")).strip() != normalized_id:
                continue
            operations = item.get("operations", None)
            if isinstance(operations, list) and len(operations) == 0:
                history.pop(idx)
            return

    def record_ai_change(self, change_set_id: str, operation: dict) -> None:
        """Append one operation to the session-scoped AI change history."""

        if not isinstance(operation, dict):
            return
        history = self._ensure_ai_change_history()
        normalized_set_id = str(change_set_id).strip()
        if normalized_set_id == "":
            normalized_set_id = "ai-run-" + datetime.now().astimezone().strftime("%Y%m%d%H%M%S%f")
        change_set = self._ensure_ai_change_set(history, normalized_set_id)
        op_copy = dict(operation)
        op_copy["change_set_id"] = normalized_set_id
        change_set["operations"].append(op_copy)
        self._refresh_ai_change_set_name(change_set)

    def _table_exists(self, table_name: str) -> bool:
        cur = self.app.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cur.fetchone() is not None

    def _can_undo_create_category(self, cur, op):
        catid = int(op.get("catid", -1))
        if catid <= 0:
            return False, "invalid", None
        cur.execute("SELECT catid, name, owner FROM code_cat WHERE catid=?", (catid,))
        row = cur.fetchone()
        if row is None:
            return False, "missing", None
        expected_name = str(op.get("name", "")).strip()
        if expected_name != "" and row[1] != expected_name:
            return False, "changed", row
        if str(row[2]) != "AI Agent":
            return False, "changed", row
        return True, "ok", row

    def _can_undo_create_code(self, cur, op):
        cid = int(op.get("cid", -1))
        if cid <= 0:
            return False, "invalid", None
        cur.execute("SELECT cid, name, owner FROM code_name WHERE cid=?", (cid,))
        row = cur.fetchone()
        if row is None:
            return False, "missing", None
        expected_name = str(op.get("name", "")).strip()
        if expected_name != "" and row[1] != expected_name:
            return False, "changed", row
        if str(row[2]) != "AI Agent":
            return False, "changed", row
        return True, "ok", row

    def _can_undo_create_coding_text(self, cur, op):
        ctid = int(op.get("ctid", -1))
        if ctid <= 0:
            return False, "invalid", None
        cur.execute(
            "SELECT ctid, cid, fid, pos0, pos1, owner, ifnull(seltext,'') FROM code_text WHERE ctid=?",
            (ctid,),
        )
        row = cur.fetchone()
        if row is None:
            return False, "missing", None
        expected = {
            "cid": int(op.get("cid", -1)),
            "fid": int(op.get("fid", -1)),
            "pos0": int(op.get("pos0", -1)),
            "pos1": int(op.get("pos1", -1)),
            "owner": str(op.get("owner", "AI Agent")),
            "seltext": str(op.get("seltext", "")),
        }
        if expected["cid"] > 0 and row[1] != expected["cid"]:
            return False, "changed", row
        if expected["fid"] > 0 and row[2] != expected["fid"]:
            return False, "changed", row
        if expected["pos0"] >= 0 and row[3] != expected["pos0"]:
            return False, "changed", row
        if expected["pos1"] >= 0 and row[4] != expected["pos1"]:
            return False, "changed", row
        if expected["owner"] != "" and row[5] != expected["owner"]:
            return False, "changed", row
        if expected["seltext"] != "" and row[6] != expected["seltext"]:
            return False, "changed", row
        return True, "ok", row

    def _count_code_codings(self, cur, cid: int) -> tuple[int, int]:
        total = 0
        non_ai = 0
        for table in ("code_text", "code_av", "code_image"):
            if not self._table_exists(table):
                continue
            cur.execute(
                f"SELECT count(*), sum(case when owner != 'AI Agent' then 1 else 0 end) FROM {table} WHERE cid=?",
                (cid,),
            )
            row = cur.fetchone()
            if row is None:
                continue
            total += int(row[0] or 0)
            non_ai += int(row[1] or 0)
        return total, non_ai

    def _build_ai_change_impact_text(self, change_set):
        operations = change_set.get("operations", [])
        if not isinstance(operations, list) or len(operations) == 0:
            return ""
        cur = self.app.conn.cursor()

        code_ids = set()
        category_ids = set()
        coding_ctids = set()
        skipped_changed = 0
        skipped_missing = 0

        for op in operations:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("type", "")).strip()
            if op_type == "create_code":
                ok, reason, row_data = self._can_undo_create_code(cur, op)
                if ok:
                    code_ids.add(int(op.get("cid", -1)))
                elif reason == "changed":
                    skipped_changed += 1
                elif reason == "missing":
                    skipped_missing += 1
            elif op_type == "create_category":
                ok, reason, row_data = self._can_undo_create_category(cur, op)
                if ok:
                    category_ids.add(int(op.get("catid", -1)))
                elif reason == "changed":
                    skipped_changed += 1
                elif reason == "missing":
                    skipped_missing += 1
            elif op_type == "create_coding_text":
                ok, reason, row_data = self._can_undo_create_coding_text(cur, op)
                if ok:
                    ctid = int(op.get("ctid", -1))
                    cid = int(op.get("cid", -1))
                    if ctid > 0 and cid not in code_ids:
                        coding_ctids.add(ctid)
                elif reason == "changed":
                    skipped_changed += 1
                elif reason == "missing":
                    skipped_missing += 1

        code_codings_total = 0
        code_codings_non_ai = 0
        for cid in code_ids:
            ct_total, ct_non_ai = self._count_code_codings(cur, cid)
            code_codings_total += ct_total
            code_codings_non_ai += ct_non_ai

        detach_codes = 0
        detach_subcats = 0
        for catid in category_ids:
            cur.execute("SELECT count(*) FROM code_name WHERE catid=?", (catid,))
            detach_codes += int((cur.fetchone() or [0])[0] or 0)
            cur.execute("SELECT count(*) FROM code_cat WHERE supercatid=?", (catid,))
            detach_subcats += int((cur.fetchone() or [0])[0] or 0)

        standalone_codings_total = 0
        standalone_codings_non_ai = 0
        for ctid in coding_ctids:
            cur.execute("SELECT owner FROM code_text WHERE ctid=?", (ctid,))
            row = cur.fetchone()
            if row is None:
                continue
            standalone_codings_total += 1
            if str(row[0]) != "AI Agent":
                standalone_codings_non_ai += 1

        lines = []
        if len(code_ids) > 0:
            lines.append(
                _("Undo will remove ") + str(len(code_ids)) + _(" code(s) and ") +
                str(code_codings_total) + _(" related coding(s).")
            )
            if code_codings_non_ai > 0:
                lines.append(
                    _("Warning: ") + str(code_codings_non_ai) +
                    _(" of these codings are not owned by 'AI Agent'.")
                )
        if len(category_ids) > 0:
            lines.append(
                _("Undo will remove ") + str(len(category_ids)) + _(" category(ies). ") +
                str(detach_codes) + _(" code(s) and ") + str(detach_subcats) +
                _(" subcategory(ies) would be detached, not deleted.")
            )
        if standalone_codings_total > 0:
            lines.append(
                _("Undo will remove ") + str(standalone_codings_total) + _(" standalone text coding(s).")
            )
            if standalone_codings_non_ai > 0:
                lines.append(
                    _("Warning: ") + str(standalone_codings_non_ai) +
                    _(" standalone coding(s) are not owned by 'AI Agent'.")
                )
        if skipped_changed > 0:
            lines.append(
                str(skipped_changed) + _(" operation(s) appear changed since creation and may be skipped.")
            )
        if skipped_missing > 0:
            lines.append(
                str(skipped_missing) + _(" operation(s) are already missing and may be skipped.")
            )
        return "\n".join(lines)

    def _undo_ai_change_set(self, change_set):
        operations = change_set.get("operations", [])
        if not isinstance(operations, list):
            return {"undone": 0, "skipped": 0}
        stats = {
            "undone": 0,
            "skipped_changed": 0,
            "skipped_missing": 0,
            "skipped_invalid": 0,
            "deleted_code_codings": 0,
            "deleted_code_codings_non_ai": 0,
        }
        cur = self.app.conn.cursor()
        try:
            for op in reversed(operations):
                if not isinstance(op, dict):
                    stats["skipped_invalid"] += 1
                    continue
                op_type = str(op.get("type", "")).strip()

                if op_type == "create_coding_text":
                    ok, reason, row = self._can_undo_create_coding_text(cur, op)
                    if not ok:
                        if reason == "changed":
                            stats["skipped_changed"] += 1
                        elif reason == "missing":
                            stats["skipped_missing"] += 1
                        else:
                            stats["skipped_invalid"] += 1
                        continue
                    cur.execute("DELETE FROM code_text WHERE ctid=?", (int(row[0]),))
                    if cur.rowcount > 0:
                        stats["undone"] += 1
                    continue

                if op_type == "create_code":
                    ok, reason, row = self._can_undo_create_code(cur, op)
                    if not ok:
                        if reason == "changed":
                            stats["skipped_changed"] += 1
                        elif reason == "missing":
                            stats["skipped_missing"] += 1
                        else:
                            stats["skipped_invalid"] += 1
                        continue
                    cid = int(row[0])
                    coding_total, coding_non_ai = self._count_code_codings(cur, cid)
                    stats["deleted_code_codings"] += coding_total
                    stats["deleted_code_codings_non_ai"] += coding_non_ai
                    if self._table_exists("code_text"):
                        cur.execute("DELETE FROM code_text WHERE cid=?", (cid,))
                    if self._table_exists("code_av"):
                        cur.execute("DELETE FROM code_av WHERE cid=?", (cid,))
                    if self._table_exists("code_image"):
                        cur.execute("DELETE FROM code_image WHERE cid=?", (cid,))
                    cur.execute("DELETE FROM code_name WHERE cid=?", (cid,))
                    if cur.rowcount > 0:
                        stats["undone"] += 1
                    continue

                if op_type == "create_category":
                    ok, reason, row = self._can_undo_create_category(cur, op)
                    if not ok:
                        if reason == "changed":
                            stats["skipped_changed"] += 1
                        elif reason == "missing":
                            stats["skipped_missing"] += 1
                        else:
                            stats["skipped_invalid"] += 1
                        continue
                    catid = int(row[0])
                    cur.execute("UPDATE code_name SET catid=NULL WHERE catid=?", (catid,))
                    cur.execute("UPDATE code_cat SET supercatid=NULL WHERE supercatid=?", (catid,))
                    cur.execute("DELETE FROM code_cat WHERE catid=?", (catid,))
                    if cur.rowcount > 0:
                        stats["undone"] += 1
                    continue

                stats["skipped_invalid"] += 1

            self.app.conn.commit()
            self.app.delete_backup = False
        except Exception:
            self.app.conn.rollback()
            raise
        return stats

    def undo_ai_agent_changes(self):
        """Undo one selected AI-agent change set from the current app session."""

        history = self._ensure_ai_change_history()
        options = []
        for item in history:
            if not isinstance(item, dict):
                continue
            ops = item.get("operations", [])
            if isinstance(ops, list) and len(ops) > 0:
                self._refresh_ai_change_set_name(item, allow_db_lookup=True, use_relative_time=True)
                options.append(item)
        if len(options) == 0:
            Message(self.app, _("Undo AI changes"), _("No AI changes available to undo.")).exec()
            return

        ui = DialogSelectItems(self.app, options, _("Select AI changes to undo"), "single")
        ui.ui.listView.setStyleSheet(self._ai_change_list_stylesheet(ui.ui.listView))
        
        ok = ui.exec()
        if not ok:
            return
        selected = ui.get_selected()
        if not isinstance(selected, dict):
            return

        impact_text = self._build_ai_change_impact_text(selected)
        confirm_text = _("Undo AI changes from {timestamp}").format(timestamp=str(selected.get("name", "")))
        if impact_text != "":
            confirm_text += "\n\n" + impact_text
        confirm_text += "\n\n" + _("Do you want to continue?")
        confirm = DialogConfirmDelete(self.app, confirm_text, _("Undo AI changes"))
        if not confirm.exec():
            return

        stats = self._undo_ai_change_set(selected)
        if selected in history:
            history.remove(selected)

        msg = _("Undo AI changes from {timestamp}").format(timestamp=str(selected.get("name", ""))) + "\n"
        msg += _("Undone operations: ") + str(stats.get("undone", 0)) + "\n"
        skipped = int(stats.get("skipped_changed", 0)) + int(stats.get("skipped_missing", 0)) + int(stats.get("skipped_invalid", 0))
        if skipped > 0:
            msg += _("Skipped operations: ") + str(skipped) + "\n"
        non_ai_loss = int(stats.get("deleted_code_codings_non_ai", 0))
        if non_ai_loss > 0:
            msg += _("Warning: removed codings not owned by 'AI Agent': ") + str(non_ai_loss) + "\n"
        if self.parent_text_edit is not None:
            try:
                self.parent_text_edit.append(msg)
            except Exception:
                pass
        Message(self.app, _("Undo AI changes"), msg).exec()

    # Icons (https://pictogrammers.com/library/mdi/)
    def code_analysis_icon(self):
        return qta.icon('mdi6.tag-text-outline', color=self.app.highlight_color())

    def topic_analysis_icon(self):
        return qta.icon('mdi6.star-outline', color=self.app.highlight_color())

    def search_icon(self):
        return qta.icon('mdi6.magnify', color=self.app.highlight_color())
    
    def text_analysis_icon(self):
        return qta.icon('mdi6.text-box-outline', color=self.app.highlight_color())

    def general_chat_icon(self):
        return qta.icon('mdi6.chat-question-outline', color=self.app.highlight_color())

    def prompt_scope_icon(self):
        return qta.icon('mdi6.folder-open-outline', color=self.app.highlight_color())

    def prompt_icon(self):
        return qta.icon('mdi6.script-text-outline', color=self.app.highlight_color())
        
    def init_llm(self, main_window, rebuild_vectorstore=False, enable_ai=False):  
        try:
            self.main_window = main_window      
            if enable_ai or self.app.settings['ai_enable'] == 'True':
                self.parent_text_edit.append(_('AI: Starting up...'))
                QtWidgets.QApplication.processEvents()  # update ui
                self._status = 'starting'

                # init LLMs
                # set_llm_cache(InMemoryCache())
                if int(self.app.settings['ai_model_index']) >= len(self.app.ai_models): # model index out of range
                    self.app.settings['ai_model_index'] = -1
                if int(self.app.settings['ai_model_index']) < 0:
                    msg = _('AI: In the follwoing window, please set up the AI model.')
                    Message(self.app, _('AI Setup'), msg).exec()

                    main_window.change_settings(section='AI', enable_ai=True)
                    if int(self.app.settings['ai_model_index']) < 0:
                        # Still no model selected, disable AI:
                        self.app.settings['ai_enable'] = 'False'
                        self.parent_text_edit.append(_('AI: No model selected, AI is disabled.'))
                        self._status = ''
                        return
                    else: 
                        # Success, model was selected. But since the "change_settings" function will start 
                        # a new "init_llm" anyway, we are going to quit here
                        return    
                
                curr_model = self.app.ai_models[int(self.app.settings['ai_model_index'])]
                                                    
                # OpenAI: Check for outdated models:            
                if curr_model['large_model'].find('gpt-4-turbo') > -1:
                    self.parent_text_edit.append(_('AI: You are still using the outdated GPT-4 turbo. Consider switching to a newer model, such as GPT 4.1. Go to Project > Settings to change the AI profile and model.'))
                
                # Anthropic: Check for outdated models
                if curr_model['large_model'] == 'claude-opus-4-20250514':
                    self.parent_text_edit.append(_('AI: You are using the outdated Claude Opus 4 model from Anthropic. Consider switching to a newer model, such as Opus 4.1. Go to Project > Settings to change the AI profile and model.'))
                
                large_model = curr_model['large_model']
                self.large_llm_context_window = int(curr_model['large_model_context_window'])
                fast_model = curr_model['fast_model']
                self.fast_llm_context_window = int(curr_model['fast_model_context_window'])
                api_base = curr_model['api_base']
                api_key = curr_model['api_key']
                if api_key == '':
                    msg = _('Please enter an API-key for the AI in the following dialog.')
                    Message(self.app, _('AI API-key'), msg).exec()
                    main_window.change_settings(section='AI', enable_ai=True)
                    curr_model = self.app.ai_models[int(self.app.settings['ai_model_index'])]
                    if curr_model['api_key'] == '':
                        # still no API-key, disable AI:
                        self.app.settings['ai_enable'] = 'False'
                        self.parent_text_edit.append(_('AI: No API key set, AI is disabled.'))
                        self._status = ''
                        return
                    else: 
                        # Success, API-key was set. But since the "change_settings" function will start 
                        # a new "init_llm" anyways, we are going to quit here
                        return    
                if large_model == '' or fast_model == '':
                    msg = _('In the following dialog, go to "Advanced AI Options" and select a large and a fast AI model (both can be the same).')
                    Message(self.app, _('AI Model Selection'), msg).exec()
                    main_window.change_settings(section='advanced AI', enable_ai=True)
                    curr_model = self.app.ai_models[int(self.app.settings['ai_model_index'])]
                    if curr_model['large_model'] == '' or curr_model['fast_model'] == '':
                        # still no model chosen, disable AI:
                        self.app.settings['ai_enable'] = 'False'
                        self.parent_text_edit.append(_('AI: No large/fast model selected, AI is disabled.'))
                        self._status = ''
                        return
                    else: 
                        # Success, models were selected. But since the "change_settings" function will start 
                        # a new "init_llm" anyways, we are going to quit here
                        return
                temp = float(self.app.settings.get('ai_temperature', '1.0'))
                top_p = float(self.app.settings.get('ai_top_p', '1.0'))
                timeout = float(self.app.settings.get('ai_timeout', '30.0'))
                self.app.settings['ai_timeout'] = str(timeout)
                if api_base.find('azure.com') != -1:  # using Microsoft Azure
                    is_azure = True
                    large_llm_params = {
                        'azure_endpoint': api_base,
                        'azure_deployment': large_model,    
                        'api_version': '2024-12-01-preview',
                        'api_key': api_key,
                        'temperature': temp,
                        'top_p': top_p,
                        'max_tokens': None,
                        'timeout': timeout,
                        'max_retries': 2,
                        'cache': False,
                        'streaming': True
                    }        
                    fast_llm_params = large_llm_params.copy()
                    fast_llm_params['model'] = fast_model
                else: # OpenAI or compatible API
                    is_azure = False
                    large_llm_params = {
                        'model': large_model, 
                        'openai_api_key': api_key, 
                        'openai_api_base': api_base, 
                        'cache': False,
                        'temperature': temp,
                        'top_p': top_p,
                        'streaming': True,
                        'timeout': timeout,
                    }   
                    fast_llm_params = large_llm_params.copy()
                    fast_llm_params['model'] = fast_model
                    
                if 'reasoning_effort' in curr_model and curr_model['reasoning_effort'] in ['low', 'medium', 'high']:
                    large_llm_params['reasoning_effort'] = curr_model['reasoning_effort']
                    # raise the timeout for reasoning models
                    large_llm_params['timeout'] = (['low', 'medium', 'high'].index(curr_model['reasoning_effort']) + 1) * timeout
                
                if large_model.lower().find('claude') != -1:  # Anthropic
                    # omitting top_p, since Antrhopic does not accept temperature and top_p at the same time
                    try:
                        large_llm_params.pop('top_p')
                        fast_llm_params.pop('top_p')
                    except:
                        pass
                
                if is_azure:
                    self.large_llm = AzureChatOpenAI(**large_llm_params)
                    self.fast_llm = AzureChatOpenAI(**fast_llm_params)
                else:    
                    self.large_llm = ChatOpenAI(**large_llm_params)
                    self.fast_llm = ChatOpenAI(**fast_llm_params)

                self.ai_streaming_output = ''
                self.app.settings['ai_enable'] = 'True'
                
                # init vectorstore
                if not self.sources_vectorstore.is_open():
                    self.sources_vectorstore.init_vectorstore(rebuild_vectorstore)
                else:
                    self._status = ''
                    self.parent_text_edit.append(_('AI: Ready'))
            else:
                self.close()
        except Exception as e:
            type_e = type(e)
            value = e
            tb_obj = e.__traceback__
            # log the exception and show error msg
            qt_exception_hook.exception_hook(type_e, value, tb_obj)
            self.close()
            self.app.settings['ai_enable'] = 'False'
            msg = _('An error occured during AI initialization. The AI features will be disabled. Click on Project > Settings to reenable them.')
            Message(self.app, _('AI Initialization'), msg, 'Information').exec()
        
    def close(self):
        self._status = 'closing'
        self.cancel(False)
        self.sources_vectorstore.close()
        self.large_llm = None
        self.fast_llm = None
        self._status = ''
        
    def cancel(self, ask: bool) -> bool:
        if not self.is_busy():
            return True
        if ask:
            msg = _('Do you really want to cancel the AI operation?')
            msg_box = Message(self.app, 'AI Cancel', msg)
            msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            reply = msg_box.exec()
            if reply == QtWidgets.QMessageBox.StandardButton.No:
                return False
        # cancel all waiting threads:
        self.threadpool.clear()
        self.ai_async_is_canceled = True
        self.threadpool.waitForDone(5000)
        if ask and self.is_busy():
            msg = _('The AI operation could not be aborted immediately. It may take a moment for the AI to be ready again.')
            msg_box = Message(self.app, 'AI Cancel', msg)
            msg_box.exec()            
        return True

    def get_status(self) -> str:
        """Return the status of the AI system:
        - 'disabled'
        - 'starting' (in the process of loading all its modules)
        - 'no data' (the vectorstore is not available, propably because no project is open)
        - 'reading data' (in the process of adding empirical douments to its internal memory)
        - 'busy' (in the process of sending a prompt to the LLM and streaming the response)
        - 'ready' (fully loaded and idle, ready for a task)
        - 'closing' (in the process of shutting down)
        - 'closed' 
        """
        if self._status != '':
            return self._status  # 'starting' and 'closing' are set by the corresponding procedures
        elif self.app.settings['ai_enable'] != 'True':
            return 'disabled'
        elif self.sources_vectorstore is None or not self.sources_vectorstore.is_open():
            return 'no data'
        elif self.sources_vectorstore.ai_worker_running():
            return 'reading data'
        elif self.large_llm is None or self.fast_llm is None:
            return 'closed'
        elif self.threadpool.activeThreadCount() > 0:
            return 'busy'
        else:
            return 'ready'
    
    def is_busy(self) -> bool:
        return self.get_status() == 'busy'
        # return self.threadpool.activeThreadCount() > 0

    def is_ready(self):
        return self.get_status() == 'ready'
        #return (self.sources_vectorstore is not None) and \
        #            (self.sources_vectorstore.is_ready()) and \
        #            (self.large_llm is not None) and \
        #            (self.fast_llm is not None) and \
        #            (not self.is_busy())
    
    def get_default_system_prompt(self) -> str:
        p = 'You are assisting a team of qualitative social researchers.'
        project_memo = extract_ai_memo(self.app.get_project_memo())
        if self.app.settings.get('ai_send_project_memo', 'True') == 'True' and len(project_memo) > 0:
            p += f' Here is some background information about the research project the team is working on:\n{project_memo}'
        return p
        
    def _ai_async_progress(self, msg):
        self.ai_async_progress_msg = self.ai_async_progress_msg + '\n' + msg
        
    def _ai_async_error(self, exception_type, value, tb_obj):
        try:
            self.ai_async_is_errored = True
            value_text = html_to_text(self._safe_to_text(value))
            msg = _('AI Error:\n')
            msg += exception_type.__name__ + ': ' + str(value_text)
            tb = '\n'.join(traceback.format_tb(tb_obj))
            logger.error(_("Uncaught exception: ") + msg + '\n' + tb)
            # Trigger message box show
            qt_exception_hook._exception_caught.emit(msg, tb)
        except Exception as err:
            logger.error(_("Uncaught exception while handling AI error: ") + self._safe_to_text(err))

    def _ai_async_finished(self):
        self.ai_async_is_finished = True
    
    def _ai_async_abort_button_clicked(self):
        self.ai_async_is_canceled = True
    
    def ai_async_stream(self, llm, messages, result_callback=None, progress_callback=None, streaming_callback=None, error_callback=None):       
        """Calls the LLM in a background thread and streams back the results 

        Args:
            llm: can be either self.fast_llm or self.large_llm
            messages: list of AI messages (e.g. a conversation)
            result_callback (optional): Defaults to None.
            progress_callback (optional): Defaults to None.
            streaming_callback (optional): Defaults to None.
            error_callback (optional): Defaults to None.
        """
        # start async worker 
        self.ai_async_is_finished = False
        self.ai_async_is_errored = False
        self.ai_async_progress_msg = ''
        self.ai_async_progress_count = -1
        worker = Worker(self._ai_async_stream, llm=llm, messages=messages)
        if result_callback is not None: 
            worker.signals.result.connect(result_callback)
        if progress_callback is not None:
            worker.signals.progress.connect(progress_callback)
        if streaming_callback is not None:
            worker.signals.streaming.connect(streaming_callback)
        if error_callback is not None:
            worker.signals.error.connect(error_callback)
        else:
            worker.signals.error.connect(self._ai_async_error)
        self.threadpool.start(worker)

    def _ai_async_stream(self, signals, llm, messages):
        self.ai_async_is_canceled = False
        self.ai_streaming_output = ''
        req_id = self.log_llm_request(llm, messages, context='ai_async_stream')
        stream_iter = None
        try:
            stream_iter = llm.stream(messages)
            for chunk in stream_iter:
                if self.ai_async_is_canceled:
                    break  # cancel the streaming
                else:
                    chunk_text = str(getattr(chunk, 'content', ''))
                    self.ai_streaming_output += chunk_text
                    if signals is not None:
                        if signals.streaming is not None:
                            signals.streaming.emit(chunk_text)
                        if signals.progress is not None:
                            self.ai_async_progress_count += len(chunk_text)
                            signals.progress.emit(str(self.ai_async_progress_count))
        except Exception as err:
            self.log_llm_error(req_id, llm, err, context='ai_async_stream')
            # Some providers emit malformed trailing streaming events after content is already complete.
            # Prefer returning the accumulated text instead of failing the whole turn.
            if self.ai_streaming_output != '':
                res = self.ai_streaming_output
                self.ai_streaming_output = ''
                if not self.ai_async_is_canceled:
                    self.log_llm_response(req_id, llm, res, context='ai_async_stream_partial')
                return res
            raise
        finally:
            if stream_iter is not None:
                close_fn = getattr(stream_iter, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception as close_err:
                        self.log_llm_error(req_id, llm, close_err, context='ai_async_stream_close')
        res = self.ai_streaming_output
        self.ai_streaming_output = ''
        if not self.ai_async_is_canceled:
            self.log_llm_response(req_id, llm, res, context='ai_async_stream')
        return res

    def ai_async_query(self, func, result_callback, *args, progress_callback=None, **kwargs):        
        """Calls an AI related function in a background thread

        Args:
            func: the function to be called.  *args, **kwargs are handed over to this function. 
            result_callback: callback function
            progress_callback: callback function for progress/status updates
        """
        self.ai_async_is_canceled = False
        
        # start async worker
        self.ai_async_is_finished = False
        self.ai_async_is_errored = False
        self.ai_async_progress_msg = ''
        self.ai_async_progress_count = -1
        worker = Worker(func, *args, **kwargs)  # Any other args, kwargs are passed to the run function
        if result_callback is not None: 
            worker.signals.result.connect(result_callback)
        worker.signals.finished.connect(self._ai_async_finished)
        if progress_callback is not None:
            worker.signals.progress.connect(progress_callback)
        else:
            worker.signals.progress.connect(self._ai_async_progress)
        worker.signals.error.connect(self._ai_async_error)
        self.threadpool.start(worker)
                        
    def get_curr_language(self):
        """Determine the current language of the UI and/or the project. 
        Used to instruct the AI answering in the correct language. 
        """ 
        if self.app.settings.get('ai_language_ui', 'True') == 'True':
            # get ui language
            lang_long = {"de": "Deutsch", "en": "English", "es": "Español", "fr": "Français", "it": "Italiano", "pt": "Português"}
            lang = lang_long[self.app.settings['language']] 
            if lang is None:
                lang = 'English'
        else:
            lang = self.app.settings.get('ai_language', 'English')
        return lang
    
    def _get_response_format_json_schema(self, schema_name: str, schema: dict):
        """Build a valid OpenAI-compatible response_format payload for JSON Schema mode."""
        if not schema_name or not isinstance(schema, dict) or len(schema) == 0:
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True
            }
        }

    def get_response_format_json_schema(self, schema_name: str, schema: dict):
        """Public helper for building JSON Schema response_format payloads."""
        return self._get_response_format_json_schema(schema_name, schema)
    
    def generate_code_descriptions(self, code_name, code_memo='') -> list:
        """Prompts the AI to create a list of 10 short descriptions of the given code.
        This is used to get a better basis for the semantic search in the vectorstore. 

        Args:
            code_name (str): the name of the code
            code_memo (str): a memo, optional

        Returns:
            list: list of strings
        """

        # example result: 
        json_result = """
{
    "descriptions": [
        "first description",
        "second description"
    ]
}
"""
        # validation schema:
        response_schema = {
            "type": "object",
            "properties": {
                "descriptions": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["descriptions"],
            "additionalProperties": False
        }

        """
        code_descriptions_prompt = [
            SystemMessage(
                content=self.get_default_system_prompt()
            ),
            HumanMessage(
                content=(f'We are searching for empirical data that fits a code named "{code_name}" '
                    f'with the following code memo: "{extract_ai_memo(code_memo)}". \n'
                    'Your task: Give back a list of 10 short descriptions of the meaning of this code. '
                    'Try to give a variety of diverse code-descriptions. Use simple language. '
                    f'Always answer in the following language: "{self.get_curr_language()}". Do not use numbers or bullet points. '
                    'Do not explain anything or repeat the code name, just give back the descriptive text. '
                    'Return the list as a valid JSON object in the following form:\n'
                    f'{json_result}')
            )
        ]
        """

        # revised version 7/25:
        code_descriptions_prompt = [
            SystemMessage(
                content=self.get_default_system_prompt() + '\n /no_think'
            ),
            HumanMessage(
                content=(f'We are searching for empirical data that fits a code named "{code_name}" '
                    f'with the following code memo: "{extract_ai_memo(code_memo)}". \n'
                    'Your task: Give back a list of up to 10 reformulated variants of the code, using '
                    'synonyms or directly related concepts. Also consider the memo, if available. '
                    'Try to give a variety of diverse reformulations. Use simple language.'
                    f'Always answer in the following language: "{self.get_curr_language()}". Do not use numbers or bullet points. '
                    'Do not explain anything or repeat the code name, just give back the list of variants. '
                    'Return the list as a valid JSON object in the following form:\n'
                    f'{json_result}')
            )
        ]

        logger.debug(_('AI generate_code_descriptions\n'))
        logger.debug(_('Prompt:\n') + str(code_descriptions_prompt))
        
        # callback to show percentage done    
        config = RunnableConfig()
        config['callbacks'] = [MyCustomSyncHandler(self)]
        self.ai_async_progress_max = round(1000 / 4)  # estimated token count of the result (1000 chars)

        response_format = self._get_response_format_json_schema("code_descriptions", response_schema)
        res = self.invoke_with_logging(
            self.large_llm,
            code_descriptions_prompt,
            response_format=response_format,
            config=config,
            context='generate_code_descriptions',
            fallback_without_response_format=True,
            fallback_exceptions=(BadRequestError, ValidationError),
        )
        logger.debug(str(res.content))
        res.content = strip_think_blocks(res.content)
        code_descriptions = list(json_repair.loads(str(res.content))['descriptions'])
        code_descriptions.insert(0, code_name) # insert the original as well
        return code_descriptions

    def retrieve_similar_data(self, result_callback, code_name, code_memo='', doc_ids=None):
        """Retrieve pieces of data from the vectorstore that are related to the given code.
        This function will be performed in the background, following these steps:
        1) Semantic extension: Let the LLM generate a list of 10 desciptions of the code in simple language
        2) Semantic search: Search in the vectorstore for data that has a semantic similarity to these descriptions
        3) Ranking: Sort the results for relevance

        Args:
            result_callback: Callback funtion, recieved the results as a list of documents type langchain_core.documents.base.Document
            code_name: str
            code_memo (optional): Defaults to ''.
            doc_ids (list, optional): Filter. If not None, only results from these documents will be returned. Defaults to [].
        """
        self.ai_async_query(self._retrieve_similar_data, result_callback, code_name, code_memo, doc_ids)

    def _retrieve_similar_data(self, code_name, code_memo='', doc_ids=None, progress_callback=None, signals=None) -> list:
        # Get a list of code descriptions from the llm
        if progress_callback is not None:
            progress_callback.emit(_('Stage 1:\nSearching data related to "') + code_name + '"') 
        descriptions = self.generate_code_descriptions(code_name, code_memo)
        if self.ai_async_is_canceled:
            return []
        return self._retrieve_from_vectorstore(descriptions, doc_ids, progress_callback, signals)
    
    def _retrieve_from_vectorstore(self, search_strings, doc_ids=None, progress_callback=None, signals=None,
                                   score_threshold=0.5, k=50) -> list:
        # Use the list of search_strings to retrieve related data from the vectorstore
        try:
            threshold = float(score_threshold)
        except (TypeError, ValueError):
            threshold = 0.5
        threshold = max(0.0, min(threshold, 1.0))

        try:
            top_k = int(k)
        except (TypeError, ValueError):
            top_k = 50
        top_k = max(1, min(top_k, 500))

        search_kwargs = {'score_threshold': threshold, 'k': top_k}
        chunks_meta_list = []
        for _str in search_strings:
            res = self.sources_vectorstore.faiss_db.similarity_search_with_relevance_scores(_str, **search_kwargs)
            if doc_ids is not None and len(doc_ids) > 0:
                # filter results by document ids
                res_filtered = []
                for chunk in res:
                    if chunk[0].metadata['id'] in doc_ids:
                        res_filtered.append(chunk)
                chunks_meta_list.append(res_filtered)
            else: 
                chunks_meta_list.append(res)

        # Consolidate and rank results:
        # Flatten the lists of chunks in chunks_lists and collect all the chunks in a master list.
        # Duplicate chunks are collected only once. The list is sorted by the frequency 
        # of a chunk counted over all lists + the similarity score that faiss returns.
        # This way, frequent and relevant chunks should be sorted to the top
        # (see: "Reciprocal Rank Fusion" (https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf))

        def chunk_unique_str(chunk):
            # helper
            chunk_str = str(chunk.metadata['id']) + ", "
            chunk_str += str(chunk.metadata['start_index']) + ", "
            return chunk_str
            
        # Flatten the lists and count the frequency of each chunk
        chunk_count_list = {}  # contains the chunk count
        chunk_master_list = []  # contains all chunks from all lists but no doubles
        for lst in chunks_meta_list:
            for chunk in lst:            
                chunk_doc = chunk[0]
                chunk_score = chunk[1]
                chunk_str = chunk_unique_str(chunk_doc)
                chunk_in_count_list = chunk_count_list.get(chunk_str, None)
                if chunk_in_count_list: 
                    chunk_count_list[chunk_str] += 1 + chunk_score
                else:
                    chunk_count_list[chunk_str] = 1 + chunk_score
                    chunk_master_list.append(chunk_doc)
                    
        # add scores
        for chunk_doc in chunk_master_list:
            chunk_doc.metadata['score'] = chunk_count_list[chunk_unique_str(chunk_doc)]
        
        # Sort the common items by their score in descending order
        chunk_master_list.sort(key=lambda chunk: chunk.metadata['score'], reverse=True)
                                
        logger.debug('First 10 chunks of retrieved data:\n' + str(chunk_master_list[:10]))
        
        return chunk_master_list
    
    def search_analyze_chunk(self, result_callback, chunk, code_name, code_memo, search_prompt: PromptItem):
        """Letting the AI analyze a chunk of data, using the given prompt

        Args:
            result_callback: Callback function. Revieves the result in as a langchain_core.documents.base.Document with the folowing added fields:
                             - 'quote_start': the start_index of the quote
                             - 'quote': the selected quote
                             - 'interpretation': the interpretation of the LLM
                             If the AI discarded this chunk as not relevant, the returned document will be None 
            chunk: piece of data, type langchain_core.documents.base.Document
            code_name: str
            code_memo: str
            search_prompt (PromptItem): the prompt used for the analysis
        """
        self.ai_async_query(self._search_analyze_chunk, result_callback, chunk, code_name, code_memo, search_prompt)
        
    def _search_analyze_chunk(self, chunk, code_name, code_memo, search_prompt: PromptItem, progress_callback=None, signals=None):                
        if progress_callback is not None:
            progress_callback.emit(_("Stage 2:\nInspecting the data more closely..."))        

        # build up the prompt
        # example result:
        json_result = """
{
    "interpretation": "your brief reasoning",
    "related": true,
    "quote": "selected quote"
}
"""
        # validation schema:
        response_schema = {
            "type": "object",
            "properties": {
                "interpretation": {"type": "string"},
                "related": {"type": "boolean"},
                "quote": {"type": "string"}
            },
            "required": ["interpretation", "related", "quote"],
            "additionalProperties": False
        }

        prompt = [
            SystemMessage(
                content=self.get_default_system_prompt()
            ),
            HumanMessage(
                content= (f'You are discussing the code named "{code_name}" with the following code memo: "{extract_ai_memo(code_memo)}". \n'
                    'At the end of this message, you will find a chunk of empirical data. \n'
                    'Your task is to use the following instructions to analyze the chunk of empirical data and decide wether it relates to the given code or not. '
                    f'Instructions: "{search_prompt.text}". \n'
                    'Summarize your reasoning briefly in the field "interpretation" of the result. '
                    f'In this particular field, always answer in the language "{self.get_curr_language()}".\n'
                    'If you came to the conclusion that the chunk of data '
                    'is not related to the code, give back false in the field "related", otherwise true. '
                    'Use JSON booleans, not strings.\n'
                    'If the previous step resulted in \'True\', identify a quote from the chunk of empirical data that contains the part which is '
                    'relevant for the analysis of the given code. Include enough context so that the quote is comprehensable. '
                    'Give back this quote in the field "quote" exactly like in the original, '
                    'including errors. Do not leave anything out, do not translate the text or change it in any other way. '
                    'If you cannot identify a particular quote, return the whole chunk of empirical data in the field "quote".\n'
                    'If the previous step resulted in \'False\', return an empty quote ("") in the field "quote".\n'
                    f'Make sure to return nothing else but a valid JSON object in the following form: \n{json_result}.'
                    f'\n\nThe chunk of empirical data for you to analyze: \n"{chunk.page_content}"')
                )
            ]

        # callback to show percentage done    
        config = RunnableConfig()
        config['callbacks'] = [MyCustomSyncHandler(self)]
        self.ai_async_progress_max = 130  # estimated average token count of the result
        
        # send the query to the llm 
        response_format = self._get_response_format_json_schema("search_analyze_chunk", response_schema)
        res = self.invoke_with_logging(
            self.large_llm,
            prompt,
            response_format=response_format,
            config=config,
            context='search_analyze_chunk',
            fallback_without_response_format=True,
            fallback_exceptions=(BadRequestError, ValidationError),
        )
        res.content = strip_think_blocks(res.content)
        res_json = json_repair.loads(str(res.content))
        
        # analyse and format the answer
        if 'related' in res_json and res_json['related'] in [True, 'True', 'true'] and \
           'quote' in res_json and res_json['quote'] != '':  # found something
            # Adjust quote_start
            doc = {}
            doc['metadata'] = chunk.metadata          
            quote_start, quote_end = ai_quote_search(res_json['quote'], chunk.page_content)
            if quote_start > -1 < quote_end:
                doc['quote_start'] = quote_start + doc['metadata']['start_index']
                doc['quote'] = chunk.page_content[quote_start:quote_end]
            else:  # quote not found, make the whole chunk the quote
                doc['quote_start'] = doc['metadata']['start_index']
                doc['quote'] = chunk.page_content        
            doc['interpretation'] = res_json['interpretation']
        else:  # No quote means the AI discarded this chunk as not relevant
            doc = None
        return doc   

    
    
