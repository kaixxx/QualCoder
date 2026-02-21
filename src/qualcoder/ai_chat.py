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

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt, QEvent, QObject, pyqtSignal
from PyQt6.QtGui import QCursor, QGuiApplication, QAction, QPalette
from PyQt6.QtWidgets import QTextEdit
import qtawesome as qta

from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents.base import Document

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import html as html_lib
import logging
import traceback
import os
import sqlite3
import webbrowser
import re

from .ai_search_dialog import DialogAiSearch
from .GUI.ui_ai_chat import Ui_Dialog_ai_chat
from .helpers import Message
from .confirm_delete import DialogConfirmDelete
from .ai_prompts import PromptItem
from .ai_llm import extract_ai_memo, ai_quote_search, strip_think_blocks
from .ai_mcp_server import AiMcpServer
from .error_dlg import qt_exception_hook
from .html_parser import html_to_text

path = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

topic_analysis_max_chunks = 30

class AIChatSignalEmitter(QObject):
    newTextChatSignal = pyqtSignal(int, str, str, int, object)  # will start a new text analysis chat

ai_chat_signal_emitter = AIChatSignalEmitter()  # Create a global instance of the signal emitter


class DialogAIChat(QtWidgets.QDialog):
    """ AI chat window
    """    
    app = None
    parent_textEdit = None
    chat_history_conn = None
    current_chat_idx = -1
    current_streaming_chat_idx = -1
    chat_msg_list = [] 
    is_updating_chat_window = False
    ai_semantic_search_chunks = []
    last_export_dir = ''
    # filenames = []

    def __init__(self, app, parent_text_edit: QTextEdit, main_window: QtWidgets.QMainWindow):

        self.app = app
        self.parent_textEdit = parent_text_edit
        self.main_window = main_window
        # Set up the user interface from Designer.
        QtWidgets.QDialog.__init__(self)
        self.ui = Ui_Dialog_ai_chat()
        self.ui.setupUi(self)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        # self.ui.scrollArea_ai_output.verticalScrollBar().rangeChanged.connect(self.ai_output_scroll_to_bottom)
        self.ui.plainTextEdit_question.installEventFilter(self)
        self.ui.pushButton_question.pressed.connect(self.button_question_clicked)
        self.ui.progressBar_ai.setMaximum(100)
        self.ui.plainTextEdit_question.setPlaceholderText(_('<your question>'))
        self.ui.pushButton_new_analysis.clicked.connect(self.button_new_clicked)
        self.ui.pushButton_delete.clicked.connect(self.delete_chat)
        self.ui.pushButton_delete.setShortcut('Delete')
        self.ui.listWidget_chat_list.itemSelectionChanged.connect(self.chat_list_selection_changed)
        # Enable editing of items on double click and when pressing F2
        self.ui.listWidget_chat_list.setEditTriggers(QtWidgets.QListWidget.EditTrigger.DoubleClicked | QtWidgets.QListWidget.EditTrigger.EditKeyPressed)
        self.ui.listWidget_chat_list.itemChanged.connect(self.chat_list_item_changed)
        self.ui.listWidget_chat_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.listWidget_chat_list.customContextMenuRequested.connect(self.open_context_menu)
        self.ui.ai_output.linkHovered.connect(self.on_linkHovered)
        self.ui.ai_output.linkActivated.connect(self.on_linkActivated)
        self.ui.pushButton_help.pressed.connect(self.help)
        ai_chat_signal_emitter.newTextChatSignal.connect(self.new_text_chat)
        self.init_styles()
        self.ai_busy_timer = QtCore.QTimer(self)
        self.ai_busy_timer.timeout.connect(self.update_ai_busy)
        self.ai_busy_timer.start(100)
        self.ai_streaming_output = ''
        self.ai_stream_buffer = ""
        self.ai_stream_in_ref = False
        self.curr_codings = None
        self.ai_mcp_server = AiMcpServer(self.app)
        self.ai_prompt = None
        self.ai_search_code_name = None
        self.ai_search_code_memo = None
        self.chat_list = []
        self.ai_search_file_ids = []
        self.ai_search_code_ids = []
        self.ai_text_doc_id = None
        self.ai_text_doc_name = ''
        self.ai_text_text = ''
        self.ai_text_start_pos = -1
        self.ai_output_autoscroll = True
        self.ui.scrollArea_ai_output.verticalScrollBar().valueChanged.connect(self.on_ai_output_scroll)

    def init_styles(self):
        """Set up the stylesheets for the ui and the chat entries
        """
        self.font = f'font: {self.app.settings["fontsize"]}pt "{self.app.settings["font"]}";'
        self.setStyleSheet(self.font)
        # Set progressBar color to default highlight color
        self.ui.progressBar_ai.setStyleSheet(f"""
            QProgressBar::chunk {{
                background-color: {self.app.highlight_color()};
            }}
        """)
        self.ui.pushButton_help.setIcon(qta.icon('mdi6.help'))
        self.ui.pushButton_help.setFixedHeight(self.ui.pushButton_delete.height())
        self.ui.pushButton_help.setFixedWidth(self.ui.pushButton_help.height())
        doc_font = f'font: {self.app.settings["docfontsize"]}pt \'{self.app.settings["font"]}\';'
        self.ai_response_style = f'"{doc_font} color: #356399;"'
        self.ai_user_style = f'"{doc_font} color: #287368;"'
        self.ai_info_style = f'"{doc_font}"'
        self.ai_status_style = f'"{doc_font} color: #808080;"'
        self.ai_actions_style = f'"{doc_font}"'
        if self.app.settings['stylesheet'] in ['dark', 'rainbow']:
            self.ai_response_style = f'"{doc_font} color: #8FB1D8;"'
            self.ai_user_style = f'"{doc_font} color: #35998A;"'
            self.ai_info_style = f'"{doc_font}"'
            self.ai_status_style = f'"{doc_font} color: #B5B5B5;"'
        elif self.app.settings['stylesheet'] == 'native':
            # Determine whether dark or light native style is active:
            style_hints = QGuiApplication.styleHints()
            # Older versions fot PyQt6 may not have QGuiApplication.styleHints().colorScheme() e.g. PtQ66 vers 6.2.3
            try:
                if style_hints.colorScheme() == QtCore.Qt.ColorScheme.Dark:
                    self.ai_response_style = f'"{doc_font} color: #8FB1D8;"'
                    self.ai_user_style = f'"{doc_font} color: #35998A;"'
                    self.ai_info_style = f'"{doc_font}"'
                    self.ai_status_style = f'"{doc_font} color: #B5B5B5;"'
                else:
                    self.ai_response_style = f'"{doc_font} color: #356399;"'
                    self.ai_user_style = f'"{doc_font} color: #287368;"'
                    self.ai_info_style = f'"{doc_font}"'
                    self.ai_status_style = f'"{doc_font} color: #808080;"'
            except AttributeError as e_:
                print(f"Using older version of PyQT6? {e_}")
                logger.debug(f"Using older version of PyQT6? {e_}")
                pass
        else:
            self.ai_response_style = f'"{doc_font} color: #356399;"'
            self.ai_user_style = f'"{doc_font} color: #287368;"'
            self.ai_info_style = f'"{doc_font}"'
            self.ai_status_style = f'"{doc_font} color: #808080;"'
        self.ui.plainTextEdit_question.setStyleSheet(self.ai_user_style[1:-1])
        default_bg_color = self.ui.plainTextEdit_question.palette().color(self.ui.plainTextEdit_question.viewport().backgroundRole())
        self.ui.ai_output.setStyleSheet(doc_font)
        self.ui.ai_output.setAutoFillBackground(True)
        self.ui.ai_output.setStyleSheet('QWidget:focus {border: none;}')
        self.ui.ai_output.setStyleSheet(f'background-color: {default_bg_color.name()};')
        self.ui.scrollArea_ai_output.setStyleSheet(f'background-color: {default_bg_color.name()};')
        self.update_chat_window()
        
    def init_ai_chat(self, app=None):
        if app is not None:
            self.app = app
            self.ai_mcp_server = AiMcpServer(self.app)
        # init chat history
        self.chat_history_folder = self.app.project_path + '/ai_data'
        if not os.path.exists(self.chat_history_folder):
            os.makedirs(self.chat_history_folder)
        self.chat_history_path = self.chat_history_folder + '/chat_history.sqlite'            
        self.chat_history_conn = sqlite3.connect(self.chat_history_path)
        cursor = self.chat_history_conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS chats (
                                id INTEGER PRIMARY KEY,
                                name TEXT,
                                analysis_type TEXT,
                                summary TEXT,
                                date TEXT,
                                analysis_prompt TEXT)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
                                id INTEGER PRIMARY KEY,
                                chat_id INTEGER,
                                msg_type TEXT,
                                msg_author TEXT,
                                msg_content TEXT,
                                FOREIGN KEY (chat_id) REFERENCES chats(id))''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS topic_chat_embeddings (
                                id INTEGER PRIMARY KEY,
                                chat_id INTEGER,
                                docstore_id TEXT,
                                position INTEGER,
                                used_flag INTEGER,
                                FOREIGN KEY (chat_id) REFERENCES chats(id))''')
        self.chat_history_conn.commit()
        self.current_chat_idx = -1
        self.fill_chat_list()
    
    def close(self):
        if self.chat_history_conn is not None:
            self.chat_history_conn.close()
            
    def help(self):
        """ Open help in browser. """
        self.app.help_wiki("5.1.-AI-chat-based-analysis")

    def get_chat_list(self):
        """Load the current chat list from the database into self.chat_list
        """
        cursor = self.chat_history_conn.cursor()
        cursor.execute('SELECT id, name, analysis_type, summary, date, analysis_prompt FROM chats ORDER BY date DESC')
        self.chat_list = cursor.fetchall()
        if self.current_chat_idx >= len(self.chat_list):
            self.current_chat_idx = len(self.chat_list) - 1    
            
    def fill_chat_list(self):
        self.ui.listWidget_chat_list.clear()
        self.get_chat_list()
        for i in range(len(self.chat_list)):
            chat = self.chat_list[i]
            id_, name, analysis_type, summary, date, analysis_prompt = chat
            if analysis_type != 'general chat':
                tooltip_text = f"{name}\nType: {analysis_type}\nSummary: {summary}\nDate: {date}\nPrompt: {analysis_prompt}"
            else:
                tooltip_text = f"{name}\nType: {analysis_type}\nSummary: {summary}\nDate: {date}"

            # Creating a new QListWidgetItem
            if analysis_type == 'general chat':
                icon = self.app.ai.general_chat_icon()
            elif analysis_type == 'topic chat':
                icon = self.app.ai.topic_analysis_icon()
            elif analysis_type == 'text chat':
                icon = self.app.ai.text_analysis_icon()
            elif analysis_type == 'code chat':
                icon = self.app.ai.code_analysis_icon()

            item = QtWidgets.QListWidgetItem(icon, name)
            item.setToolTip(tooltip_text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            
            # Adding the item to the QListWidget
            self.ui.listWidget_chat_list.addItem(item)
            #if i == self.current_chat_idx:
            #    item.setSelected(True)
        if self.current_chat_idx >= len(self.chat_list):
            self.current_chat_idx = len(self.chat_list) - 1
        self.ui.listWidget_chat_list.setCurrentRow(self.current_chat_idx)
        self.chat_list_selection_changed(force_update=True)

    def new_chat(self, name, analysis_type, summary, analysis_prompt):
        date = datetime.now()
        date_text = date.strftime('%Y-%m-%d %H:%M:%S')
        cursor = self.chat_history_conn.cursor()
        cursor.execute('''INSERT INTO chats (name, analysis_type, summary, date, analysis_prompt)
                            VALUES (?, ?, ?, ?, ?)''', (name, analysis_type, summary, date_text, analysis_prompt))
        self.chat_history_conn.commit()
        self.current_chat_idx = -1
        self.fill_chat_list()
        # select new chat
        self.current_chat_idx = self.find_chat_idx(cursor.lastrowid)
        self.ui.listWidget_chat_list.setCurrentRow(self.current_chat_idx)
        self.ai_output_autoscroll = True
        self.chat_list_selection_changed()

    def new_general_chat(self, name, summary):
        if self.app.project_name == "":
            msg = _('No project open.')
            Message(self.app, _('AI not enabled'), msg, "warning").exec()
            return
        if self.app.settings['ai_enable'] != 'True':
            msg = _('The AI is disabled. Go to "AI > Setup Wizard" first.')
            Message(self.app, _('AI not enabled'), msg, "warning").exec()
            return

        self.new_chat(name, 'general chat', summary, '')
        system_prompt = self._general_chat_base_system_prompt()
        self.process_message('system', system_prompt)    
        self.update_chat_window()  

    def _agent_md_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "ai_skills", "agent.md")

    def _load_agent_md_content(self) -> str:
        """Load global agent instructions from agent.md if present."""

        agent_md_path = self._agent_md_path()
        if not os.path.exists(agent_md_path):
            return ""
        try:
            with open(agent_md_path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError:
            return ""

    def _general_chat_base_system_prompt(self) -> str:
        """Build the base system prompt for general chat from agent.md + project memo."""

        base_prompt = self._load_agent_md_content()
        if base_prompt == "":
            return self.app.ai.get_default_system_prompt()

        project_memo = extract_ai_memo(self.app.get_project_memo())
        if self.app.settings.get('ai_send_project_memo', 'True') == 'True' and len(project_memo) > 0:
            base_prompt += '\n\n# Information about the current project\n\n'
            base_prompt += 'Here is some background information about the research project the team is working on:\n'
            base_prompt += project_memo
        return base_prompt

    def _build_mcp_combined_system_prompt(self, phase_prompt: str) -> str:
        """Combine global agent instructions with phase-specific technical instructions."""

        base_prompt = self._general_chat_base_system_prompt().strip()
        phase_text = str(phase_prompt if phase_prompt is not None else "").strip()
        if base_prompt == "":
            return phase_text
        if phase_text == "":
            return base_prompt
        return base_prompt + "\n\n# Current task contract\n\n" + phase_text

    def new_text_analysis(self):
        """analyze a piece of text from an empirical document"""
        if self.app.project_name == "":
            msg = _('No project open.')
            Message(self.app, _('AI not enabled'), msg, "warning").exec()
            return
        if self.app.settings['ai_enable'] != 'True':
            msg = _('The AI is disabled. Go to "AI > Setup Wizard" first.')
            Message(self.app, _('AI not enabled'), msg, "warning").exec()
            return

        msg = _('We will now switch to the text coding workspace.\n There you can open a document, select a piece of text, right click on it and choose "AI Text Analysis" from the context menu.')
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setStyleSheet("* {font-size:" + str(self.app.settings['fontsize']) + "pt} ")
        reply = msg_box.question(
            self,
            _('AI Text Analysis'),
            msg,
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok  # <--- Default button
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Ok:
            self.main_window.text_coding(task='documents')
        else:
            return

    def new_code_chat(self):
        """chat about codings"""
        if self.app.project_name == "":
            msg = _('No project open.')
            Message(self.app, _('AI not enabled'), msg, "warning").exec()
            return
        if self.app.settings['ai_enable'] != 'True':
            msg = _('The AI is disabled. Go to "AI > Setup Wizard" first.')
            Message(self.app, _('AI not enabled'), msg, "warning").exec()
            return
       
        ui = DialogAiSearch(self.app, 'code_analysis')
        ret = ui.exec()
        if ret == QtWidgets.QDialog.DialogCode.Accepted:
            self.ai_text_doc_id = None
            self.ai_search_code_name = ui.selected_code_name
            self.ai_search_code_memo = ui.selected_code_memo
            self.ai_search_file_ids = ui.selected_file_ids
            self.ai_search_code_ids = ui.selected_code_ids
            self.ai_search_coder_names = ui.coder_names
            self.ai_prompt = ui.current_prompt
            # fetch data
            # This SQL sorts the results by file id, but not like 1, 1, 1, 2, 2, 3... 
            # Instead, the results are mixed up in this order: file id = 1, 2, 3, 1, 2, 1...
            # This tries to ensure that even if the data send to the AI must be cut off at some point 
            # because of the token limit, there will at least be data from as many different files as 
            # possible included in the analysis.
            # The JOIN also adds the source.name so that the AI can refer to a certain document
            # by its name.     

            code_ids = list(self.ai_search_code_ids or [])
            file_ids = list(self.ai_search_file_ids or [])
            coder_names = list(self.ai_search_coder_names or [])

            code_ph  = ",".join(["?"] * len(code_ids))
            file_ph  = ",".join(["?"] * len(file_ids))
            owner_ph = ",".join(["?"] * len(coder_names))

            sql = f"""
                SELECT ordered.*, source.name, code_name.name AS code_name
                FROM (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY fid ORDER BY ctid) AS rn
                    FROM code_text
                    WHERE cid IN ({code_ph})
                    AND fid IN ({file_ph})
                    AND owner IN ({owner_ph})
                ) AS ordered
                JOIN source ON ordered.fid = source.id
                JOIN code_name ON ordered.cid = code_name.cid
                ORDER BY ordered.rn, ordered.fid;
            """

            params = [*code_ids, *file_ids, *coder_names]

            cursor = self.app.conn.cursor()
            cursor.execute(sql, params)
            self.curr_codings = cursor.fetchall()
            
            if len(self.curr_codings) == 0:
                msg = _('No codings found for this particuar combination of coder, document filter, and code.')
                Message(self.app, _('Code analysis'), msg, 'warning').exec()
                return
            
            ai_data = []
            # Limit the amount of data (characters) send to the ai, so the maximum context window is not exceeded.
            # As a rough estimation, one token is about 4 characters long (in english). 
            # We want to fill not more than half the context window with our data, so that there is enough
            # room for the answer and further chats.
            max_ai_data_length = round(0.5 * (self.app.ai.large_llm_context_window * 4)) 
            max_ai_data_length_reached = False
            ai_data_length = 0
            for row in self.curr_codings:
                if ai_data_length >= max_ai_data_length:
                    max_ai_data_length_reached = True
                    break
                
                fulltext = self.app.get_text_fulltext(row[2])
                line_start, line_end = self.app.get_line_numbers(fulltext, row[4], row[5])
                ai_data.append({
                    'source_id': row[0],
                    'source_name': row[12],
                    'quote': row[3],
                    'line_start': line_start,
                    'line_end': line_end,
                    'code_name': row[13]
                })
                ai_data_length = ai_data_length + len(row[3])
            if len(ai_data) == 0:
                msg = _('No coded text found. Please select another code or category, or refine you filters.')
                Message(self.app, _('AI code analysis'), msg, "warning").exec()
                return    
            ai_data_json = json.dumps(ai_data)
            
            ai_instruction = (
                f'You are discussing the code or category named "{self.ai_search_code_name}" with the following code memo: "{self.ai_search_code_memo}". \n'
                f'Here is a list of quotes from the empirical data that have been coded with the given code or with subcodes under the given category:\n'
                f'{ai_data_json}\n'
                f'Your task is to analyze the given empirical data following these instructions: {self.ai_prompt.text}\n'
                f'The whole discussion should be based upon the the empirical data provided and its proper interpretation. '
                f'Do not make any assumptions which are not supported by the data '
                f'Please mention the sources that your refer to from the given empirical data, using an html anchor tag of the following form: '
                '<a href="coding:{source_id}">{source_name}: {line_start} - {line_end}</a>\n' 
                f'Always answer in the following language: "{self.app.ai.get_curr_language()}".'
            )    
            
            summary = _('Analyzing the data coded as "{}" ({} pieces of data sent to the AI.)').format(self.ai_search_code_name, len(ai_data))
            if max_ai_data_length_reached:
                summary += _('\nATTENTION: There was more coded data found, but it had to be truncated because of the limited context window of the AI.')
            logger.debug(f'New code chat. Prompt:\n{ai_instruction}')
            self.new_chat(_('Code') + f' "{self.ai_search_code_name}"', 'code chat', summary, self.ai_prompt.name_and_scope())
            # warn if project memo empty 
            project_memo = extract_ai_memo(self.app.get_project_memo())
            if self.app.settings.get('ai_send_project_memo', 'True') == 'True' and len(project_memo) == 0:
                msg = _('Note that it is highly recommended to use the project memo (Menu "Project > Project Memo") \
to include a short description of your project\'s research topics, questions, objectives, and the empirical \
data collected. This information will accompany every prompt sent to the AI, resulting in much more targeted results.')
                self.process_message('info', msg)
            # start analysis
            self.process_message('system', self.app.ai.get_default_system_prompt())
            self.process_message('instruct', ai_instruction)
            self.update_chat_window()  
 
    def new_topic_chat(self):
        """chat about a free topic in the data"""
        if self.app.project_name == "":
            msg = _('No project open.')
            Message(self.app, _('AI not enabled'), msg, "warning").exec()
            return
        if self.app.settings['ai_enable'] != 'True':
            msg = _('The AI is disabled. Go to "AI > Setup Wizard" first.')
            Message(self.app, _('AI not enabled'), msg, "warning").exec()
            return
       
        ui = DialogAiSearch(self.app, 'topic_analysis')
        ret = ui.exec()
        if ret == QtWidgets.QDialog.DialogCode.Accepted:
            self.ai_text_doc_id = None
            self.ai_search_code_name = ui.selected_code_name
            self.ai_search_code_memo = ui.selected_code_memo
            
            self.ai_search_file_ids = ui.selected_file_ids
            self.ai_prompt = ui.current_prompt
            # self.filenames = self.app.get_filenames()
            
            summary = _('Analyzing the free topic "{}" in the data.').format(self.ai_search_code_name)
            if self.ai_search_code_memo != '':
                summary += _('\nDescription:') + f' {self.ai_search_code_memo}'
            logger.debug(f'New topic chat.')
            self.new_chat(_('Topic') + f' "{self.ai_search_code_name}"', 'topic chat', summary, self.ai_prompt.name_and_scope())
            # warn if project memo empty 
            project_memo = extract_ai_memo(self.app.get_project_memo())
            if self.app.settings.get('ai_send_project_memo', 'True') == 'True' and len(project_memo) == 0:
                msg = _('Note that it is highly recommended to use the project memo (Menu "Project > Project Memo") \
to include a short description of your project\'s research topics, questions, objectives, and the empirical \
data collected. This information will accompany every prompt sent to the AI, resulting in much more targeted results.')
                self.process_message('info', msg)
            # start analysis
            self.process_message('system', self.app.ai.get_default_system_prompt())
            self.process_message('info', _('Searching for related data...'))
            self.update_chat_window()  

            self.app.ai.retrieve_similar_data(self.new_topic_chat_callback,  
                                            self.ai_search_code_name, self.ai_search_code_memo,
                                            self.ai_search_file_ids)

    def get_filename(self, id_) -> str:
        """Return the filename for a source id
        Args:
            id_: source id
        Returns:
            str: name | '' if nothing found
        """
        # This might be called from a different thread (ai asynch operations), so have to create a new database connection
        conn = sqlite3.connect(os.path.join(self.app.project_path, 'data.qda'))
        cur = conn.cursor()
        cur.execute(f'select name from source where id = {id_}')
        res = cur.fetchone()[0]
        if res is not None:
            return res
        else:
            return ''

    def new_topic_chat_callback(self, chunks: List[Document]):
        # Analyze the data found
        if self.app.ai.ai_async_is_canceled:
            self.process_message('info', _('Chat has been canceled by the user.'))
            self.update_chat_window()  
            return
        if chunks is None or len(chunks) == 0:
            msg = _('Sorry, the AI could could not find any data related to "') + self.ai_search_code_name + '".'
            self.process_message('info', msg)
            self.update_chat_window()  
            return
        
        self.ai_semantic_search_chunks = chunks                
        msg = _('Found related data. Analyzing the most relevant segments closer.')
        self.process_message('info', msg)
        self.update_chat_window()
        
        # store the found chunks in the table "topic_chat_embeddings" for later
        cursor = self.chat_history_conn.cursor()
        chat_id = int(self.chat_list[self.current_chat_idx][0])
        for i in range(len(chunks)):
            cursor.execute('''
                INSERT INTO topic_chat_embeddings (chat_id, docstore_id, position, used_flag)
                VALUES (?, ?, ?, ?)
            ''', (chat_id, chunks[i].id, i, (1 if i < topic_analysis_max_chunks else 0)))
        self.chat_history_conn.commit()                                

        ai_data = []
        max_ai_data_length = round(0.5 * (self.app.ai.large_llm_context_window * 4)) 
        max_ai_data_length_reached = False  # TODO varaible not used
        ai_data_length = 0
        for i in range(0, topic_analysis_max_chunks):
            if i >= len(chunks): 
                break
            if ai_data_length >= max_ai_data_length:
                max_ai_data_length_reached = True  # TODO variable not used
                break
            chunk = chunks[i]
            fulltext = self.app.get_text_fulltext(chunk.metadata["id"])
            line_start, line_end = self.app.get_line_numbers(fulltext, 
                                                             chunk.metadata["start_index"], 
                                                             chunk.metadata["start_index"] + len(chunk.page_content))
            ai_data.append({
                'source_id': f'{chunk.metadata["id"]}_{chunk.metadata["start_index"]}_{len(chunk.page_content)}_{line_start}_{line_end}',
                'source_name': self.get_filename(int(chunk.metadata['id'])),
                'quote': chunk.page_content,
                'line_start': line_start,
                'line_end': line_end
            })
            ai_data_length += len(chunk.page_content)
        
        ai_data_json = json.dumps(ai_data)
            
        ai_instruction = (
            f'You are analyzing the topic "{self.ai_search_code_name}" with the following description: "{self.ai_search_code_memo}". \n'
            f'A semantic search in the empirical data resulted in the the following list of chunks of empirical data which might be relevant '
            f'for the analysis of the given topic:\n'   
            f'{ai_data_json}\n'
            f'Your task is to analyze the given empirical data following these instructions: {self.ai_prompt.text}\n'
            f'The whole discussion should be based updon the the empirical data provided and its proper interpretation. '
            f'Do not make any assumptions which are not supported by the data. '
            f'Please mention the sources that your refer to from the given empirical data, using an html anchor tag of the following form: '
            '(<a href="chunk:{source_id}">{source_name}: {line_start} - {line_end}</a>)\n' 
            f'Always answer in the following language: "{self.app.ai.get_curr_language()}".'
        )    
        logger.debug(f'Topic chat prompt:\n{ai_instruction}')
        self.process_message('instruct', ai_instruction)
        self.update_chat_window()   
                
    def topic_chat_get_actions(self) -> List[str]:
        # Analyze more data found in the semantic search
        cursor = self.chat_history_conn.cursor()
        chat_id = int(self.chat_list[self.current_chat_idx][0])
        cursor.execute(f'''
            SELECT id, docstore_id
            FROM topic_chat_embeddings
            WHERE chat_id = {chat_id} AND used_flag = 0
            ORDER BY position
            LIMIT 1
        ''')
        if cursor.fetchone() is None: # no data left
            return []
        
        msg = '<a href="action:topic_chat_analyze_more">' + _('Analyze more data...') + '</a>'
        return [msg]
            
    def topic_chat_analyze_more(self): 
        # Analyze more data found in the semantic search
        self.ai_output_autoscroll = True
        cursor = self.chat_history_conn.cursor()
        chat_id = int(self.chat_list[self.current_chat_idx][0])
        cursor.execute(f'''
            SELECT id, docstore_id
            FROM topic_chat_embeddings
            WHERE chat_id = {chat_id} AND used_flag = 0
            ORDER BY position
            LIMIT 30
        ''')
        res = cursor.fetchall()
        
        if res and len(res) > 0:
            topic_chat_embeddings_ids = [row[0] for row in res]
            docstore_ids = [row[1] for row in res]
            chunks = self.app.ai.sources_vectorstore.faiss_db_retrieve_documents(docstore_ids)  
        else:
            chunks = None   
        
        if chunks is None or len(chunks) == 0:
            msg = _('Error: There is no more data to analyze.')
            self.process_message('info', msg)
            self.update_chat_window()  
            return
        
        msg = _('Expanding the analysis with more data.')
        self.process_message('info', msg)
        self.update_chat_window()  
                        
        # self.ai_semantic_search_chunks = chunks
        ai_data = []
        max_ai_data_length = round(0.5 * (self.app.ai.large_llm_context_window * 4)) 
        max_ai_data_length_reached = False  # TODO varaible not used
        ai_data_length = 0
        for i in range(0, len(chunks)):
            if ai_data_length >= max_ai_data_length:
                max_ai_data_length_reached = True  # TODO variable not used
                break
            chunk = chunks[i]
            fulltext = self.app.get_text_fulltext(chunk.metadata["id"])
            line_start, line_end = self.app.get_line_numbers(fulltext, 
                                                             chunk.metadata["start_index"], 
                                                             chunk.metadata["start_index"] + len(chunk.page_content))
            ai_data.append({
                'source_id': f'{chunk.metadata["id"]}_{chunk.metadata["start_index"]}_{len(chunk.page_content)}_{line_start}_{line_end}',
                'source_name': self.get_filename(int(chunk.metadata['id'])),
                'quote': chunk.page_content,
                'line_start': line_start,
                'line_end': line_end
            })
            ai_data_length += len(chunk.page_content)
        
        ai_data_json = json.dumps(ai_data)
            
        ai_instruction = (
            f'Here are more chunks of empirical data from the semantic search described at the beginning '
            'of this conversation: \n'
            f'{ai_data_json}\n\n'
            f'Considering this data, are there any important aspects we must add to the analysis above '
            f'or do we need to revise our conclusions? Make sure to not digress. Ignore any data that is '
            f'not related to the topic of this analysis. Keep your answer short. '
            f'(Do not refer to these instructions in your answer, as they are not visible to the user.)'
        )    
        
        logger.debug(f'Topic chat analyze more prompt:\n{ai_instruction}')
        self.process_message('instruct', ai_instruction)
        self.update_chat_window()
        
        # mark all newly analyzed chunks of data as 'used':
        placeholders = ','.join(['?'] * len(topic_chat_embeddings_ids))
        query = f'''
            UPDATE topic_chat_embeddings
            SET used_flag = 1
            WHERE id IN ({placeholders})
        '''
        cursor.execute(query, topic_chat_embeddings_ids)
        self.chat_history_conn.commit()           
        
    def new_text_chat(self, doc_id, doc_name, text, start_pos, prompt: PromptItem):
        """Analyze a text passage from an empirical document
        """
        if self.app.project_name == "":
            msg = _('No project open.')
            Message(self.app, _('AI not enabled'), msg, "warning").exec()
            return
        if self.app.settings['ai_enable'] != 'True':
            msg = _('The AI is disabled. Go to "AI > Setup Wizard" first.')
            Message(self.app, _('AI not enabled'), msg, "warning").exec()
            return
        # Limit the amount of data (characters) send to the ai, so the maximum context window is not exceeded.
        # As a rough estimation, one token is about 4 characters long (in english). 
        # We want to fill not more than half the context window with our data, so that there is enough
        # room for the answer and further chats.
        max_ai_data_length = round(0.5 * (self.app.ai.large_llm_context_window * 4)) 
        if len(text) > max_ai_data_length:
            msg = _('The text is too long to be analyzed in one go. Please select a shorter passage.')
            Message(self.app, _('AI text analysis'), msg, "warning").exec()
            return
        
        self.main_window.ai_go_chat()  # show chat dialog
        
        self.ai_prompt = prompt
        self.ai_text_doc_id = doc_id
        self.ai_text_doc_name = doc_name
        self.ai_text_text = text
        self.ai_text_start_pos = start_pos
        
        ai_instruction = (
            f'At the end of this message, you will find a passage of text extracted from the empirical ' 
            f'document named "{doc_name}".\n'
            f'Your task is to analyze this text based on the following instructions: \n'
            f'"{prompt.text}"\n\n'
            f'Always answer in the following language: "{self.app.ai.get_curr_language()}".\n'
            f'Be sure to include references to the original data, using this format '
            'definition: `[REF: "{The text from the original data that you want to reference. '
            'I have to match this against the original, so it is very important that you don\'t '
            'change the quoted text in any way. Do not translate or correct errors. Create a '
            'new reference for every single quote.}"]`. \n'
            'These references are invisible text. If you want a direct quote to be '
            'visible to the user, include it in the normal text and add an additional reference '
            'in the above format.\n'
            f'This is the text from the empirical document:\n'
            f'-- BEGIN EMPIRICAL DATA --'
            f'"{text}"'
        )    
        
        summary = _('Analyzing text from ') + \
                  f'<a href="quote:{doc_id}_{start_pos}_{len(text)}">{doc_name}</a> (' + \
                  str(len(text)) + _(' characters).')
        logger.debug(f'New text analysis chat. Prompt:\n{ai_instruction}')
        self.new_chat(_('Text analysis') + f' "{doc_name}"', 'text chat', summary, prompt.name_and_scope())
        self.process_message('system', self.app.ai.get_default_system_prompt())
        self.process_message('instruct', ai_instruction)
        self.update_chat_window()  
        
    def delete_chat(self):
        """Deletes the currently selected chat, connected to the button
           'pushButton_delete'
        """
        if self.current_chat_idx <= -1:
            return
        chat_id = int(self.chat_list[self.current_chat_idx][0])
        chat_name = self.chat_list[self.current_chat_idx][1]
        msg = _('Do you really want to delete ') + '"' + chat_name + '"?'
        ui = DialogConfirmDelete(self.app, msg, _('Delete Chat'))
        ok = ui.exec()
        if not ok:
            return
        cursor = self.chat_history_conn.cursor()
        try:
            cursor.execute('DELETE from chat_messages WHERE chat_id = ?', (chat_id,))
            cursor.execute('DELETE from chats WHERE id = ?', (chat_id,))
            self.chat_history_conn.commit()
        except Exception as e_:
            print(e_)
            self.chat_history_conn.rollback()
            raise
        self.fill_chat_list()

    def find_chat_idx(self, chat_id) -> int | None:
        """Returns the index of the chat with the id 'chat_id' in self.chat_list
        """
        if chat_id is None:
            return None 
        for i in range(len(self.chat_list)):
            if self.chat_list[i][0] == chat_id:
                return i
        return None    
    
    def update_ai_busy(self):
        """update question button + progress bar"""
        if self.app.ai is None or not self.app.ai.is_busy():
            self.ui.pushButton_question.setIcon(qta.icon('mdi6.message-fast-outline', color=self.app.highlight_color()))
            self.ui.pushButton_question.setToolTip(_('Send your question to the AI'))
            self.ui.progressBar_ai.setRange(0, 100)  # Stops the animation
        else:
            if self.ui.progressBar_ai.maximum() > 0: 
                spin_icon = qta.icon("mdi.loading", color=self.app.highlight_color(), animation=qta.Spin(self.ui.pushButton_question))
                self.ui.pushButton_question.setIcon(spin_icon)
                self.ui.pushButton_question.setToolTip(_('Cancel AI generation'))
                self.ui.progressBar_ai.setRange(0, 0)  # Starts the animation
        # update ai status in the statusBar of the main window
        if self.app.ai is not None:
            if self.app.ai.get_status() == 'reading data' and self.app.ai.sources_vectorstore.reading_doc != '':
                self.main_window.statusBar().showMessage(_('AI: ') + _('reading data') + ' (' + self.app.ai.sources_vectorstore.reading_doc + ')')
            else:
                self.main_window.statusBar().showMessage(_('AI: ') + _(self.app.ai.get_status()))
        else: 
            self.main_window.statusBar().showMessage('')

    def on_ai_output_scroll(self, value):
        """Normally, if the AI is generating text, the scrollArea_ai_output scrolls to the bottom
        automatically so that the new text becomes visible. 
        This function ensures the if the user scroll up during the text generation, the auto
        scrolling stops. 
        If the user scroll back down the the end, the auto scrolling is re-enabled. 

        Args:
            value (int): current scroll position
        """
        max_value = self.ui.scrollArea_ai_output.verticalScrollBar().maximum()
        if value >= max_value:
            self.ai_output_autoscroll = True
        else:
            self.ai_output_autoscroll = False

    def update_chat_window(self, scroll_to_bottom=True):
        """load current chat into self.ai_output"""
        if self.current_chat_idx > -1:
            self.is_updating_chat_window = True
            try:
                html = ''
                self.ui.plainTextEdit_question.setEnabled(True)
                self.ui.pushButton_question.setEnabled(True)
                chat = self.chat_list[self.current_chat_idx]
                id_, name, analysis_type, summary, date, analysis_prompt = chat
                if analysis_type == 'text chat':
                    # Extract doc info from the summary field:
                    doc_info_pattern = r'<a href="quote:(\d+)_(\d+)_(\d+)">(.+?)</a>'
                    m = re.search(doc_info_pattern, summary)
                    if m:
                        try:
                            self.ai_text_doc_id = int(m.group(1))
                            self.ai_text_start_pos = int(m.group(2))
                            len_text = int(m.group(3))
                            cursor = self.app.conn.cursor()
                            sql = f'SELECT name, fulltext FROM source WHERE id = {self.ai_text_doc_id}'
                            cursor.execute(sql)
                            source = cursor.fetchone()
                            self.ai_text_doc_name = source[0]
                            self.ai_text_text = source[1][self.ai_text_start_pos:self.ai_text_start_pos + len_text] 
                        except:
                            self.ai_text_doc_id = None
                            self.ai_text_start_pos = None
                            self.ai_text_doc_name = None
                            self.ai_text_text = ''
                    else:
                        self.ai_text_doc_id = None
                        self.ai_text_start_pos = None
                        self.ai_text_doc_name = None   
                        self.ai_text_text = ''                   
                            
                self.ui.ai_output.setText('')  # Clear chat window
                # Show title
                html += f'<h1 style={self.ai_info_style}>{name}</h1>'
                summary_br = summary.replace('\n', '<br />')
                if analysis_type != 'general chat':
                    html += (f"<p style={self.ai_info_style}><b>{_('Type:')}</b> {analysis_type}<br /><b>{_('Summary:')}</b> {summary_br}<br /><b>{_('Date:')}</b> {date}<br /><b>{_('Prompt:')}</b> {analysis_prompt}<br /></p>")
                else:
                    html += (f"<p style={self.ai_info_style}><b>{_('Type:')}</b> {analysis_type}<br /><b>{_('Summary:')}</b> {summary_br}<br /><b>{_('Date:')}</b> {date}<br /></p>")
                # Show chat messages:
                agent_status_lines = []
                agent_status_author = ''

                def flush_agent_status_block():
                    nonlocal html, agent_status_lines, agent_status_author
                    if len(agent_status_lines) == 0:
                        return
                    author = agent_status_author if agent_status_author != '' else 'unknown'
                    body = '<br />'.join(agent_status_lines)
                    block = f'<b>{_("AI")} ({author}) {_("Agent")}:</b><br />{body}'
                    html += f'<p style={self.ai_status_style}>{block}</p>'
                    agent_status_lines = []
                    agent_status_author = ''

                for msg in self.chat_msg_list:
                    msg_type = str(msg[2])
                    if msg_type == 'agent_status':
                        status_line = html_lib.escape(str(msg[4] if msg[4] is not None else '')).replace('\n', '<br />')
                        if status_line.strip() == '':
                            continue
                        status_author = str(msg[3] if msg[3] is not None else '').strip()
                        if status_author == '':
                            status_author = 'unknown'
                        if len(agent_status_lines) > 0 and status_author != agent_status_author:
                            flush_agent_status_block()
                        if agent_status_author == '':
                            agent_status_author = status_author
                        agent_status_lines.append(status_line)
                        continue

                    # Only visible non-status message types flush the buffered status block.
                    if msg_type in ('user', 'ai', 'info'):
                        flush_agent_status_block()

                    if msg_type == 'user':
                        txt = msg[4].replace('\n', '<br />')
                        author = msg[3]
                        if author is None or author == '':
                            author = 'unkown'
                        txt = f'<b>{_("User")} ({author}):</b><br />{txt}'
                        html += f'<p style={self.ai_user_style}>{txt}</p>'
                    elif msg_type == 'ai':
                        txt = msg[4]
                        txt = txt.replace('\n', '<br />')
                        author = msg[3]
                        if author is None or author == '':
                            author = 'unkown'
                        txt = f'<b>{_("AI")} ({author}):</b><br />{txt}'                        
                        html += f'<p style={self.ai_response_style}>{txt}</p>'
                    elif msg_type == 'info':
                        txt = msg[4].replace('\n', '<br />')
                        txt = '<b>' + _('Info:') + '</b><br />' + txt
                        html += f'<p style={self.ai_info_style}>{txt}</p>'
                flush_agent_status_block()
                # add partially streamed ai response if needed
                if len(self.app.ai.ai_streaming_output) > 0:
                    txt = self.app.ai.ai_streaming_output
                    txt = strip_think_blocks(txt)
                    if len(self.app.ai.ai_streaming_output) != len(txt) and len(txt) == 0:
                        txt = _('Thinking...')
                    txt = self.replace_references(txt, streaming=True)
                    txt = txt.replace('\n', '<br />')
                    author = self.app.ai_models[int(self.app.settings['ai_model_index'])]['name']
                    if author is None or author == '':
                        author = 'unkown'
                    txt = f'<b>AI ({author}):</b><br />{txt}'                        
                    html += f'<p style={self.ai_response_style}>{txt}</p>'
                elif not self.app.ai.is_busy(): # streaming finished, add actions
                    actions_list = []
                    if analysis_type == 'topic chat':
                        actions_list.extend(self.topic_chat_get_actions())                        
                    if len(actions_list):
                        # html += f'<p style={self.ai_actions_style}>&nbsp;</p>'
                        button_color = self.ui.pushButton_question.palette().color(QPalette.ColorRole.Button).name()
                        actions_html = '<table border="0" cellspacing="3" cellpadding="10"><tr>'
                        for action in actions_list:
                            actions_html += f'<td style="background-color: {button_color}">{action}</td>'
                        actions_html += '</tr></table>' 
                        html += f'<p style={self.ai_actions_style}>{actions_html}</p>'
                self.ui.ai_output.setText(html)
            finally:
                if scroll_to_bottom:
                    self.ai_output_scroll_to_bottom()
                    self.ui.plainTextEdit_question.setFocus()
                else:
                    self.ui.scrollArea_ai_output.verticalScrollBar().setValue(0)
                self.is_updating_chat_window = False
        else:
            self.ui.ai_output.setText('')
            self.ui.plainTextEdit_question.setEnabled(False)
            self.ui.pushButton_question.setEnabled(False)
            
    def _replace_text_references(self, text, streaming=False):
        if self.ai_text_doc_id is None: 
            # we are not in text analysis chat
            return text
                
        pattern = r'\[REF: ([\"\'“”„‘’«»])(.+?)([\"\'“”„‘’«»])\]'
        fulltext = self.app.get_text_fulltext(self.ai_text_doc_id)    
        
        # Replacement function
        def replace_match(match):
            if streaming:
                return f'({self.ai_text_doc_name})'
            quote = match.group(2)
            
            quote_start, quote_end = ai_quote_search(quote, self.ai_text_text)
            if quote_start > -1 < quote_end:
                quote = self.ai_text_text[quote_start:quote_end]
                quote_start += self.ai_text_start_pos
                quote_end += self.ai_text_start_pos
                line_start, line_end = self.app.get_line_numbers(fulltext, quote_start, quote_end)
                if line_start + line_end > 0:
                    if line_start == line_end:  # one line
                        a = f'(<a href="quote:{self.ai_text_doc_id}_{quote_start}_{len(quote)}">{self.ai_text_doc_name}: {line_start}</a>)'
                    else:  # multiple lines
                        a = f'(<a href="quote:{self.ai_text_doc_id}_{quote_start}_{len(quote)}">{self.ai_text_doc_name}: {line_start} - {line_end}</a>)'
                else:  # no lines found
                    a = f'(<a href="quote:{self.ai_text_doc_id}_{quote_start}_{len(quote)}">{self.ai_text_doc_name}</a>)'
                return a
            else:  # not found
                return _('(unknown reference)')
            
        # Use re.sub with replacement function
        res = re.sub(pattern, replace_match, text)
        
        # If streaming, delete incomplete references at the end
        if streaming:
            incomplete = re.search(r'\[REF:[^\]]*$', res)
            if incomplete:
                res = res[:incomplete.start()].rstrip()
        
        return res

    def replace_references(self, text, streaming=False, chat_idx=None):
        """Replace text-analysis and MCP/general-chat references with clickable links."""

        res = str(text)

        # Text-analysis chat keeps its existing deterministic REF->quote flow.
        if self.ai_text_doc_id is not None:
            return self._replace_text_references(res, streaming=streaming)

        # General/MCP chat: convert [REF: "..."] by matching quotes against retrieved evidence.
        if streaming:
            res = re.sub(r'\[REF:[^\]]*\]', _('(source reference)'), res)
            incomplete_ref = re.search(r'\[REF:[^\]]*$', res)
            if incomplete_ref:
                res = res[:incomplete_ref.start()].rstrip()
            return res

        candidates = self._collect_ref_candidates(chat_idx)
        ref_pattern = r'\[REF:\s*(.+?)\s*\]'

        def replace_ref(match):
            raw = str(match.group(1)).strip()
            quote = raw.strip(' "\'')
            return self._resolve_ref_quote_to_anchor(quote, candidates)

        return re.sub(ref_pattern, replace_ref, res)

    def chat_list_selection_changed(self, force_update=False):
        self.ui.pushButton_delete.setEnabled(self.current_chat_idx > -1)
        if (not force_update) and (self.current_chat_idx == self.ui.listWidget_chat_list.currentRow()):
            return
        if self.app.ai.cancel(True):
            # AI generation is either finished or canceled, we can change to another chat
            self.current_chat_idx = self.ui.listWidget_chat_list.currentRow()
            self.ui.pushButton_delete.setEnabled(self.current_chat_idx > -1)
            self.history_update_message_list()
            self.update_chat_window(scroll_to_bottom=False)
        else:  # return to previous chat
            self.ui.listWidget_chat_list.setCurrentRow(self.current_chat_idx)
        
    def chat_list_item_changed(self, item: QtWidgets.QListWidgetItem):
        """This method is called whenever the name of a chat is edited in the list"""
        chat_id = self.chat_list[self.current_chat_idx][0]
        curr_name = item.text()
        cursor = self.chat_history_conn.cursor()
        cursor.execute('UPDATE chats SET name = ? WHERE id = ?', (curr_name, chat_id))
        self.chat_history_conn.commit()
        self.get_chat_list()
        self.update_chat_window()

    def open_context_menu(self, position):
        context_menu = QtWidgets.QMenu(self)
        if self.ui.listWidget_chat_list.count() > 0:
            if self.current_chat_idx > -1:
                edit_action = QAction("Edit Title", self)
                delete_action = QAction("Delete Chat", self)
                export_action = QAction("Export Chat", self)
                context_menu.addAction(edit_action)
                context_menu.addAction(delete_action)
                context_menu.addAction(export_action)
                edit_action.triggered.connect(self.edit_title)
                delete_action.triggered.connect(self.delete_chat)
                export_action.triggered.connect(self.export_chat)

            # The search function will be implemented later:
            # search_action = QAction("Search all Chats", self)
            # context_menu.addAction(search_action)
            # search_action.triggered.connect(self.search_chat)

        if len(context_menu.actions()) > 0:
            context_menu.exec(self.ui.listWidget_chat_list.mapToGlobal(position))

    def edit_title(self):
        """Edit the title of the current chat"""
        selected_item = self.ui.listWidget_chat_list.currentItem()
        if selected_item:
            self.ui.listWidget_chat_list.editItem(selected_item)

    def export_chat(self):
        """Export the current chat into a html or txt file"""
        chat_content = self.ui.ai_output.text()
        default_file_name = self.chat_list[self.current_chat_idx][1]
        default_file_name = default_file_name.replace('"', '')
        if self.last_export_dir != '':
            default_file_name = os.path.join(self.last_export_dir, default_file_name)
        else:
            default_file_name = os.path.join(os.path.dirname(self.app.project_path), default_file_name)
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.Option.DontUseNativeDialog
        file_name, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, 
            _("Export Chat"), 
            default_file_name, 
            "HTML (*.html);;Text only (*.txt)", 
            options=options            
        )                
        if file_name:
            self.last_export_dir = os.path.dirname(file_name)
            if not any(file_name.endswith(ext) for ext in [".html", ".txt"]):
                if "HTML" in selected_filter:
                    file_name += ".html"
                elif "Text" in selected_filter:
                    file_name += ".txt"
            if os.path.exists(file_name):
                msg = _('The file already exists. Do you want to override it?')
                msg_box = Message(self.app, _('Export Chat'), msg, "critical")
                if msg_box.question(self, _('Export Chat'), msg) == QtWidgets.QMessageBox.StandardButton.No:
                    return
            if file_name.endswith(".html"):
                self._export_to_html(file_name, chat_content)
            elif file_name.endswith(".txt"):
                self._export_to_txt(file_name, chat_content)

    def _export_to_html(self, file_name, content):
        # Write the chat content as HTML
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write("<html><head><meta charset='utf-8'></head><body>")
            file.write(content)
            file.write("</body></html>")

    def _export_to_txt(self, file_name, content):
        # Strip tags for plain text export and write the content as plain text
        from PyQt6.QtGui import QTextDocument
        document = QTextDocument()
        document.setHtml(content)
        plain_text_content = document.toPlainText()
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(plain_text_content)        
    
    """    
    def search_chat(self):
        # Fulll text search over all chats, will be implemented later
        selected_item = self.ui.listWidget_chat_list.currentItem()
        if selected_item:
            print(f"Searching chat: {selected_item.text()}")
    """

    def button_new_clicked(self):
        # Create QMenu
        menu = QtWidgets.QMenu()
        menu.setStyleSheet(self.font)
        menu.setToolTipsVisible(True)

        # Add actions
        action_topic_analysis = menu.addAction(_('New topic analysis chat'))
        action_topic_analysis.setIcon(self.app.ai.topic_analysis_icon())
        action_topic_analysis.setToolTip(_('Analyzing a free-search topic together with the AI.'))
        action_text_analysis = menu.addAction(_('New text analysis chat'))
        action_text_analysis.setIcon(self.app.ai.text_analysis_icon())
        action_text_analysis.setToolTip(_('Analyse a piece of text from your empirical data together with the AI.'))
        action_codings_analysis = menu.addAction(_('New code analysis chat'))
        action_codings_analysis.setIcon(self.app.ai.code_analysis_icon())
        action_codings_analysis.setToolTip(_('Analyze the data collected under a certain code together with the AI.'))
        action_general_chat = menu.addAction(_('New general chat'))
        action_general_chat.setIcon(self.app.ai.general_chat_icon())
        action_general_chat.setToolTip(_('Ask the AI anything, not related to your data.'))

        # Obtain the bottom-left point of the button in global coordinates
        button_rect = self.ui.pushButton_new_analysis.rect()  # Get the button's rect
        bottom_left_point = button_rect.bottomLeft()  # Bottom-left point
        global_bottom_left_point = self.ui.pushButton_new_analysis.mapToGlobal(bottom_left_point)  # Map to global

        # Execute the menu at the calculated position
        action = menu.exec(global_bottom_left_point)

        # Check which action was selected and do something
        if action == action_text_analysis:
            self.new_text_analysis()
        elif action == action_codings_analysis:
            self.new_code_chat()
        elif action == action_topic_analysis:
            self.new_topic_chat()
        elif action == action_general_chat:
            self.new_general_chat('New general chat', '')

    def ai_output_scroll_to_bottom(self, minVal=None, maxVal=None):  # toDO minVal, maxVal unused
        #self._ai_output_scroll_to_bottom()
        # Delay the scrolling a little to make sure that the updated text is fully rendered before scrolling to the bottom: 
        QtCore.QTimer.singleShot(200, self._ai_output_scroll_to_bottom)
        
    def _ai_output_scroll_to_bottom(self):
        if self.ai_output_autoscroll:
            self.ui.scrollArea_ai_output.verticalScrollBar().setValue(self.ui.scrollArea_ai_output.verticalScrollBar().maximum())
            self.ai_output_autoscroll = True
                                
    def history_update_message_list(self, db_conn=None):
        """Update sel.chat_msg_list from the database

        Args:
            db_conn: database conncetion, if None, use defaults to self.chat:history_conn
        """
        if self.current_chat_idx > -1:
            curr_chat_id = self.chat_list[self.current_chat_idx][0]
            if db_conn is None:
                db_conn = self.chat_history_conn 
            cursor = db_conn.cursor()
            cursor.execute('SELECT * FROM chat_messages WHERE chat_id=? ORDER BY id', (curr_chat_id,))
            self.chat_msg_list = cursor.fetchall()
            self.ai_streaming_output = ''
        else:
            self.chat_msg_list.clear()
            self.ai_streaming_output = ''
    
    def history_get_ai_messages(self):
        messages = []
        latest_agent_state_id = -1
        for msg in self.chat_msg_list:
            if msg[2] == 'agent_state':
                try:
                    msg_id = int(msg[0])
                except Exception:
                    msg_id = -1
                if msg_id > latest_agent_state_id:
                    latest_agent_state_id = msg_id

        for msg in self.chat_msg_list:
            if msg[2] == 'system':
                messages.append(SystemMessage(content=msg[4]))
            elif msg[2] == 'instruct' or msg[2] == 'user':
                messages.append(HumanMessage(content=msg[4]))
            elif msg[2] == 'ai':
                messages.append(AIMessage(content=msg[4]))
            elif msg[2] == 'tool_call':
                messages.append(AIMessage(content=msg[4]))
            elif msg[2] == 'tool_result':
                messages.append(HumanMessage(content=msg[4]))
            elif msg[2] == 'agent_state':
                # keep only the newest compact agent-state snapshot across turns
                try:
                    msg_id = int(msg[0])
                except Exception:
                    msg_id = -1
                if msg_id != latest_agent_state_id:
                    continue
                state_payload = str(msg[4]).strip()
                if state_payload != '':
                    messages.append(HumanMessage(content='Agent state snapshot:\n' + state_payload))
            elif msg[2] == 'single_instruct':
                # one-shot instruction logs must not be replayed in later turns
                continue
        return messages
    
    def history_add_message(self, msg_type, msg_author, msg_content, chat_idx=None, db_conn=None, refresh=True, commit=True):
        self.ai_streaming_output = ''
        if chat_idx is None:
            chat_idx = self.current_chat_idx
        if chat_idx > -1:
            curr_chat_id = self.chat_list[chat_idx][0]
            if msg_type == 'ai':
                msg_content = self.replace_references(msg_content, chat_idx=chat_idx)
            if db_conn is None:
                db_conn = self.chat_history_conn
            cursor = db_conn.cursor()
            # Insert new message
            cursor.execute('INSERT INTO chat_messages (chat_id, msg_type, msg_author, msg_content)'
                           ' VALUES (?, ?, ?, ?)', (curr_chat_id, msg_type, msg_author, msg_content))
            if commit:
                db_conn.commit()
            if refresh:
                self.history_update_message_list(db_conn)

    def history_add_or_append_agent_status(self, status_text: str, chat_idx=None, msg_author='ai_agent'):
        """Persist one agent status line as its own DB row (not merged)."""
        if chat_idx is None:
            chat_idx = self.current_chat_idx
        if chat_idx <= -1:
            return
        if status_text is None or status_text.strip() == '':
            return
        curr_chat_id = self.chat_list[chat_idx][0]
        cursor = self.chat_history_conn.cursor()
        status_line = status_text.strip()

        # Guard against immediate duplicate callback events.
        cursor.execute(
            "SELECT msg_author, msg_content FROM chat_messages "
            "WHERE chat_id=? AND msg_type='agent_status' ORDER BY id DESC LIMIT 1",
            (curr_chat_id,),
        )
        row = cursor.fetchone()
        if row is not None:
            prev_author = '' if row[0] is None else str(row[0])
            prev_content = '' if row[1] is None else str(row[1])
            if prev_author == str(msg_author) and prev_content == status_line:
                return

        cursor.execute('INSERT INTO chat_messages (chat_id, msg_type, msg_author, msg_content)'
                       ' VALUES (?, ?, ?, ?)', (curr_chat_id, 'agent_status', msg_author, status_line))
        self.chat_history_conn.commit()
        self.history_update_message_list()
    
    def button_question_clicked(self):
        if self.app.ai.is_busy():
            self.app.ai.cancel(True)
        else:
            self.send_user_question()
                    
    def send_user_question(self):
        if self.app.settings['ai_enable'] != 'True':
            msg = _('The AI is disabled. Go to "AI > Setup Wizard" first.')
            Message(self.app, _('AI not enabled'), msg, "warning").exec()
            return
        elif self.app.ai.is_busy():
            msg = _('The AI is busy generating a response. Click on the button on the right to stop.')
            Message(self.app, _('AI busy'), msg, "warning").exec()
            return
        elif not self.app.ai.is_ready():
            msg = _('The AI not yet fully loaded. Please wait and retry.')
            Message(self.app, _('AI not ready'), msg, "warning").exec()
            return
        self.ai_output_autoscroll = True
        q = self.ui.plainTextEdit_question.toPlainText()
        if q != '':
            if self.process_message('user', q):
                self.ui.plainTextEdit_question.clear()
                QtWidgets.QApplication.processEvents()
                        
    def process_message(self, msg_type, msg_content, chat_idx=None, db_conn=None, refresh_history=True, commit_history=True) -> bool:
        #if not self.app.ai.is_ready():
        #    msg = _('The AI is busy or not yet fully loaded. Please wait a moment and retry.')
        #    Message(self.app, _('AI not ready'), msg, "warning").exec()
        #    return False
        if chat_idx is None:
            chat_idx = self.current_chat_idx
        if chat_idx <= -1:
            self.ai_streaming_output = ''
            self.chat_msg_list.clear()
            msg = _('Please select a chat or create a new one.')
            Message(self.app, _('Chat selection'), msg, "warning").exec()
            return False
             
        if msg_type == 'info':
            # info messages are only shown on screen, not send to the AI
            self.history_add_message(msg_type, '', msg_content, chat_idx, db_conn=db_conn, refresh=refresh_history, commit=commit_history)
            self.update_chat_window()
        elif msg_type == 'agent_status':
            self.history_add_or_append_agent_status(msg_content, chat_idx)
            if chat_idx == self.current_chat_idx:
                self.update_chat_window()
        elif msg_type == 'system':
            # system messages are only added to the chat history. They are never shown on screen. 
            # The system message will be not be send to the AI immediately,
            # but together with the next user message (as part of the chat history).
            self.history_add_message(msg_type, '', msg_content, chat_idx, db_conn=db_conn, refresh=refresh_history, commit=commit_history)
        elif msg_type == 'tool_call':
            # tool messages are persisted for multi-turn MCP context, but not rendered in the chat window
            self.history_add_message(msg_type, 'ai_agent', msg_content, chat_idx, db_conn=db_conn, refresh=refresh_history, commit=commit_history)
        elif msg_type == 'tool_result':
            # tool messages are persisted for multi-turn MCP context, but not rendered in the chat window
            self.history_add_message(msg_type, 'mcp_server', msg_content, chat_idx, db_conn=db_conn, refresh=refresh_history, commit=commit_history)
        elif msg_type == 'single_instruct':
            # single_instruct messages are persisted for audit/logging, but not rendered
            # and not sent again in future turns.
            self.history_add_message(msg_type, 'ai_agent', msg_content, chat_idx, db_conn=db_conn, refresh=refresh_history, commit=commit_history)
        elif msg_type == 'agent_state':
            # compact state memory for future turns (persisted, not rendered)
            self.history_add_message(msg_type, 'ai_agent', msg_content, chat_idx, db_conn=db_conn, refresh=refresh_history, commit=commit_history)
        elif msg_type == 'instruct':
            # instruct messages are only send to the AI, but not shown on screen
            # Other than system messages, instruct messages are send immediatly and will produce an answer that is shown on screen
            if chat_idx == self.current_chat_idx:
                self.history_add_message(msg_type, '', msg_content, chat_idx)
                messages = self.history_get_ai_messages()
                self.current_streaming_chat_idx = self.current_chat_idx
                self.app.ai.ai_async_stream(self.app.ai.large_llm, 
                                            messages, 
                                            result_callback=self.ai_message_callback, 
                                            progress_callback=None, 
                                            streaming_callback=self.ai_streaming_callback, 
                                            error_callback=None)
        elif msg_type == 'user':
            # user question, shown on screen and send to the AI
            if chat_idx == self.current_chat_idx:
                self.history_add_message(msg_type, self.app.settings['codername'], msg_content, chat_idx)
                messages = self.history_get_ai_messages()
                self.current_streaming_chat_idx = self.current_chat_idx
                analysis_type = ''
                if 0 <= chat_idx < len(self.chat_list):
                    analysis_type = self.chat_list[chat_idx][2]
                if analysis_type == 'general chat':
                    self.app.ai.ai_async_query(self._mcp_general_chat_worker,
                                               self.ai_mcp_message_callback,
                                               messages,
                                               chat_idx,
                                               progress_callback=self.ai_mcp_progress_callback)
                else:
                    self.app.ai.ai_async_stream(self.app.ai.large_llm, 
                                                messages, 
                                                result_callback=self.ai_message_callback, 
                                                progress_callback=None, 
                                                streaming_callback=self.ai_streaming_callback, 
                                                error_callback=self.ai_error_callback)
                self.update_chat_window()
        elif msg_type == 'ai':
            # ai responses.
            # create temporary db connection to make it thread safe
            db_conn = sqlite3.connect(self.chat_history_path)
            try: 
                ai_model_name = self.app.ai_models[int(self.app.settings['ai_model_index'])]['name']
                msg_content = strip_think_blocks(msg_content)
                self.history_add_message(msg_type, ai_model_name, msg_content, chat_idx, db_conn)
                self.ai_streaming_output = ''
                self.update_chat_window()
            finally:
                db_conn.close()
        return True    

    def _mcp_planner_system_prompt(self) -> str:
        return (
            "Your task: Make a plan which steps are needed to fullfill the request of the user."
            "Return ONLY one JSON object with this shape:\n"
            "{"
            "\"needs_mcp\": true|false, "
            "\"skill_decision\": \"use_skill|no_skill\", "
            "\"skill_name\": \"skill id from prompts/list if skill_decision=use_skill, else empty\", "
            "\"skill_reason\": \"short reason for the decision\", "
            "\"plan_summary\": \"one short user-facing note: immediate next step + skill decision\", "
            "\"calls\": [{\"method\": \"resources/list|resources/read|resources/templates/list|initialize|prompts/list|prompts/get\", \"params\": {}}], "
            "\"answer_brief\": \"optional draft answer idea\""
            "}\n"
            "Rules:\n"
            "- Allowed methods: initialize, resources/list, resources/templates/list, resources/read, prompts/list, prompts/get.\n"
            "- The turn is already bootstrapped with initialize + resources/list + prompts/list context; avoid repeating them with identical params.\n"
            "- Use as few calls as possible and keep them focused.\n"
            "- Prefer specific reads over broad reads. Reading full empirical documents can be costly. Do this only when it is really needed.\n"
            "- Always evaluate whether a predefined skill from prompts/list is relevant.\n"
            "- Set skill_decision explicitly: use_skill when a skill can improve method quality/structure, otherwise no_skill.\n"
            "- If skill_decision=use_skill and the skill is not already loaded in the conversation, include prompts/get with params {\"name\": \"<skill_name>\"}.\n"
            "- skill_name must match a name from prompts/list when use_skill is selected.\n"
            "- plan_summary must be one sentence, user-facing, <=160 characters.\n"
            # "- plan_summary must state the immediate next step and whether a skill is used.\n"
            # "- If skill_decision=use_skill, include the skill_name in plan_summary.\n"
            "- Simple plans can be executed directly. For more complex plans, ask the user first.\n"
            "- If enough evidence is already in the conversation, set needs_mcp=false and calls=[].\n"
            "- Do not output prose outside JSON."
        )

    def _mcp_reflection_system_prompt(self) -> str:
        return (
            "Your task: Review the collected data and decide whether more MCP calls are needed, including gatering more methodological skill support. "
            "Return ONLY one JSON object with this shape:\n"
            "{"
            "\"enough_information\": true|false, "
            "\"skill_decision\": \"use_skill|no_skill|already_applied\", "
            "\"skill_name\": \"skill id from prompts/list if needed, else empty\", "
            "\"skill_reason\": \"short reason for the decision\", "
            "\"reflection_summary\": \"one short user-facing note: next steps\", "
            "\"next_step_note\": \"optional short alias if reflection_summary is empty\", "
            "\"revised_calls\": [{\"method\": \"resources/list|resources/read|resources/templates/list|initialize|prompts/list|prompts/get\", \"params\": {}}], "
            "\"answer_brief\": \"short answer plan for final response\""
            "}\n"
            "Rules:\n"
            "- Allowed methods: initialize, resources/list, resources/templates/list, resources/read, prompts/list, prompts/get.\n"
            "- Initialize, resources/list, and prompts/list are already available in context unless explicitly changed.\n"
            "- If information is sufficient, set enough_information=true and revised_calls=[].\n"
            "- If information is insufficient, propose only necessary revised_calls.\n"
            "- Re-evaluate skill needs explicitly using skill_decision and skill_reason.\n"
            "- Use skill_decision=already_applied when the relevant skill guidance is already present in the conversation.\n"
            "- If method guidance is missing, set skill_decision=use_skill and include prompts/get with params {\"name\": \"<skill_name>\"}.\n"
            "- reflection_summary must be one sentence, user-facing, <=160 characters.\n"
            # "- reflection_summary must state the immediate next step and the current skill status (use_skill, already_applied, or no_skill).\n"
            # "- If skill_decision=use_skill or already_applied, include the skill_name in reflection_summary.\n"
            "- Avoid boilerplate like 'I will' or 'Next step is' unless strictly needed.\n"
            "- next_step_note is optional and only used when reflection_summary is empty.\n"
            "- Do not output prose outside JSON."
        )

    def _mcp_final_answer_system_prompt(self) -> str:
        return (
            "Your task: "
            "Provide a final answer for the user in normal prose based on the conversation and retrieved project context. "
            "Do not output JSON. Do not call MCP. "
            "If information is missing, state that briefly and avoid making up details. "
            "When you refer to empirical text evidence, add citations in this exact form: "
            "[REF: \"exact quote from the retrieved evidence\"]. "
            "The quote inside REF must be copied exactly from retrieved evidence (no paraphrasing, no corrections, no translation). "
            #"Do not generate HTML links yourself."
        )

    def _run_mcp_request(self, method: str, params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        request = {
            "jsonrpc": "2.0",
            "id": self.ai_mcp_server.new_request_id(),
            "method": method,
            "params": params,
        }
        response = self.ai_mcp_server.handle_request(request)
        return request, response

    def _invoke_json_llm(self, messages: List[Any]) -> Dict[str, Any]:
        """Invoke model and parse one JSON object response."""

        llm_response = self.app.ai.invoke_with_logging(
            self.app.ai.large_llm,
            messages,
            response_format={"type": "json_object"},
            context='mcp_json_control',
            fallback_without_response_format=True,
        )
        raw = str(llm_response.content).strip()
        try:
            data = json.loads(raw)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _normalize_mcp_calls(self, raw_calls: Any, allowed_methods: set[str], max_calls: int) -> List[Dict[str, Any]]:
        """Validate and clamp model-proposed MCP calls."""

        normalized: List[Dict[str, Any]] = []
        if not isinstance(raw_calls, list):
            return normalized
        for item in raw_calls:
            if len(normalized) >= max_calls:
                break
            if not isinstance(item, dict):
                continue
            method = str(item.get("method", "")).strip()
            if method not in allowed_methods:
                continue
            params = item.get("params", {})
            if not isinstance(params, dict):
                params = {}
            normalized.append({"method": method, "params": params})
        return normalized

    def _ensure_skill_prompt_call(self, skill_decision: Any, skill_name: Any,
                                  calls: List[Dict[str, Any]], mcp_cache: Dict[str, Dict[str, Any]],
                                  max_calls: int) -> List[Dict[str, Any]]:
        """Ensure prompts/get is planned when the model decided to use a skill."""

        decision = str(skill_decision if skill_decision is not None else "").strip().lower()
        if decision not in ("use_skill", "needs_skill", "skill"):
            return calls

        resolved_name = str(skill_name if skill_name is not None else "").strip()
        if resolved_name == "":
            return calls

        wanted_params = {"name": resolved_name}
        wanted_key = self._mcp_call_key("prompts/get", wanted_params)
        if wanted_key in mcp_cache:
            return calls

        for call in calls:
            if not isinstance(call, dict):
                continue
            method = str(call.get("method", "")).strip()
            params = call.get("params", {})
            if not isinstance(params, dict):
                params = {}
            if method == "prompts/get" and str(params.get("name", "")).strip() == resolved_name:
                return calls

        merged: List[Dict[str, Any]] = [{"method": "prompts/get", "params": wanted_params}]
        seen: set[str] = {wanted_key}
        for call in calls:
            if not isinstance(call, dict):
                continue
            method = str(call.get("method", "")).strip()
            params = call.get("params", {})
            if not isinstance(params, dict):
                params = {}
            key = self._mcp_call_key(method, params)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"method": method, "params": params})
            if len(merged) >= max_calls:
                break
        return merged

    def _json_bool(self, value: Any, default: bool) -> bool:
        """Parse relaxed JSON boolean-like values."""

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("true", "1", "yes"):
                return True
            if text in ("false", "0", "no", ""):
                return False
        return default

    def _parse_mcp_agent_action(self, raw_text: str) -> Dict[str, Any]:
        """Interpret one structured control message from the model."""

        try:
            data = json.loads(raw_text)
        except Exception:
            return {"action": "done"}
        if not isinstance(data, dict):
            return {"action": "done"}

        action = str(data.get("action", "")).strip().lower()
        if action in ("done", "finish", "finished"):
            return {"action": "done"}

        if action in ("mcp_call", "call_mcp", "call"):
            method = str(data.get("method", "")).strip()
            params = data.get("params", {})
            if not isinstance(params, dict):
                params = {}
            return {"action": "mcp_call", "method": method, "params": params}

        # fallback shape: {"method": "...", "params": {...}}
        if "method" in data:
            method = str(data.get("method", "")).strip()
            params = data.get("params", {})
            if not isinstance(params, dict):
                params = {}
            return {"action": "mcp_call", "method": method, "params": params}
        return {"action": "done"}

    def _mcp_call_key(self, method: str, params: Dict[str, Any]) -> str:
        try:
            params_key = json.dumps(params, sort_keys=True, ensure_ascii=False)
        except Exception:
            params_key = str(params)
        return method + "|" + params_key

    def _extract_mcp_response_cache(self, messages: List[Any]) -> Dict[str, Dict[str, Any]]:
        cache: Dict[str, Dict[str, Any]] = {}
        pending_key: Optional[str] = None
        prefix = "MCP response:\n"
        for msg in messages:
            if isinstance(msg, AIMessage):
                action = self._parse_mcp_agent_action(str(msg.content))
                if action.get("action") == "mcp_call":
                    method = str(action.get("method", "")).strip()
                    params = action.get("params", {})
                    if not isinstance(params, dict):
                        params = {}
                    pending_key = self._mcp_call_key(method, params)
                else:
                    pending_key = None
                continue

            if isinstance(msg, HumanMessage) and pending_key is not None:
                raw = str(msg.content)
                if raw.startswith(prefix):
                    try:
                        response = json.loads(raw[len(prefix):])
                        if isinstance(response, dict):
                            cache[pending_key] = response
                    except Exception:
                        pass
                pending_key = None
            else:
                pending_key = None
        return cache

    def _emit_mcp_status(self, signals, chat_idx: int, status_event: Optional[Dict[str, Any]]):
        status_msg = self.ai_mcp_server.status_event_to_text(status_event)
        if status_msg.strip() == '':
            return
        if signals is None or signals.progress is None:
            return
        payload = {"chat_idx": chat_idx, "status": status_msg, "status_event": status_event}
        signals.progress.emit(json.dumps(payload, ensure_ascii=False))

    def _normalize_progress_note(self, text: Any, max_length: int = 220) -> str:
        """Normalize model progress notes for compact UI display."""

        note = str(text if text is not None else "").replace("\r", " ").replace("\n", " ")
        note = " ".join(note.split()).strip()
        if note == "":
            return ""
        if len(note) > max_length:
            note = note[: max_length - 3].rstrip() + "..."
        return note

    def _emit_mcp_status_text(self, signals, chat_idx: int, status_text: str):
        """Emit one free-text MCP progress line."""

        status = self._normalize_progress_note(status_text)
        if status == "":
            return
        if signals is None or signals.progress is None:
            return
        payload = {"chat_idx": chat_idx, "status": status, "status_event": None}
        signals.progress.emit(json.dumps(payload, ensure_ascii=False))

    def _short_reflection_next_step_note(self, reflection_summary: str,
                                         model_next_step_note: str) -> str:
        """Keep reflection content, only shorten/format it for UI display."""

        candidate = self._normalize_progress_note(reflection_summary, max_length=1000)
        if candidate == "":
            candidate = self._normalize_progress_note(model_next_step_note, max_length=1000)
        if candidate == "":
            return ""

        max_length = 160
        if len(candidate) <= max_length:
            return candidate

        # Prefer a full sentence boundary before hard truncation.
        sentence_end = max(candidate.rfind("." , 0, max_length),
                           candidate.rfind("!", 0, max_length),
                           candidate.rfind("?", 0, max_length))
        if sentence_end >= 40:
            return candidate[: sentence_end + 1].strip()
        return self._normalize_progress_note(candidate, max_length=max_length)

    def _safe_int(self, value: Any, default: int = -1) -> int:
        try:
            if value is None or isinstance(value, bool):
                return default
            return int(value)
        except Exception:
            return default

    def _tool_result_messages_for_chat(self, chat_idx: Optional[int]) -> List[str]:
        """Return raw tool_result message contents for one chat."""

        if chat_idx is None:
            chat_idx = self.current_chat_idx
        if chat_idx is None or chat_idx < 0 or chat_idx >= len(self.chat_list):
            return []
        curr_chat_id = self.chat_list[chat_idx][0]
        cursor = self.chat_history_conn.cursor()
        cursor.execute(
            "SELECT msg_content FROM chat_messages WHERE chat_id=? AND msg_type='tool_result' ORDER BY id",
            (curr_chat_id,),
        )
        rows = cursor.fetchall()
        return [str(row[0]) for row in rows if isinstance(row, tuple) and len(row) > 0 and row[0] is not None]

    def _extract_ref_candidates_from_tool_result(self, tool_result_raw: str) -> List[Dict[str, Any]]:
        """Extract evidence spans from one persisted MCP tool_result message."""

        prefix = "MCP response:\n"
        if not isinstance(tool_result_raw, str) or not tool_result_raw.startswith(prefix):
            return []
        try:
            rpc_response = json.loads(tool_result_raw[len(prefix):])
        except Exception:
            return []
        if not isinstance(rpc_response, dict):
            return []

        result = rpc_response.get("result", {})
        if not isinstance(result, dict):
            return []
        contents = result.get("contents", [])
        if not isinstance(contents, list):
            return []

        candidates: List[Dict[str, Any]] = []
        for content in contents:
            if not isinstance(content, dict):
                continue
            uri = str(content.get("uri", "")).split("?", 1)[0]
            text_blob = content.get("text", None)
            if not isinstance(text_blob, str) or text_blob.strip() == "":
                continue
            try:
                payload = json.loads(text_blob)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            if re.fullmatch(r"qualcoder://codes/segments/\d+", uri):
                segments = payload.get("segments", [])
                if not isinstance(segments, list):
                    continue
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    source_id = self._safe_int(seg.get("fid", None), -1)
                    start = self._safe_int(seg.get("pos0", None), -1)
                    quote = str(seg.get("quote", ""))
                    if source_id <= 0 or start < 0 or quote.strip() == "":
                        continue
                    source_name = str(seg.get("source_name", "")).strip()
                    if source_name == "":
                        source_name = self.get_filename(source_id)
                    candidates.append(
                        {
                            "source_id": source_id,
                            "source_name": source_name,
                            "start": start,
                            "length": len(quote),
                            "text": quote,
                        }
                    )
                continue

            if re.fullmatch(r"qualcoder://documents/text/\d+", uri):
                source_id = self._safe_int(payload.get("id", None), -1)
                start = self._safe_int(payload.get("start", None), -1)
                excerpt = str(payload.get("text", ""))
                if source_id <= 0 or start < 0 or excerpt.strip() == "":
                    continue
                source_name = str(payload.get("name", "")).strip()
                if source_name == "":
                    source_name = self.get_filename(source_id)
                candidates.append(
                    {
                        "source_id": source_id,
                        "source_name": source_name,
                        "start": start,
                        "length": len(excerpt),
                        "text": excerpt,
                    }
                )
        return candidates

    def _collect_ref_candidates(self, chat_idx: Optional[int]) -> List[Dict[str, Any]]:
        """Collect deduplicated empirical evidence spans from MCP tool results."""

        tool_results = self._tool_result_messages_for_chat(chat_idx)
        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw in tool_results:
            for item in self._extract_ref_candidates_from_tool_result(raw):
                source_id = self._safe_int(item.get("source_id", None), -1)
                start = self._safe_int(item.get("start", None), -1)
                text = str(item.get("text", ""))
                key = f"{source_id}|{start}|{len(text)}|{text[:120]}"
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(item)
        return candidates

    def _resolve_ref_quote_to_anchor(self, quote: str, candidates: List[Dict[str, Any]]) -> str:
        """Resolve one exact/fuzzy quote to a quote: anchor."""

        quote_text = str(quote if quote is not None else "").strip()
        if quote_text == "":
            return _('(unknown reference)')

        best_candidate: Optional[Dict[str, Any]] = None
        best_local_start = -1
        best_local_end = -1

        for item in candidates:
            segment_text = str(item.get("text", ""))
            local_start = segment_text.find(quote_text)
            if local_start > -1:
                best_candidate = item
                best_local_start = local_start
                best_local_end = local_start + len(quote_text)
                break

        if best_candidate is None:
            for item in candidates:
                segment_text = str(item.get("text", ""))
                local_start, local_end = ai_quote_search(quote_text, segment_text)
                if local_start > -1 < local_end:
                    best_candidate = item
                    best_local_start = local_start
                    best_local_end = local_end
                    break

        if best_candidate is None:
            print(quote_text)
            return _('(unknown reference)')

        source_id = self._safe_int(best_candidate.get("source_id", None), -1)
        span_start = self._safe_int(best_candidate.get("start", None), -1)
        source_name = str(best_candidate.get("source_name", "")).strip()
        if source_name == "":
            source_name = self.get_filename(source_id)
        if source_id <= 0 or span_start < 0:
            return _('(unknown reference)')

        abs_start = span_start + best_local_start
        fulltext = self.app.get_text_fulltext(source_id)
        if fulltext is None:
            return _('(unknown reference)')
        full_len = len(fulltext)
        if full_len <= 0:
            return _('(unknown reference)')
        if abs_start < 0:
            abs_start = 0
        if abs_start >= full_len:
            return _('(unknown reference)')

        abs_end = abs_start + max(1, best_local_end - best_local_start)
        if abs_end > full_len:
            abs_end = full_len
        if abs_end <= abs_start:
            abs_end = min(full_len, abs_start + 1)
            if abs_end <= abs_start:
                return _('(unknown reference)')
        abs_len = abs_end - abs_start

        line_start, line_end = self.app.get_line_numbers(fulltext, abs_start, abs_end)
        if line_start > 0 and line_end > 0:
            if line_start == line_end:
                label = f"{source_name}: {line_start}"
            else:
                label = f"{source_name}: {line_start} - {line_end}"
        else:
            label = source_name if source_name != "" else str(source_id)
        return f'(<a href="quote:{source_id}_{abs_start}_{abs_len}">{label}</a>)'

    def _mcp_general_chat_worker(self, messages: List[Any], chat_idx: int, signals=None) -> Dict[str, Any]:
        """Background worker: staged agent flow with MCP resource access."""

        result: Dict[str, Any] = {
            "chat_idx": chat_idx,
            "stream_messages": [],
            "tool_messages": [],
            "canceled": False,
        }
        allowed_methods = {
            "initialize",
            "resources/list",
            "resources/templates/list",
            "resources/read",
            "prompts/list",
            "prompts/get",
        }

        try:
            history_messages: List[Any] = list(messages)
            agent_messages: List[Any] = [msg for msg in history_messages if not isinstance(msg, SystemMessage)]
            final_hint = ''
            tool_messages: List[Dict[str, str]] = []
            tool_messages_streamed = signals is not None and getattr(signals, "progress", None) is not None
            mcp_cache = self._extract_mcp_response_cache(history_messages)
            bootstrap_calls: List[Tuple[str, Dict[str, Any]]] = [
                ("initialize", {}),
                ("resources/list", {}),
                ("prompts/list", {}),
            ]
            max_calls_per_round = 4
            max_reflection_rounds = 4
            max_total_tool_calls = 12 + len(bootstrap_calls)
            total_tool_calls = 0
            stop_reason = ""

            def append_tool_exchange(method_name: str, method_params: Dict[str, Any], rpc_response: Dict[str, Any]):
                call_content = json.dumps(
                    {"action": "mcp_call", "method": method_name, "params": method_params},
                    ensure_ascii=False,
                )
                result_content = "MCP response:\n" + json.dumps(rpc_response, ensure_ascii=False)
                agent_messages.append(AIMessage(content=call_content))
                agent_messages.append(HumanMessage(content=result_content))
                if tool_messages_streamed:
                    payload_call = {
                        "chat_idx": chat_idx,
                        "msg_type": "tool_call",
                        "msg_author": "ai_agent",
                        "msg_content": call_content,
                    }
                    signals.progress.emit(json.dumps(payload_call, ensure_ascii=False))
                    payload_result = {
                        "chat_idx": chat_idx,
                        "msg_type": "tool_result",
                        "msg_author": "mcp_server",
                        "msg_content": result_content,
                    }
                    signals.progress.emit(json.dumps(payload_result, ensure_ascii=False))
                else:
                    tool_messages.append({"msg_type": "tool_call", "msg_author": "ai_agent", "msg_content": call_content})
                    tool_messages.append({"msg_type": "tool_result", "msg_author": "mcp_server", "msg_content": result_content})

            def append_single_instruct_log(phase: str, role: str, content: str):
                payload = json.dumps(
                    {"phase": phase, "role": role, "content": content},
                    ensure_ascii=False,
                )
                if tool_messages_streamed:
                    progress_payload = {
                        "chat_idx": chat_idx,
                        "msg_type": "single_instruct",
                        "msg_author": "ai_agent",
                        "msg_content": payload,
                    }
                    signals.progress.emit(json.dumps(progress_payload, ensure_ascii=False))
                else:
                    tool_messages.append({"msg_type": "single_instruct", "msg_author": "ai_agent", "msg_content": payload})

            # Ensure baseline environment context exists before planning.
            for method, params in bootstrap_calls:
                if self.app.ai.ai_async_is_canceled:
                    result["canceled"] = True
                    return result
                call_key = self._mcp_call_key(method, params)
                if call_key in mcp_cache:
                    continue
                status_event = self.ai_mcp_server.describe_status_event(method, params)
                self._emit_mcp_status(signals, chat_idx, status_event)
                _request, response = self._run_mcp_request(method, params)
                mcp_cache[call_key] = response
                total_tool_calls += 1
                append_tool_exchange(method, params, response)

            planner_system_prompt = self._build_mcp_combined_system_prompt(self._mcp_planner_system_prompt())
            planner_user_prompt = "Create the initial MCP plan now."
            append_single_instruct_log("planning", "system", planner_system_prompt)
            append_single_instruct_log("planning", "user", planner_user_prompt)
            self._emit_mcp_status_text(signals, chat_idx, _("Planning..."))
            planner_messages: List[Any] = [SystemMessage(content=planner_system_prompt)]
            planner_messages.extend(agent_messages)
            planner_messages.append(HumanMessage(content=planner_user_prompt))
            plan_data = self._invoke_json_llm(planner_messages)
            planned_calls = self._normalize_mcp_calls(plan_data.get("calls", []), allowed_methods, max_calls_per_round)
            planned_calls = self._ensure_skill_prompt_call(
                plan_data.get("skill_decision", ""),
                plan_data.get("skill_name", ""),
                planned_calls,
                mcp_cache,
                max_calls_per_round,
            )
            needs_mcp = self._json_bool(plan_data.get("needs_mcp", True), True)
            plan_summary = str(plan_data.get("plan_summary", "")).strip()
            latest_plan_summary = plan_summary
            latest_reflection_summary = ""
            if plan_summary != "":
                self._emit_mcp_status_text(signals, chat_idx, plan_summary)
            initial_brief = str(plan_data.get("answer_brief", "")).strip()
            if initial_brief != "":
                final_hint = initial_brief
            if not needs_mcp:
                planned_calls = []

            for reflection_round in range(max_reflection_rounds):
                if self.app.ai.ai_async_is_canceled:
                    result["canceled"] = True
                    return result

                executed_any_call = False
                for call in planned_calls:
                    if self.app.ai.ai_async_is_canceled:
                        result["canceled"] = True
                        return result
                    if total_tool_calls >= max_total_tool_calls:
                        stop_reason = "max_total_tool_calls_reached"
                        break
                    method = str(call.get("method", "")).strip()
                    params = call.get("params", {})
                    if not isinstance(params, dict):
                        params = {}
                    if method not in allowed_methods:
                        response = {
                            "jsonrpc": "2.0",
                            "id": self.ai_mcp_server.new_request_id(),
                            "error": {"code": -32601, "message": "Method not found", "data": method},
                        }
                    else:
                        call_key = self._mcp_call_key(method, params)
                        if call_key in mcp_cache:
                            response = mcp_cache[call_key]
                        else:
                            status_event = self.ai_mcp_server.describe_status_event(method, params)
                            self._emit_mcp_status(signals, chat_idx, status_event)
                            _request, response = self._run_mcp_request(method, params)
                            mcp_cache[call_key] = response
                            total_tool_calls += 1
                        executed_any_call = True
                    append_tool_exchange(method, params, response)

                reflection_system_prompt = self._build_mcp_combined_system_prompt(self._mcp_reflection_system_prompt())
                append_single_instruct_log("reflection", "system", reflection_system_prompt)
                reflection_messages: List[Any] = [SystemMessage(content=reflection_system_prompt)]
                reflection_messages.extend(agent_messages)
                reflection_prompt = "Reflect on sufficiency of the collected evidence and return JSON now."
                if plan_summary != "":
                    reflection_prompt += "\nInitial plan summary:\n" + plan_summary
                append_single_instruct_log("reflection", "user", reflection_prompt)
                reflection_messages.append(HumanMessage(content=reflection_prompt))
                reflection_data = self._invoke_json_llm(reflection_messages)
                reflection_summary = str(reflection_data.get("reflection_summary", "")).strip()
                if reflection_summary != "":
                    latest_reflection_summary = reflection_summary
                reflection_brief = str(reflection_data.get("answer_brief", "")).strip()
                if reflection_brief != "":
                    final_hint = reflection_brief
                enough_information = self._json_bool(reflection_data.get("enough_information", False), False)
                reflection_next_step_note = str(reflection_data.get("next_step_note", "")).strip()
                revised_calls = self._normalize_mcp_calls(
                    reflection_data.get("revised_calls", []), allowed_methods, max_calls_per_round
                )
                revised_calls = self._ensure_skill_prompt_call(
                    reflection_data.get("skill_decision", ""),
                    reflection_data.get("skill_name", ""),
                    revised_calls,
                    mcp_cache,
                    max_calls_per_round,
                )
                short_reflection_note = self._short_reflection_next_step_note(
                    reflection_summary,
                    reflection_next_step_note,
                )
                if short_reflection_note != "":
                    self._emit_mcp_status_text(signals, chat_idx, short_reflection_note)
                if enough_information:
                    stop_reason = "enough_information"
                    break

                if len(revised_calls) == 0:
                    if executed_any_call:
                        replanner_system_prompt = self._build_mcp_combined_system_prompt(
                            self._mcp_planner_system_prompt()
                        )
                        replanner_user_prompt = (
                            "The previous reflection said more evidence may be needed. "
                            "Propose a revised MCP plan now."
                        )
                        append_single_instruct_log("replanning", "system", replanner_system_prompt)
                        append_single_instruct_log("replanning", "user", replanner_user_prompt)
                        replanner_messages: List[Any] = [SystemMessage(content=replanner_system_prompt)]
                        replanner_messages.extend(agent_messages)
                        replanner_messages.append(
                            HumanMessage(
                                content=replanner_user_prompt
                            )
                        )
                        replan_data = self._invoke_json_llm(replanner_messages)
                        replan_summary = str(replan_data.get("plan_summary", "")).strip()
                        if replan_summary != "":
                            latest_plan_summary = replan_summary
                        if replan_summary != "":
                            self._emit_mcp_status_text(signals, chat_idx, replan_summary)
                        revised_calls = self._normalize_mcp_calls(
                            replan_data.get("calls", []), allowed_methods, max_calls_per_round
                        )
                        revised_calls = self._ensure_skill_prompt_call(
                            replan_data.get("skill_decision", ""),
                            replan_data.get("skill_name", ""),
                            revised_calls,
                            mcp_cache,
                            max_calls_per_round,
                        )
                    if len(revised_calls) == 0:
                        stop_reason = "no_more_valid_calls"
                        break
                planned_calls = revised_calls
            else:
                if stop_reason == "":
                    stop_reason = "max_reflection_rounds_reached"

            self._emit_mcp_status(signals, chat_idx, self.ai_mcp_server.describe_host_status_event("final_response"))
            final_prompt = (
                "Now provide the final answer to the user in normal prose. "
                "Do not call MCP anymore. "
                "The final answer must follow the current conversation language. "
                "When referring to empirical text evidence, cite it as [REF: \"exact quote\"]."
            )
            if final_hint != '':
                final_prompt += '\nHere is a draft idea from your internal planning:\n' + final_hint
            if stop_reason not in ("", "enough_information"):
                final_prompt += (
                    "\nIf the available project evidence is incomplete, clearly state uncertainty and "
                    "mention what additional project material would help."
                )

            agent_state_snapshot = {
                "type": "mcp_agent_state",
                "latest_plan_summary": self._normalize_progress_note(latest_plan_summary, max_length=600),
                "latest_reflection_summary": self._normalize_progress_note(latest_reflection_summary, max_length=600),
                "final_hint": self._normalize_progress_note(final_hint, max_length=600),
                "stop_reason": stop_reason,
                "pending_calls": planned_calls if isinstance(planned_calls, list) else [],
            }
            agent_state_content = json.dumps(agent_state_snapshot, ensure_ascii=False)
            if tool_messages_streamed:
                progress_payload = {
                    "chat_idx": chat_idx,
                    "msg_type": "agent_state",
                    "msg_author": "ai_agent",
                    "msg_content": agent_state_content,
                }
                signals.progress.emit(json.dumps(progress_payload, ensure_ascii=False))
            else:
                tool_messages.append(
                    {
                        "msg_type": "agent_state",
                        "msg_author": "ai_agent",
                        "msg_content": agent_state_content,
                    }
                )

            final_system_prompt = self._build_mcp_combined_system_prompt(self._mcp_final_answer_system_prompt())
            final_stream_messages: List[Any] = [SystemMessage(content=final_system_prompt)]
            final_stream_messages.extend(agent_messages)
            final_stream_messages.append(HumanMessage(content=final_prompt))
            result["stream_messages"] = final_stream_messages
            result["tool_messages"] = tool_messages
        except Exception as err:
            result["error"] = _('Error during MCP-based general chat: ') + str(err)
        return result

    def ai_mcp_message_callback(self, mcp_result):
        """Called when the MCP-based general chat worker has finished."""

        self.ai_streaming_output = ''
        if not isinstance(mcp_result, dict):
            self.process_message('info', _('Error: Invalid result from MCP general chat worker.'), self.current_streaming_chat_idx)
            return

        chat_idx = int(mcp_result.get("chat_idx", self.current_streaming_chat_idx))
        if chat_idx < 0 or chat_idx >= len(self.chat_list):
            return

        if mcp_result.get("canceled", False):
            self.process_message('info', _('Chat has been canceled by the user.'), chat_idx)
            return

        err = str(mcp_result.get("error", "")).strip()
        if err != '':
            self.process_message('info', err, chat_idx)
            return

        stream_messages = mcp_result.get("stream_messages", None)
        if stream_messages is None or not isinstance(stream_messages, list) or len(stream_messages) == 0:
            self.process_message('info', _('Error: Invalid message stream from MCP general chat worker.'), chat_idx)
            return

        tool_messages = mcp_result.get("tool_messages", None)
        if isinstance(tool_messages, list) and len(tool_messages) > 0:
            db_conn = sqlite3.connect(self.chat_history_path)
            try:
                for item in tool_messages:
                    if not isinstance(item, dict):
                        continue
                    msg_type = str(item.get("msg_type", "")).strip()
                    msg_content = str(item.get("msg_content", ""))
                    if msg_type in ("tool_call", "tool_result", "single_instruct", "agent_state"):
                        self.process_message(
                            msg_type,
                            msg_content,
                            chat_idx,
                            db_conn=db_conn,
                            refresh_history=False,
                            commit_history=False,
                        )
                db_conn.commit()
                self.history_update_message_list(db_conn)
            finally:
                db_conn.close()

        self.current_streaming_chat_idx = chat_idx
        self.app.ai.ai_async_stream(self.app.ai.large_llm,
                                    stream_messages,
                                    result_callback=self.ai_message_callback,
                                    progress_callback=None,
                                    streaming_callback=self.ai_streaming_callback,
                                    error_callback=self.ai_error_callback)
        self.update_chat_window()

    def ai_mcp_progress_callback(self, progress_msg):
        """Receive live MCP status updates from the worker thread."""
        try:
            payload = json.loads(str(progress_msg))
        except Exception:
            payload = None

        if isinstance(payload, dict):
            chat_idx = int(payload.get("chat_idx", self.current_chat_idx))
            msg_type = str(payload.get("msg_type", "")).strip()
            if msg_type in ("tool_call", "tool_result", "single_instruct", "agent_state"):
                msg_content = str(payload.get("msg_content", ""))
                if msg_content != "":
                    self.process_message(
                        msg_type,
                        msg_content,
                        chat_idx,
                        refresh_history=False,
                        commit_history=True,
                    )
            status = str(payload.get("status", "")).strip()
            if status == '':
                status = self.ai_mcp_server.status_event_to_text(payload.get("status_event", None)).strip()
        else:
            status = str(progress_msg).strip()
            chat_idx = self.current_chat_idx
        if status == '':
            return
        self.process_message('agent_status', status, chat_idx)
    
    def ai_streaming_callback(self, streamed_text):  # TODO streamed_text unused
        self.update_chat_window()

    def _send_message(self, messages, progress_callback=None):    # TODO progress_callback unused
        # Callback for async call
        self.ai_streaming_output = ''
        self.ai_stream_buffer = ""
        self.ai_stream_in_ref = False
        self.current_streaming_chat_idx = self.current_chat_idx
        req_id = self.app.ai.log_llm_request(self.app.ai.large_llm, messages, context='dialog_send_message')
        try:
            for chunk in self.app.ai.large_llm.stream(messages):
                if self.app.ai.ai_async_is_canceled:
                    break  # Cancel the streaming
                elif self.current_chat_idx != self.current_streaming_chat_idx:
                    # switched to another chat, cancel also
                    break
                else:
                    # check if we need to process reference:
                    curr_text = self.ai_streaming_output
                    new_data = str(chunk.content)
                    for char in new_data:
                        if self.ai_stream_in_ref:
                            if char == "]":
                                # End of reference reached
                                ref_replacement = self.ai_stream_process_reference(self.buffer)
                                curr_text += ref_replacement
                                self.ai_stream_buffer = ""
                                self.ai_stream_in_ref = False
                            else:
                                self.ai_stream_buffer += char
                        else:
                            curr_text += char
                            # Check for the start of a reference
                            if curr_text.endswith('[REF:'):
                                self.ai_stream_in_ref = True
                                self.ai_stream_buffer = '[REF:'
                                curr_text = curr_text[:-(len(self.buffer))]
                    self.ai_streaming_output = curr_text
                    if not self.is_updating_chat_window:
                        self.update_chat_window()
        except Exception as err:
            self.app.ai.log_llm_error(req_id, self.app.ai.large_llm, err, context='dialog_send_message')
            raise
        if not self.app.ai.ai_async_is_canceled and self.current_chat_idx == self.current_streaming_chat_idx:
            self.app.ai.log_llm_response(req_id, self.app.ai.large_llm, self.ai_streaming_output, context='dialog_send_message')
        return self.ai_streaming_output
    
    def ai_stream_process_reference(self, reference):
        '''Replace a reference to the empirical data woth a clicable link'''
        return " [REFERENCE] "

    
    def ai_message_callback(self, ai_result):
        """Called if the AI has finished sending its response.
        The streamed resonse is now replaced with the final one.
        """
        self.ai_streaming_output = ''
        if ai_result != '':
            self.process_message('ai', ai_result, self.current_streaming_chat_idx)
        else:
            if self.app.ai.ai_async_is_canceled:
                self.process_message('info', _('Chat has been canceled by the user.'))
            else:
                self.process_message('info', _('Error: The AI returned an empty result. This may indicate that the AI model is not available at the moment. Try again later or choose a different model.'), self.current_streaming_chat_idx)
            
    def ai_error_callback(self, exception_type, value, tb_obj):
        """Called if the AI returns an error"""
        self.ai_streaming_output = ''
        ai_model_name = self.app.ai_models[int(self.app.settings['ai_model_index'])]['name']
        msg = _('Error communicating with ' + ai_model_name + '\n')
        msg += exception_type.__name__ + ': ' + str(html_to_text(value))
        if hasattr(value, 'message'):
            msg += f' {value.message}'
        tb = '\n'.join(traceback.format_tb(tb_obj))
        if hasattr(value, 'body'):
            tb += f'\n{value.body}\n'
        logger.error(_("Uncaught exception: ") + msg + '\n' + tb)
        # Error msg in chat and trigger message box show
        self.process_message('info', msg, self.current_streaming_chat_idx)    
        qt_exception_hook._exception_caught.emit(msg, tb)        
    
    def eventFilter(self, source, event):
        # Check if the event is a KeyPress, source is the lineEdit, and the key is Enter
        if (event.type() == QEvent.Type.KeyPress and source is self.ui.plainTextEdit_question and
            (event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter)):
            # Shift + Return/Enter creates a new line. Just pressing Return/Enter sends the question to the AI:
            if not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.send_user_question()
                return True  # Event handled
        # For all other cases, return super's eventFilter result
        return super().eventFilter(source, event)
    
    def on_linkHovered(self, link: str):

        if link:
            # Show tooltip when hovering over a link
            if link.startswith('coding:'):
                try:
                    coding_id = link[len('coding:'):]
                    cursor = self.app.conn.cursor()
                    sql = (f'SELECT code_text.ctid, source.name, code_text.seltext '
                            f'FROM code_text JOIN source ON code_text.fid = source.id '
                            f'WHERE code_text.ctid = {coding_id}')
                    cursor.execute(sql)
                    coding = cursor.fetchone()
                except Exception as e:
                    logger.debug(f'Link: "{link}" - Error: {e}')
                    coding = None                
                if coding is not None:
                    tooltip_txt = f'{coding[1]}:\n'  # file name
                    tooltip_txt += f'"{coding[2]}"'  # seltext
                else:
                    tooltip_txt = _('Invalid source reference.')
                QtWidgets.QToolTip.showText(QCursor.pos(), tooltip_txt, self.ui.ai_output)
            elif link.startswith('chunk:'):
                try:
                    chunk_id = link[len('chunk:'):]
                    chunk_id_elem = chunk_id.split('_')
                    if len(chunk_id_elem) == 3:  # legacy format
                        source_id, start, length = chunk_id_elem
                        line_start = 0
                        line_end = 0
                    else:
                        source_id, start, length, line_start, line_end = chunk_id_elem
                    cursor = self.app.conn.cursor()
                    sql = f'SELECT name, fulltext FROM source WHERE id = {source_id}'
                    cursor.execute(sql)
                    source = cursor.fetchone()
                    tooltip_txt = f'{source[0]}: {line_start} - {line_end}\n'  # File name
                    tooltip_txt += f'"{source[1][int(start):int(start) + int(length)]}"'  # Chunk extracted from fulltext                    
                except Exception as e:
                    logger.debug(f'Link: "{link}" - Error: {e}')
                    source = None  # TODO source not used
                    tooltip_txt = _('Invalid source reference.')
                QtWidgets.QToolTip.showText(QCursor.pos(), tooltip_txt, self.ui.ai_output)
            elif link.startswith('quote:'):
                # tooltip_txt = _('Open source document')
                tooltip_txt = ''
                try:
                    quote_id = link[len('quote:'):]
                    source_id, start, length = quote_id.split('_')
                    tooltip_txt = f'"{self.app.get_text_fulltext(int(source_id), int(start), int(length))}"'
                except Exception as e:
                    print(e)
                    tooltip_txt = ''
                if tooltip_txt == '':
                    tooltip_txt = _('Error retrieving source text')
                QtWidgets.QToolTip.showText(QCursor.pos(), tooltip_txt, self.ui.ai_output)
            elif link.startswith('action:topic_chat_analyze_more'):
                tooltip_txt = _('This expands the data basis for the analysis. However, '
                                'be careful not to overdo it, as this can also dilute '
                                'the focus of the analysis.')
                QtWidgets.QToolTip.showText(QCursor.pos(), tooltip_txt, self.ui.ai_output)
        else:
            QtWidgets.QToolTip.hideText()
            
    def on_linkActivated(self, link: str):

        if link:
            # Open doc in coding window 
            if link.startswith('coding:'):
                try:
                    coding_id = link[len('coding:'):]
                    cursor = self.app.conn.cursor()
                    sql = (f'SELECT fid, pos0, pos1 '
                            f'FROM code_text '
                            f'WHERE code_text.ctid = {coding_id}')
                    cursor.execute(sql)
                    coding = cursor.fetchone()
                except Exception as e:
                    logger.debug(f'Link: "{link}" - Error: {e}')
                    coding = None
                if coding is not None:
                    self.main_window.text_coding(task='documents', 
                                                 doc_id=int(coding[0]), 
                                                 doc_sel_start=int(coding[1]), 
                                                 doc_sel_end=int(coding[2]))
                else:
                    msg = _('Invalid source reference.')
                    Message(self.app, _('AI Chat'), msg, icon='critical').exec()
            elif link.startswith('chunk:'):
                try:
                    chunk_id = link[len('chunk:'):]
                    chunk_id_elem = chunk_id.split('_')
                    if len(chunk_id_elem) == 3:  # legacy format
                        source_id, start, length = chunk_id_elem
                        line_start = 0
                        line_end = 0
                    else:
                        source_id, start, length, line_start, line_end = chunk_id_elem
                    end = int(start) + int(length)
                    self.main_window.text_coding(task='documents',
                                                 doc_id=int(source_id), 
                                                 doc_sel_start=int(start), 
                                                 doc_sel_end=end)
                except Exception as e:
                    logger.debug(f'Link: "{link}" - Error: {e}')
                    source_id = None  # TODO source_id not used
                    msg = _('Invalid source reference.')
                    Message(self.app, _('AI Chat'), msg, icon='critical').exec()  
            elif link.startswith('quote:'):
                    quote_id = link[len('quote:'):]
                    source_id, start, length = quote_id.split('_')
                    end = int(start) + int(length)
                    self.main_window.text_coding(task='documents',
                                                 doc_id=int(source_id), 
                                                 doc_sel_start=int(start), 
                                                 doc_sel_end=end)
            elif link.startswith('action:topic_chat_analyze_more'):
                self.topic_chat_analyze_more()

# Helper:
class LlmCallbackHandler(BaseCallbackHandler):
    def __init__(self, dialog_ai_chat: DialogAIChat):
        self.dialog = dialog_ai_chat
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.dialog.ai_streaming_output += token
        if not self.dialog.is_updating_chat_window:
            self.dialog.update_chat_window()        
