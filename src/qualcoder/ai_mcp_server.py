# -*- coding: utf-8 -*-

"""
Internal MCP server for QualCoder (read-only phase).

This module uses the official MCP Python SDK (low-level server) and exposes
an in-process JSON-RPC bridge (`handle_request`) so the current chat flow can
call it without transport setup.
"""

import asyncio
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.server import ReadResourceContents


class AiMcpServer:
    """Internal read-only MCP server for QualCoder project data."""

    protocol_version = "2025-06-18"
    server_name = "qualcoder-internal-mcp"
    server_version = "0.1.0"
    max_read_length = 12000
    default_read_length = 4000
    STATUS_REVIEW_AVAILABLE_MATERIALS = 'Reviewing available project materials...'
    STATUS_REVIEW_DOCUMENT_LIST = 'Reviewing the list of text documents...'
    STATUS_REVIEW_CODE_TREE = 'Reviewing the current code structure...'
    STATUS_REVIEW_JOURNAL_LIST = 'Reviewing the list of journals...'
    STATUS_REVIEW_PROJECT_MEMO = 'Reviewing the project memo...'
    STATUS_REVIEW_DOCUMENT = 'Reviewing text document "{name}"...'
    STATUS_REVIEW_JOURNAL = 'Reviewing journal entry "{name}"...'
    STATUS_REVIEW_RESOURCE = 'Reviewing project material...'
    STATUS_FORMULATE_RESPONSE = 'Formulating a response based on the selected materials...'

    def __init__(self, app):
        self.app = app
        self._request_seq = 1
        self._sdk_server = Server(
            self.server_name,
            version=self.server_version,
            instructions=(
                "This QualCoder internal MCP server is read-only in the current phase. "
                "Use resources/list and resources/read for project data."
            ),
        )
        self._register_sdk_handlers()

    def new_request_id(self) -> int:
        req_id = self._request_seq
        self._request_seq += 1
        return req_id

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC request and return a JSON-RPC response."""

        request_id = request.get("id")
        jsonrpc = request.get("jsonrpc", "2.0")
        method = request.get("method")
        params = request.get("params", {})

        if jsonrpc != "2.0":
            return self._error_response(request_id, -32600, "Invalid Request", "jsonrpc must be '2.0'.")
        if not isinstance(method, str) or method.strip() == "":
            return self._error_response(request_id, -32600, "Invalid Request", "Missing method.")
        if not isinstance(params, dict):
            return self._error_response(request_id, -32602, "Invalid params", "params must be an object.")

        try:
            if method == "initialize":
                result = self._initialize_result()
            elif method == "resources/list":
                req = types.ListResourcesRequest(params=self._pagination_params(params))
                result = self._dispatch_sdk(types.ListResourcesRequest, req)
            elif method == "resources/templates/list":
                req = types.ListResourceTemplatesRequest(params=self._pagination_params(params))
                result = self._dispatch_sdk(types.ListResourceTemplatesRequest, req)
            elif method == "resources/read":
                uri = params.get("uri")
                if not isinstance(uri, str) or uri.strip() == "":
                    raise ValueError("Missing resource uri.")
                uri_with_window = self._with_read_window(uri, params.get("start"), params.get("length"))
                req = types.ReadResourceRequest(params=types.ReadResourceRequestParams(uri=uri_with_window))
                result = self._dispatch_sdk(types.ReadResourceRequest, req)
            elif method == "tools/list":
                req = types.ListToolsRequest(params=self._pagination_params(params))
                result = self._dispatch_sdk(types.ListToolsRequest, req)
            elif method == "tools/call":
                name = str(params.get("name", "")).strip()
                if name == "":
                    raise ValueError("Missing tool name.")
                args = params.get("arguments")
                if args is not None and not isinstance(args, dict):
                    raise ValueError("Tool arguments must be an object.")
                req = types.CallToolRequest(params=types.CallToolRequestParams(name=name, arguments=args))
                result = self._dispatch_sdk(types.CallToolRequest, req)
            elif method == "prompts/list":
                req = types.ListPromptsRequest(params=self._pagination_params(params))
                result = self._dispatch_sdk(types.ListPromptsRequest, req)
            elif method == "prompts/get":
                name = str(params.get("name", "")).strip()
                if name == "":
                    raise ValueError("Missing prompt name.")
                args = params.get("arguments")
                if args is not None and not isinstance(args, dict):
                    raise ValueError("Prompt arguments must be an object.")
                req = types.GetPromptRequest(params=types.GetPromptRequestParams(name=name, arguments=args))
                result = self._dispatch_sdk(types.GetPromptRequest, req)
            else:
                return self._error_response(request_id, -32601, "Method not found", method)
            return self._result_response(request_id, result)
        except ValueError as err:
            return self._error_response(request_id, -32602, "Invalid params", str(err))
        except RuntimeError as err:
            return self._error_response(request_id, -32000, "Runtime error", str(err))
        except Exception as err:
            return self._error_response(request_id, -32603, "Internal error", str(err))

    def describe_status_event(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Describe a user-facing status event for a request, if applicable."""

        if method == "initialize":
            return None
        if method == "resources/templates/list":
            return None
        if method == "resources/list":
            return {
                "status_code": "resources_list",
                "phase": "start",
                "method": method,
                "message_id": self.STATUS_REVIEW_AVAILABLE_MATERIALS,
                "message_args": {},
            }
        if method != "resources/read":
            return None

        uri = str(params.get("uri", ""))
        uri_base = uri.split("?", 1)[0]
        base_event = {
            "phase": "start",
            "method": method,
            "uri": uri_base,
            "message_args": {},
        }

        if uri_base == "qualcoder://documents":
            return {
                **base_event,
                "status_code": "documents_list",
                "entity_type": "documents",
                "message_id": self.STATUS_REVIEW_DOCUMENT_LIST,
            }
        if uri_base == "qualcoder://codes/tree":
            return {
                **base_event,
                "status_code": "codes_tree",
                "entity_type": "codes_tree",
                "message_id": self.STATUS_REVIEW_CODE_TREE,
            }
        if uri_base == "qualcoder://journals":
            return {
                **base_event,
                "status_code": "journals_list",
                "entity_type": "journals",
                "message_id": self.STATUS_REVIEW_JOURNAL_LIST,
            }
        if uri_base == "qualcoder://project/memo":
            return {
                **base_event,
                "status_code": "project_memo",
                "entity_type": "project_memo",
                "message_id": self.STATUS_REVIEW_PROJECT_MEMO,
            }

        doc_match = re.fullmatch(r"qualcoder://documents/text/(\d+)", uri_base)
        if doc_match is not None:
            doc_id = int(doc_match.group(1))
            doc_name = self._fetch_source_name(doc_id)
            if doc_name is None or doc_name == "":
                doc_name = f"Document {doc_id}"
            return {
                **base_event,
                "status_code": "document_read",
                "entity_type": "document",
                "entity_id": doc_id,
                "entity_name": doc_name,
                "message_id": self.STATUS_REVIEW_DOCUMENT,
                "message_args": {"id": doc_id, "name": doc_name},
            }

        journal_match = re.fullmatch(r"qualcoder://journals/(\d+)", uri_base)
        if journal_match is not None:
            jid = int(journal_match.group(1))
            journal_name = self._fetch_journal_name(jid)
            if journal_name is None or journal_name == "":
                journal_name = f"Journal {jid}"
            return {
                **base_event,
                "status_code": "journal_read",
                "entity_type": "journal",
                "entity_id": jid,
                "entity_name": journal_name,
                "message_id": self.STATUS_REVIEW_JOURNAL,
                "message_args": {"id": jid, "name": journal_name},
            }

        return {
            **base_event,
            "status_code": "resource_read",
            "entity_type": "resource",
            "message_id": self.STATUS_REVIEW_RESOURCE,
        }

    def describe_host_status_event(self, phase: str) -> Optional[Dict[str, Any]]:
        """Describe non-MCP but related host-side status events."""

        if phase == "final_response":
            return {
                "status_code": "final_response",
                "phase": "start",
                "message_id": self.STATUS_FORMULATE_RESPONSE,
                "message_args": {},
            }
        return None

    def status_event_to_text(self, status_event: Optional[Dict[str, Any]]) -> str:
        """Convert a status event into user-facing text, translated if gettext is available."""

        if status_event is None:
            return ""
        message_id = str(status_event.get("message_id", "")).strip()
        if message_id == "":
            return ""
        message_args = status_event.get("message_args", {})
        if not isinstance(message_args, dict):
            message_args = {}
        translator = globals().get("_", None)
        if translator is None:
            translated = message_id
        else:
            # Keep literal msgids in _() calls so gettext extraction can discover them.
            if message_id == self.STATUS_REVIEW_AVAILABLE_MATERIALS:
                translated = translator('Reviewing available project materials...')
            elif message_id == self.STATUS_REVIEW_DOCUMENT_LIST:
                translated = translator('Reviewing the list of text documents...')
            elif message_id == self.STATUS_REVIEW_CODE_TREE:
                translated = translator('Reviewing the current code structure...')
            elif message_id == self.STATUS_REVIEW_JOURNAL_LIST:
                translated = translator('Reviewing the list of journals...')
            elif message_id == self.STATUS_REVIEW_PROJECT_MEMO:
                translated = translator('Reviewing the project memo...')
            elif message_id == self.STATUS_REVIEW_DOCUMENT:
                translated = translator('Reviewing text document "{name}"...')
            elif message_id == self.STATUS_REVIEW_JOURNAL:
                translated = translator('Reviewing journal entry "{name}"...')
            elif message_id == self.STATUS_REVIEW_RESOURCE:
                translated = translator('Reviewing project material...')
            elif message_id == self.STATUS_FORMULATE_RESPONSE:
                translated = translator('Formulating a response based on the selected materials...')
            else:
                translated = translator(message_id)
        try:
            return translated.format(**message_args)
        except Exception:
            return translated

    def _initialize_result(self) -> Dict[str, Any]:
        result = types.InitializeResult(
            protocolVersion=self.protocol_version,
            capabilities=types.ServerCapabilities(
                resources=types.ResourcesCapability(subscribe=False, listChanged=False),
                tools=types.ToolsCapability(listChanged=False),
                prompts=types.PromptsCapability(listChanged=False),
            ),
            serverInfo=types.Implementation(name=self.server_name, version=self.server_version),
            instructions=(
                "This QualCoder internal MCP server is read-only in the current phase. "
                "Use resources/list and resources/read for project data."
            ),
        )
        return result.model_dump(mode="json", exclude_none=True)

    def _dispatch_sdk(self, req_type: type, req_obj: Any) -> Dict[str, Any]:
        handler = self._sdk_server.request_handlers.get(req_type)
        if handler is None:
            raise RuntimeError(f"MCP handler not registered for {req_type.__name__}.")
        server_result = asyncio.run(handler(req_obj))
        if hasattr(server_result, "model_dump"):
            return server_result.model_dump(mode="json", exclude_none=True)
        return dict(server_result)

    def _pagination_params(self, params: Dict[str, Any]) -> Optional[types.PaginatedRequestParams]:
        cursor = params.get("cursor")
        if cursor is None:
            return None
        return types.PaginatedRequestParams(cursor=str(cursor))

    def _with_read_window(self, uri: str, start: Any, length: Any) -> str:
        if start is None and length is None:
            return uri
        parts = urlsplit(uri)
        query = parse_qs(parts.query, keep_blank_values=True)
        if start is not None:
            start_i = max(0, self._to_int(start, 0))
            query["start"] = [str(start_i)]
        if length is not None:
            length_i = max(1, min(self._to_int(length, self.default_read_length), self.max_read_length))
            query["length"] = [str(length_i)]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment))

    def _parse_read_window(self, uri: str) -> Tuple[str, int, int]:
        parts = urlsplit(uri)
        query = parse_qs(parts.query, keep_blank_values=True)
        start = max(0, self._to_int(query.get("start", [0])[0], 0))
        length = max(
            1,
            min(
                self._to_int(query.get("length", [self.default_read_length])[0], self.default_read_length),
                self.max_read_length,
            ),
        )
        query.pop("start", None)
        query.pop("length", None)
        base_uri = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment))
        return base_uri, start, length

    def _register_sdk_handlers(self) -> None:
        @self._sdk_server.list_resources()
        async def _list_resources(_: types.ListResourcesRequest) -> types.ListResourcesResult:
            return types.ListResourcesResult(resources=self._base_resources())

        @self._sdk_server.list_resource_templates()
        async def _list_resource_templates() -> List[types.ResourceTemplate]:
            return [
                types.ResourceTemplate(
                    uriTemplate="qualcoder://documents/text/{id}",
                    name="Document by id",
                    description="Read a text document by source id.",
                    mimeType="application/json",
                ),
                types.ResourceTemplate(
                    uriTemplate="qualcoder://journals/{jid}",
                    name="Journal by id",
                    description="Read a journal entry by journal id.",
                    mimeType="application/json",
                ),
            ]

        @self._sdk_server.read_resource()
        async def _read_resource(uri: str) -> List[ReadResourceContents]:
            uri_str = str(uri)
            base_uri, start, req_length = self._parse_read_window(uri_str)
            payload = self._read_resource_payload(base_uri, start, req_length)
            return [ReadResourceContents(content=json.dumps(payload, ensure_ascii=False), mime_type="application/json")]

        @self._sdk_server.list_tools()
        async def _list_tools(_: types.ListToolsRequest) -> types.ListToolsResult:
            return types.ListToolsResult(tools=[])

        @self._sdk_server.call_tool()
        async def _call_tool(_name: str, _arguments: Dict[str, Any]) -> Dict[str, Any]:
            raise RuntimeError("No MCP tools are enabled yet. Current phase is read-only resources.")

        @self._sdk_server.list_prompts()
        async def _list_prompts(_: types.ListPromptsRequest) -> types.ListPromptsResult:
            return types.ListPromptsResult(prompts=[])

        @self._sdk_server.get_prompt()
        async def _get_prompt(_name: str, _arguments: Optional[Dict[str, str]]) -> types.GetPromptResult:
            raise RuntimeError("No MCP prompts are exposed yet.")

    def _read_resource_payload(self, uri: str, start: int, req_length: int) -> Dict[str, Any]:
        if uri == "qualcoder://project/summary":
            return self._project_summary()
        if uri == "qualcoder://project/memo":
            return self._project_memo()
        if uri == "qualcoder://codes/tree":
            return self._codes_tree()
        if uri == "qualcoder://documents":
            return {"documents": self._fetch_text_documents()}
        if uri == "qualcoder://journals":
            return {"journals": self._fetch_journal_entries()}

        doc_match = re.fullmatch(r"qualcoder://documents/text/(\d+)", uri)
        if doc_match is not None:
            doc_id = int(doc_match.group(1))
            return self._read_document(doc_id, start, req_length)

        journal_match = re.fullmatch(r"qualcoder://journals/(\d+)", uri)
        if journal_match is not None:
            jid = int(journal_match.group(1))
            return self._read_journal(jid, start, req_length)

        raise ValueError(f"Unknown resource uri: {uri}")

    def _result_response(self, request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _error_response(self, request_id: Any, code: int, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": request_id, "error": error}

    def _base_resources(self) -> List[types.Resource]:
        return [
            types.Resource(
                uri="qualcoder://project/summary",
                name="Project summary",
                description="Current project name and coder context.",
                mimeType="application/json",
            ),
            types.Resource(
                uri="qualcoder://project/memo",
                name="Project memo",
                description="Full project memo text.",
                mimeType="application/json",
            ),
            types.Resource(
                uri="qualcoder://codes/tree",
                name="Codes and categories",
                description="Code tree with categories and code metadata.",
                mimeType="application/json",
            ),
            types.Resource(
                uri="qualcoder://documents",
                name="Text documents",
                description="List text documents in the project.",
                mimeType="application/json",
            ),
            types.Resource(
                uri="qualcoder://journals",
                name="Journals",
                description="List journal entries in the project.",
                mimeType="application/json",
            ),
        ]

    def _project_summary(self) -> Dict[str, Any]:
        coder_name = ""
        if hasattr(self.app, "settings") and isinstance(self.app.settings, dict):
            coder_name = str(self.app.settings.get("codername", ""))
        return {
            "project_name": getattr(self.app, "project_name", ""),
            "project_path": getattr(self.app, "project_path", ""),
            "codername": coder_name,
        }

    def _project_memo(self) -> Dict[str, Any]:
        row = self._fetchone("SELECT ifnull(memo,'') FROM project")
        memo = "" if row is None else row[0]
        return {"memo": memo}

    def _codes_tree(self) -> Dict[str, Any]:
        categories = []
        for row in self._fetchall(
            "SELECT catid, name, ifnull(memo,''), owner, date, supercatid "
            "FROM code_cat ORDER BY lower(name)"
        ):
            categories.append(
                {
                    "catid": row[0],
                    "name": row[1],
                    "memo": row[2],
                    "owner": row[3],
                    "date": row[4],
                    "supercatid": row[5],
                }
            )

        codes = []
        for row in self._fetchall(
            "SELECT cid, name, ifnull(memo,''), catid, color, owner, date "
            "FROM code_name ORDER BY lower(name)"
        ):
            codes.append(
                {
                    "cid": row[0],
                    "name": row[1],
                    "memo": row[2],
                    "catid": row[3],
                    "color": row[4],
                    "owner": row[5],
                    "date": row[6],
                }
            )
        return {"categories": categories, "codes": codes}

    def _fetch_text_documents(self) -> List[Dict[str, Any]]:
        docs = []
        for row in self._fetchall(
            "SELECT id, name, ifnull(memo,''), owner, date, ifnull(length(fulltext),0) "
            "FROM source WHERE fulltext is not null ORDER BY lower(name)"
        ):
            docs.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "memo": row[2],
                    "owner": row[3],
                    "date": row[4],
                    "length": row[5],
                }
            )
        return docs

    def _fetch_journal_entries(self) -> List[Dict[str, Any]]:
        journals = []
        for row in self._fetchall(
            "SELECT jid, name, owner, date, ifnull(length(jentry),0) FROM journal ORDER BY date desc"
        ):
            journals.append(
                {
                    "jid": row[0],
                    "name": row[1],
                    "owner": row[2],
                    "date": row[3],
                    "length": row[4],
                }
            )
        return journals

    def _fetch_source_name(self, doc_id: int) -> Optional[str]:
        row = self._fetchone("SELECT name FROM source WHERE id=?", (doc_id,))
        if row is None:
            return None
        return row[0]

    def _fetch_journal_name(self, jid: int) -> Optional[str]:
        row = self._fetchone("SELECT name FROM journal WHERE jid=?", (jid,))
        if row is None:
            return None
        return row[0]

    def _read_document(self, doc_id: int, start: int, length: int) -> Dict[str, Any]:
        row = self._fetchone(
            "SELECT id, name, ifnull(memo,''), owner, date, ifnull(fulltext,'') "
            "FROM source WHERE id=? AND fulltext is not null",
            (doc_id,),
        )
        if row is None:
            raise ValueError(f"Document id {doc_id} not found.")
        fulltext = row[5]
        end_pos = min(start + length, len(fulltext))
        excerpt = fulltext[start:end_pos]
        return {
            "id": row[0],
            "name": row[1],
            "memo": row[2],
            "owner": row[3],
            "date": row[4],
            "total_length": len(fulltext),
            "start": start,
            "length": len(excerpt),
            "text": excerpt,
        }

    def _read_journal(self, jid: int, start: int, length: int) -> Dict[str, Any]:
        row = self._fetchone(
            "SELECT jid, name, owner, date, ifnull(jentry,'') FROM journal WHERE jid=?",
            (jid,),
        )
        if row is None:
            raise ValueError(f"Journal id {jid} not found.")
        jentry = row[4]
        end_pos = min(start + length, len(jentry))
        excerpt = jentry[start:end_pos]
        return {
            "jid": row[0],
            "name": row[1],
            "owner": row[2],
            "date": row[3],
            "total_length": len(jentry),
            "start": start,
            "length": len(excerpt),
            "text": excerpt,
        }

    def _to_int(self, value: Any, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _project_db_path(self) -> str:
        project_path = getattr(self.app, "project_path", "")
        if project_path is None or project_path == "":
            raise RuntimeError("No project open.")
        db_path = os.path.join(project_path, "data.qda")
        if not os.path.exists(db_path):
            raise RuntimeError("Project database not found.")
        return db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._project_db_path())

    def _fetchall(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Tuple[Any, ...]]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return cur.fetchall()
        finally:
            conn.close()

    def _fetchone(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Tuple[Any, ...]]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return cur.fetchone()
        finally:
            conn.close()
