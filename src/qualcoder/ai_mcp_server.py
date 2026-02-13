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
    default_segments_max_segments = 40
    max_segments_limit = 200
    default_segments_max_chars = 8000
    max_segments_chars_limit = 50000
    STATUS_REVIEW_AVAILABLE_MATERIALS = 'Reviewing available project materials...'
    STATUS_REVIEW_DOCUMENT_LIST = 'Reviewing the list of text documents...'
    STATUS_REVIEW_CODE_TREE = 'Reviewing the current code structure...'
    STATUS_REVIEW_JOURNAL_LIST = 'Reviewing the list of journals...'
    STATUS_REVIEW_PROJECT_MEMO = 'Reviewing the project memo...'
    STATUS_REVIEW_CODERS = 'Reviewing coder visibility settings...'
    STATUS_REVIEW_DOCUMENT = 'Reviewing text document "{name}"...'
    STATUS_REVIEW_JOURNAL = 'Reviewing journal entry "{name}"...'
    STATUS_REVIEW_RESOURCE = 'Reviewing project material...'
    STATUS_REVIEW_CODE_SEGMENTS = 'Reviewing coded text segments for "{name}"...'
    STATUS_FORMULATE_RESPONSE = 'Formulating a response based on the selected materials...'

    def __init__(self, app):
        self.app = app
        self._request_seq = 1
        self._sdk_server = Server(
            self.server_name,
            version=self.server_version,
            instructions=self._server_instructions(),
        )
        self._register_sdk_handlers()

    def _server_instructions(self) -> str:
        return (
            "QualCoder is a qualitative data analysis application used to analyze empirical material such as "
            "interviews, field notes, images, and video. It supports coding, memo writing, text annotations, "
            "researcher journaling, and reports. "
            #"Code tree invariant: codes are leaf nodes and cannot contain subcodes; only categories can contain "
            #"codes (and subcategories). "
            #"Global optional convention: speaker categories are marked by a category name prefix '📌 ' followed by "
            #"a localized label. This convention may or may not be present in a given project. "
            "This internal MCP server currently exposes read-only access to empirical documents (text only, "
            "no images or videos), codes/categories (code tree), project memo, and journals for analytic assistance. "
            "Coder visibility applies in the project. For coded segments, default retrieval uses visible coders; "
            "you can optionally request one specific coder via the owner query parameter. "
            "Use resources/list and resources/read to inspect available material. "
            "Do not assume write operations are available in this phase."
        )

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
        code_segments_match = re.fullmatch(r"qualcoder://codes/segments/(\d+)", uri_base)
        if code_segments_match is not None:
            cid = int(code_segments_match.group(1))
            code_name = self._fetch_code_name(cid)
            if code_name is None or code_name == "":
                code_name = f"Code {cid}"
            return {
                **base_event,
                "status_code": "code_segments",
                "entity_type": "code_segments",
                "entity_id": cid,
                "entity_name": code_name,
                "message_id": self.STATUS_REVIEW_CODE_SEGMENTS,
                "message_args": {"id": cid, "name": code_name},
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
        if uri_base == "qualcoder://project/coders":
            return {
                **base_event,
                "status_code": "project_coders",
                "entity_type": "project_coders",
                "message_id": self.STATUS_REVIEW_CODERS,
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
            elif message_id == self.STATUS_REVIEW_CODERS:
                translated = translator('Reviewing coder visibility settings...')
            elif message_id == self.STATUS_REVIEW_DOCUMENT:
                translated = translator('Reviewing text document "{name}"...')
            elif message_id == self.STATUS_REVIEW_JOURNAL:
                translated = translator('Reviewing journal entry "{name}"...')
            elif message_id == self.STATUS_REVIEW_RESOURCE:
                translated = translator('Reviewing project material...')
            elif message_id == self.STATUS_REVIEW_CODE_SEGMENTS:
                translated = translator('Reviewing coded text segments for "{name}"...')
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
            instructions=self._server_instructions(),
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
                types.ResourceTemplate(
                    uriTemplate="qualcoder://codes/segments/{cid}",
                    name="Coded text segments by code id",
                    description=(
                        "Read coded text segments for a code id. Optional query params: strategy "
                        "(diverse_by_document|recent_first|sequential), max_segments, max_chars, cursor, file_ids, owner."
                    ),
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
        parts = urlsplit(uri)
        uri_no_query = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
        query = parse_qs(parts.query, keep_blank_values=True)

        if uri_no_query == "qualcoder://project/summary":
            return self._project_summary()
        if uri_no_query == "qualcoder://project/memo":
            return self._project_memo()
        if uri_no_query == "qualcoder://project/coders":
            return self._project_coders()
        if uri_no_query == "qualcoder://codes/tree":
            return self._codes_tree()
        if uri_no_query == "qualcoder://documents":
            return {"documents": self._fetch_text_documents()}
        if uri_no_query == "qualcoder://journals":
            return {"journals": self._fetch_journal_entries()}

        code_segments_match = re.fullmatch(r"qualcoder://codes/segments/(\d+)", uri_no_query)
        if code_segments_match is not None:
            cid = int(code_segments_match.group(1))
            options = self._parse_code_segments_options(query)
            return self._read_code_segments(cid, options)

        doc_match = re.fullmatch(r"qualcoder://documents/text/(\d+)", uri_no_query)
        if doc_match is not None:
            doc_id = int(doc_match.group(1))
            return self._read_document(doc_id, start, req_length)

        journal_match = re.fullmatch(r"qualcoder://journals/(\d+)", uri_no_query)
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
                uri="qualcoder://project/coders",
                name="Coders",
                description="Project coders and their visibility.",
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

    def _project_coders(self) -> Dict[str, Any]:
        current_coder = ""
        if hasattr(self.app, "settings") and isinstance(self.app.settings, dict):
            current_coder = str(self.app.settings.get("codername", ""))

        all_coders = self._fetch_all_coder_names()
        visibility_map = self._fetch_coder_visibility_map()
        coders: List[Dict[str, Any]] = []
        for name in all_coders:
            coders.append(
                {
                    "name": name,
                    "visible": bool(visibility_map.get(name, True)),
                    "current": name == current_coder,
                }
            )
        coders.sort(key=lambda item: str(item.get("name", "")).casefold())
        return {
            "current_coder": current_coder,
            "coders": coders,
            "visible_coders": [c["name"] for c in coders if c["visible"]],
            "hidden_coders": [c["name"] for c in coders if not c["visible"]],
        }

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
        speaker_prefix = "📌 "
        speaker_categories = []
        for cat in categories:
            cat_name = str(cat.get("name", ""))
            if cat_name.startswith(speaker_prefix):
                speaker_categories.append({"catid": cat["catid"], "name": cat_name})

        structure_rules = {
            "codes_are_leaves": True,
            "codes_can_have_subcodes": False,
            "categories_can_contain_codes": True,
            "categories_can_have_subcategories": True,
        }
        special_conventions = {
            "speaker_category_prefix": speaker_prefix,
            "speaker_category_name_is_localized": True,
            "speaker_category_optional": True,
            "speaker_category_present": len(speaker_categories) > 0,
            "speaker_categories": speaker_categories,
            "speaker_category_ids": [item["catid"] for item in speaker_categories],
        }
        return {
            "categories": categories,
            "codes": codes,
            "structure_rules": structure_rules,
            "special_conventions": special_conventions,
        }

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

    def _fetch_code_name(self, cid: int) -> Optional[str]:
        row = self._fetchone("SELECT name FROM code_name WHERE cid=?", (cid,))
        if row is None:
            return None
        return row[0]

    def _view_exists(self, view_name: str) -> bool:
        row = self._fetchone(
            "SELECT name FROM sqlite_master WHERE type='view' AND name=?",
            (view_name,),
        )
        return row is not None

    def _to_bool(self, value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        val = str(value).strip().lower()
        if val in ("1", "true", "yes", "y", "on"):
            return True
        if val in ("0", "false", "no", "n", "off"):
            return False
        return default

    def _fetch_coder_visibility_map(self) -> Dict[str, bool]:
        try:
            rows = self._fetchall("SELECT name, visibility FROM coder_names")
        except sqlite3.Error:
            return {}

        result: Dict[str, bool] = {}
        for row in rows:
            if row[0] is None:
                continue
            coder_name = str(row[0]).strip()
            if coder_name == "":
                continue
            result[coder_name] = self._to_bool(row[1], True)
        return result

    def _fetch_all_coder_names(self) -> List[str]:
        coder_set = set()
        if hasattr(self.app, "settings") and isinstance(self.app.settings, dict):
            current = str(self.app.settings.get("codername", "")).strip()
            if current != "":
                coder_set.add(current)
        if hasattr(self.app, "get_coder_names_in_project"):
            try:
                for name in self.app.get_coder_names_in_project():
                    if name is None:
                        continue
                    text = str(name).strip()
                    if text != "":
                        coder_set.add(text)
            except Exception:
                pass
        for name in self._fetch_coder_visibility_map().keys():
            text = str(name).strip()
            if text != "":
                coder_set.add(text)
        return sorted(coder_set, key=str.casefold)

    def _resolve_coding_source(self, owner_override: Optional[str]) -> Dict[str, Any]:
        owner = None if owner_override is None else str(owner_override).strip()
        if owner is None or owner == "":
            if not self._view_exists("code_text_visible"):
                raise RuntimeError("Required view 'code_text_visible' not found.")
            return {
                "table_name": "code_text_visible",
                "owner_filter_sql": "",
                "owner_filter_params": [],
                "owner_scope": "visible",
                "owner": None,
                "visible_filter_applied": True,
            }

        known_coders = self._fetch_all_coder_names()
        if owner not in known_coders:
            raise ValueError(f"Unknown coder name: {owner}")
        return {
            "table_name": "code_text",
            "owner_filter_sql": " AND ct.owner=?",
            "owner_filter_params": [owner],
            "owner_scope": "owner_override",
            "owner": owner,
            "visible_filter_applied": False,
        }

    def _parse_code_segments_options(self, query: Dict[str, List[str]]) -> Dict[str, Any]:
        strategy = str(query.get("strategy", ["diverse_by_document"])[0]).strip()
        allowed = {"diverse_by_document", "recent_first", "sequential"}
        if strategy not in allowed:
            raise ValueError(
                "Invalid strategy. Allowed values are: diverse_by_document, recent_first, sequential."
            )

        max_segments = self._to_int(
            query.get("max_segments", [self.default_segments_max_segments])[0],
            self.default_segments_max_segments,
        )
        max_segments = max(1, min(max_segments, self.max_segments_limit))

        max_chars = self._to_int(
            query.get("max_chars", [self.default_segments_max_chars])[0],
            self.default_segments_max_chars,
        )
        max_chars = max(1, min(max_chars, self.max_segments_chars_limit))

        cursor = self._to_int(query.get("cursor", [0])[0], 0)
        cursor = max(0, cursor)

        file_ids: List[int] = []
        for raw in query.get("file_ids", []):
            parts = [p.strip() for p in str(raw).split(",")]
            for part in parts:
                if part == "":
                    continue
                file_ids.append(max(0, self._to_int(part, -1)))
        file_ids = [fid for fid in file_ids if fid > 0]
        owner = None
        owner_raw = str(query.get("owner", [""])[0]).strip()
        if owner_raw != "":
            owner = owner_raw

        return {
            "strategy": strategy,
            "max_segments": max_segments,
            "max_chars": max_chars,
            "cursor": cursor,
            "file_ids": file_ids,
            "owner": owner,
        }

    def _read_code_segments(self, cid: int, options: Dict[str, Any]) -> Dict[str, Any]:
        code_name = self._fetch_code_name(cid)
        if code_name is None:
            raise ValueError(f"Code id {cid} not found.")

        strategy = str(options.get("strategy", "diverse_by_document"))
        max_segments = int(options.get("max_segments", self.default_segments_max_segments))
        max_chars = int(options.get("max_chars", self.default_segments_max_chars))
        cursor = int(options.get("cursor", 0))
        file_ids = options.get("file_ids", [])
        owner_override = options.get("owner")
        if not isinstance(file_ids, list):
            file_ids = []

        source_cfg = self._resolve_coding_source(owner_override)
        table_name = str(source_cfg["table_name"])
        owner_filter_sql = str(source_cfg["owner_filter_sql"])
        owner_filter_params = source_cfg["owner_filter_params"]

        where_parts = ["ct.cid=?"]
        where_params: List[Any] = [cid]
        if len(file_ids) > 0:
            placeholders = ",".join(["?"] * len(file_ids))
            where_parts.append(f"ct.fid IN ({placeholders})")
            where_params.extend(file_ids)
        where_params.extend(owner_filter_params)
        where_sql = " WHERE " + " AND ".join(where_parts)
        where_sql += owner_filter_sql

        count_sql = f"SELECT count(*) FROM {table_name} AS ct" + where_sql
        count_row = self._fetchone(count_sql, tuple(where_params))
        total_segments = 0 if count_row is None else int(count_row[0])

        order_sql = "ORDER BY ct.ctid"
        select_sql = (
            "SELECT ct.ctid, ct.cid, ct.fid, ifnull(ct.seltext,''), ct.pos0, "
            "ct.pos1, ct.owner, ct.date, source.name, code_name.name "
            f"FROM {table_name} AS ct "
            "JOIN source ON source.id = ct.fid "
            "JOIN code_name ON code_name.cid = ct.cid "
            + where_sql
            + " "
        )

        if strategy == "recent_first":
            order_sql = "ORDER BY ct.date DESC, ct.ctid DESC"
            segment_rows = self._fetchall(
                select_sql + order_sql + " LIMIT ? OFFSET ?",
                tuple(where_params + [max_segments, cursor]),
            )
        elif strategy == "sequential":
            order_sql = "ORDER BY ct.ctid"
            segment_rows = self._fetchall(
                select_sql + order_sql + " LIMIT ? OFFSET ?",
                tuple(where_params + [max_segments, cursor]),
            )
        else:
            diverse_sql = (
                "SELECT ordered.ctid, ordered.cid, ordered.fid, ifnull(ordered.seltext,''), ordered.pos0, "
                "ordered.pos1, ordered.owner, ordered.date, source.name, code_name.name "
                "FROM ("
                "SELECT ct.ctid, ct.cid, ct.fid, ct.seltext, ct.pos0, ct.pos1, "
                "ct.owner, ct.date, ROW_NUMBER() OVER (PARTITION BY ct.fid ORDER BY ct.ctid) AS rn "
                f"FROM {table_name} AS ct "
                + where_sql
                + ") AS ordered "
                "JOIN source ON source.id = ordered.fid "
                "JOIN code_name ON code_name.cid = ordered.cid "
                "ORDER BY ordered.rn, ordered.fid, ordered.ctid LIMIT ? OFFSET ?"
            )
            segment_rows = self._fetchall(diverse_sql, tuple(where_params + [max_segments, cursor]))

        segments: List[Dict[str, Any]] = []
        used_chars = 0
        for row in segment_rows:
            if len(segments) >= max_segments:
                break

            quote = "" if row[3] is None else str(row[3])
            remaining_chars = max_chars - used_chars
            if remaining_chars <= 0:
                break

            quote_truncated = False
            quote_text = quote
            if len(quote) > remaining_chars:
                quote_text = quote[:remaining_chars]
                quote_truncated = True

            segments.append(
                {
                    "ctid": row[0],
                    "cid": row[1],
                    "fid": row[2],
                    "quote": quote_text,
                    "quote_truncated": quote_truncated,
                    "pos0": row[4],
                    "pos1": row[5],
                    "owner": row[6],
                    "date": row[7],
                    "source_name": row[8],
                    "code_name": row[9],
                }
            )
            used_chars += len(quote_text)
            if quote_truncated:
                break

        next_cursor = cursor + len(segments)
        if next_cursor > total_segments:
            next_cursor = total_segments
        truncated = next_cursor < total_segments

        return {
            "cid": cid,
            "code_name": code_name,
            "selection": {
                "strategy": strategy,
                "max_segments": max_segments,
                "max_chars": max_chars,
                "cursor": cursor,
                "file_ids": file_ids,
                "owner_scope": source_cfg["owner_scope"],
                "owner": source_cfg["owner"],
                "visible_filter_applied": source_cfg["visible_filter_applied"],
                "total_segments": total_segments,
                "returned_segments": len(segments),
                "returned_chars": used_chars,
                "next_cursor": next_cursor,
                "truncated": truncated,
            },
            "segments": segments,
        }

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
