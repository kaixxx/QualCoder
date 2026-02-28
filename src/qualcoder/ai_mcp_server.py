# -*- coding: utf-8 -*-

"""
Internal MCP server for QualCoder.

This module uses the official MCP Python SDK (low-level server) and exposes
an in-process JSON-RPC bridge (`handle_request`) so the current chat flow can
call it without transport setup.
"""

import asyncio
import hashlib
import json
import os
import random
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.server import ReadResourceContents
from .ai_skills import AiSkillsCatalog


class AiMcpServer:
    """Internal MCP server for QualCoder project data."""

    protocol_version = "2025-06-18"
    server_name = "qualcoder-internal-mcp"
    server_version = "0.1.0"
    max_read_length = 12000
    default_read_length = 4000
    default_segments_max_segments = 40
    max_segments_limit = 200
    default_segments_max_chars = 8000
    max_segments_chars_limit = 50000
    default_vector_page_size = 20
    max_vector_page_size = 100
    default_vector_k_per_query = 50
    max_vector_k_per_query = 200
    default_vector_score_threshold = 0.5
    default_regex_page_size = 20
    max_regex_page_size = 100
    default_regex_context_chars = 120
    max_regex_context_chars = 1000
    max_regex_hits = 20000
    AI_AGENT_OWNER = "AI Agent"

    def __init__(self, app):
        self.app = app
        self.skills_catalog = AiSkillsCatalog(app)
        self._request_seq = 1
        self._sdk_server = Server(
            self.server_name,
            version=self.server_version,
            instructions=self._server_instructions(),
        )
        self._register_sdk_handlers()

    def _server_instructions(self) -> str:
        return (
            "QualCoder internal MCP server. "
            "Use resources/list, resources/read, prompts/list, prompts/get, tools/list, and tools/call. "
            "Available resources: text documents list (qualcoder://documents), document text by id "
            "(qualcoder://documents/text/{id}), code tree (qualcoder://codes/tree), and coded text segments by code id "
            "(qualcoder://codes/segments/{cid}), semantic vector search "
            "(qualcoder://vector/search?q=...), and regular-expression search "
            "(qualcoder://search/regex?pattern=...). "
            "Available write tools: create category, create code, and create text coding. "
            "Available prompts represent QualCoder skills from system, user, and project scope."
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
                result = self._list_tools_payload()
            elif method == "tools/call":
                name = str(params.get("name", "")).strip()
                if name == "":
                    raise ValueError("Missing tool name.")
                args = params.get("arguments")
                if args is not None and not isinstance(args, dict):
                    raise ValueError("Tool arguments must be an object.")
                change_set_id = str(params.get("_ai_change_set_id", "")).strip()
                result = self._call_tool_payload(name, args, change_set_id)
            elif method == "prompts/list":
                result = self._list_prompts_payload()
            elif method == "prompts/get":
                name = str(params.get("name", "")).strip()
                if name == "":
                    raise ValueError("Missing prompt name.")
                args = params.get("arguments")
                if args is not None and not isinstance(args, dict):
                    raise ValueError("Prompt arguments must be an object.")
                result = self._get_prompt_payload(name, args)
            else:
                return self._error_response(request_id, -32601, "Method not found", method)
            return self._result_response(request_id, result)
        except ValueError as err:
            return self._error_response(request_id, -32602, "Invalid params", str(err))
        except RuntimeError as err:
            return self._error_response(request_id, -32000, "Runtime error", str(err))
        except Exception as err:
            return self._error_response(request_id, -32603, "Internal error", str(err))

    def describe_status_event(self, method: str, params: Dict[str, Any]) -> str:
        """Describe one user-facing status line for a request, if applicable."""

        if method == "initialize":
            return ""
        if method == "resources/templates/list":
            return ""
        if method == "resources/list":
            return ""
        if method == "tools/list":
            return ""
        if method == "tools/call":
            tool_name = str(params.get("name", "")).strip()
            if tool_name == "":
                return ""
            tool_args = params.get("arguments", {})
            if not isinstance(tool_args, dict):
                tool_args = {}
            if tool_name == "codes/create_category":
                cat_name = " ".join(str(tool_args.get("name", "")).split()).strip()
                if cat_name == "":
                    cat_name = _("(unnamed category)")
                return _('Creating category "{name}"...').format(name=cat_name)
            if tool_name == "codes/create_code":
                code_name = " ".join(str(tool_args.get("name", "")).split()).strip()
                if code_name == "":
                    code_name = _("(unnamed code)")
                return _('Creating code "{name}"...').format(name=code_name)
            if tool_name == "codes/create_text_coding":
                cid = self._to_int(tool_args.get("cid"), -1)
                fid = self._to_int(tool_args.get("fid"), -1)
                code_name = self._fetch_code_name(cid) if cid > 0 else None
                if code_name is None or str(code_name).strip() == "":
                    code_name = _("Code") + (f" #{cid}" if cid > 0 else "")
                doc_name = self._fetch_source_name(fid) if fid > 0 else None
                if doc_name is None or str(doc_name).strip() == "":
                    doc_name = _("Document") + (f" #{fid}" if fid > 0 else "")
                return _('Creating text coding for code "{code}" in document "{document}"...').format(
                    code=str(code_name),
                    document=str(doc_name),
                )
            return _('Executing tool "{name}"...').format(name=tool_name)
        if method != "resources/read":
            return ""

        uri = str(params.get("uri", ""))
        uri_base = uri.split("?", 1)[0]

        if uri_base == "qualcoder://documents":
            return _('Reviewing the list of text documents...')
        if uri_base == "qualcoder://codes/tree":
            return _('Reviewing the current code structure...')
        if uri_base == "qualcoder://vector/search":
            return _('Running semantic search in the project data...')
        if uri_base == "qualcoder://search/regex":
            return _('Running keyword search in the project data...')
        code_segments_match = re.fullmatch(r"qualcoder://codes/segments/(\d+)", uri_base)
        if code_segments_match is not None:
            cid = int(code_segments_match.group(1))
            code_name = self._fetch_code_name(cid)
            if code_name is None or code_name == "":
                code_name = f"Code {cid}"
            return _('Reviewing coded text segments for "{name}"...').format(name=code_name)
        doc_match = re.fullmatch(r"qualcoder://documents/text/(\d+)", uri_base)
        if doc_match is not None:
            doc_id = int(doc_match.group(1))
            doc_name = self._fetch_source_name(doc_id)
            if doc_name is None or doc_name == "":
                doc_name = f"Document {doc_id}"
            return _('Reviewing text document "{name}"...').format(name=doc_name)

        return _('Reviewing project material...')

    def describe_host_status_event(self, phase: str) -> str:
        """Describe non-MCP but related host-side status lines."""

        if phase == "planning":
            return _('Planning how to gather project evidence...')
        if phase == "execution":
            return _('Executing MCP retrieval steps...')
        if phase == "reflection":
            return _('Reflecting on retrieved evidence and revising the plan...')
        if phase == "final_response":
            return _('Formulating a response based on the selected materials...')
        return ""

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
                    uriTemplate="qualcoder://codes/segments/{cid}",
                    name="Coded text segments by code id",
                    description=(
                        "Read coded text segments for a code id. Optional query params: strategy "
                        "(diverse_by_document|recent_first|sequential), max_segments, max_chars, cursor, file_ids."
                    ),
                    mimeType="application/json",
                ),
                types.ResourceTemplate(
                    uriTemplate="qualcoder://vector/search{?q,cursor,page_size,file_ids,score_threshold,k_per_query}",
                    name="Semantic vector search",
                    description=(
                        "Search semantically similar text chunks. Pass one or more q parameters "
                        "(for example ?q=work&q=employment). Optional params: cursor, page_size, "
                        "file_ids, score_threshold, k_per_query."
                    ),
                    mimeType="application/json",
                ),
                types.ResourceTemplate(
                    uriTemplate="qualcoder://search/regex{?pattern,flags,cursor,page_size,file_ids,context_chars}",
                    name="Regular-expression search",
                    description=(
                        "Search text documents using a regular expression pattern. "
                        "Optional params: flags (imsx), cursor, page_size, file_ids, context_chars."
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
            return types.ListToolsResult.model_validate(self._list_tools_payload())

        @self._sdk_server.call_tool()
        async def _call_tool(_name: str, _arguments: Dict[str, Any]) -> Dict[str, Any]:
            return self._call_tool_payload(_name, _arguments, "")

        @self._sdk_server.list_prompts()
        async def _list_prompts(_: types.ListPromptsRequest) -> types.ListPromptsResult:
            return types.ListPromptsResult.model_validate(self._list_prompts_payload())

        @self._sdk_server.get_prompt()
        async def _get_prompt(_name: str, _arguments: Optional[Dict[str, str]]) -> types.GetPromptResult:
            return types.GetPromptResult.model_validate(self._get_prompt_payload(_name, _arguments))

    def _read_resource_payload(self, uri: str, start: int, req_length: int) -> Dict[str, Any]:
        parts = urlsplit(uri)
        uri_no_query = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
        query = parse_qs(parts.query, keep_blank_values=True)

        if uri_no_query == "qualcoder://codes/tree":
            return self._codes_tree()
        if uri_no_query == "qualcoder://documents":
            return {"documents": self._fetch_text_documents()}
        if uri_no_query == "qualcoder://vector/search":
            options = self._parse_vector_search_options(query)
            return self._read_vector_search(options)
        if uri_no_query == "qualcoder://search/regex":
            options = self._parse_regex_search_options(query)
            return self._read_regex_search(options)

        code_segments_match = re.fullmatch(r"qualcoder://codes/segments/(\d+)", uri_no_query)
        if code_segments_match is not None:
            cid = int(code_segments_match.group(1))
            options = self._parse_code_segments_options(query)
            return self._read_code_segments(cid, options)

        doc_match = re.fullmatch(r"qualcoder://documents/text/(\d+)", uri_no_query)
        if doc_match is not None:
            doc_id = int(doc_match.group(1))
            return self._read_document(doc_id, start, req_length)

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
                uri="qualcoder://vector/search",
                name="Semantic vector search",
                description="Semantic retrieval over embedded text chunks. Requires query param q.",
                mimeType="application/json",
            ),
            types.Resource(
                uri="qualcoder://search/regex",
                name="Regular-expression search",
                description="Regex keyword search over text documents. Requires query param pattern.",
                mimeType="application/json",
            ),
        ]

    def _list_prompts_payload(self) -> Dict[str, Any]:
        prompts: List[Dict[str, Any]] = []
        for skill in self.skills_catalog.list_skills():
            description = str(skill.description).strip()
            if description == "":
                description = _("No description provided.")
            prompts.append(
                {
                    "name": skill.skill_id,
                    "description": f"[{skill.scope}] {description}",
                }
            )
        return {"prompts": prompts}

    def _list_tools_payload(self) -> Dict[str, Any]:
        return {
            "tools": [
                {
                    "name": "codes/create_category",
                    "description": (
                        "Create a new code category. Use this only when explicitly needed by the user request."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "memo": {"type": "string"},
                            "supercatid": {"type": ["integer", "null"]},
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "codes/create_code",
                    "description": (
                        "Create a new code. Use catid to assign the code to a category, or null for top-level."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "memo": {"type": "string"},
                            "catid": {"type": ["integer", "null"]},
                            "color": {"type": "string"},
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "codes/create_text_coding",
                    "description": (
                        "Create one text coding by code id and quoted text in a text document."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "cid": {"type": "integer"},
                            "fid": {"type": "integer"},
                            "quote": {"type": "string"},
                            "memo": {"type": "string"},
                        },
                        "required": ["cid", "fid", "quote"],
                        "additionalProperties": False,
                    },
                },
            ]
        }

    def _get_prompt_payload(self, name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if arguments is not None and not isinstance(arguments, dict):
            raise ValueError("Prompt arguments must be an object.")
        skill = self.skills_catalog.get_skill(name)
        if skill is None:
            raise RuntimeError(f"Prompt not found: {name}")

        description = str(skill.description).strip()
        if description == "":
            description = _("No description provided.")
        content = str(skill.content).strip()
        if content == "":
            content = _("(empty skill)")

        return {
            "description": f"[{skill.scope}] {description}",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": content,
                    },
                }
            ],
        }

    def _call_tool_payload(self, name: str, arguments: Optional[Dict[str, Any]], change_set_id: str) -> Dict[str, Any]:
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object.")
        tool_name = str(name).strip()
        if tool_name == "":
            raise ValueError("Missing tool name.")

        if tool_name == "codes/create_category":
            payload = self._tool_create_category(arguments, change_set_id)
        elif tool_name == "codes/create_code":
            payload = self._tool_create_code(arguments, change_set_id)
        elif tool_name == "codes/create_text_coding":
            payload = self._tool_create_text_coding(arguments, change_set_id)
        else:
            raise ValueError(f"Unknown tool name: {tool_name}")

        return {
            "isError": False,
            "structuredContent": payload,
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False),
                }
            ],
        }

    def _tool_create_category(self, arguments: Dict[str, Any], change_set_id: str) -> Dict[str, Any]:
        name = " ".join(str(arguments.get("name", "")).split()).strip()
        if name == "":
            raise ValueError("Category name must not be empty.")
        memo = str(arguments.get("memo", "") if arguments.get("memo", "") is not None else "")
        supercatid_raw = arguments.get("supercatid", None)
        supercatid = None
        if supercatid_raw is not None:
            supercatid = self._to_int(supercatid_raw, -1)
            if supercatid <= 0:
                raise ValueError("supercatid must be a positive integer or null.")

        conn = self._connect()
        try:
            cur = conn.cursor()
            now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
            if supercatid is not None:
                row = cur.execute("SELECT catid FROM code_cat WHERE catid=?", (supercatid,)).fetchone()
                if row is None:
                    raise ValueError(f"Parent category id {supercatid} not found.")

            existing = cur.execute(
                "SELECT catid, owner, ifnull(memo,''), supercatid FROM code_cat WHERE lower(name)=lower(?)",
                (name,),
            ).fetchone()
            if existing is not None:
                return {
                    "tool": "codes/create_category",
                    "created": False,
                    "reason": "already_exists",
                    "category": {
                        "catid": int(existing[0]),
                        "name": name,
                        "owner": existing[1],
                        "memo": existing[2],
                        "supercatid": existing[3],
                    },
                }

            cur.execute(
                "INSERT INTO code_cat (name, memo, owner, date, supercatid) VALUES (?, ?, ?, ?, ?)",
                (name, memo, self.AI_AGENT_OWNER, now, supercatid),
            )
            catid = int(cur.lastrowid)
            conn.commit()
            if hasattr(self.app, "delete_backup"):
                self.app.delete_backup = False

            self._record_ai_change(
                change_set_id,
                {
                    "type": "create_category",
                    "catid": catid,
                    "name": name,
                    "memo": memo,
                    "supercatid": supercatid,
                    "owner": self.AI_AGENT_OWNER,
                    "created_at": now,
                },
            )
            return {
                "tool": "codes/create_category",
                "created": True,
                "category": {
                    "catid": catid,
                    "name": name,
                    "memo": memo,
                    "owner": self.AI_AGENT_OWNER,
                    "date": now,
                    "supercatid": supercatid,
                },
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _tool_create_code(self, arguments: Dict[str, Any], change_set_id: str) -> Dict[str, Any]:
        name = " ".join(str(arguments.get("name", "")).split()).strip()
        if name == "":
            raise ValueError("Code name must not be empty.")
        memo = str(arguments.get("memo", "") if arguments.get("memo", "") is not None else "")
        catid_raw = arguments.get("catid", None)
        catid = None
        if catid_raw is not None:
            catid = self._to_int(catid_raw, -1)
            if catid <= 0:
                raise ValueError("catid must be a positive integer or null.")
        color = self._normalize_hex_color(arguments.get("color"))
        if color == "":
            color = "#8A8A8A"

        conn = self._connect()
        try:
            cur = conn.cursor()
            now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
            if catid is not None:
                row = cur.execute("SELECT catid FROM code_cat WHERE catid=?", (catid,)).fetchone()
                if row is None:
                    raise ValueError(f"Category id {catid} not found.")

            existing = cur.execute(
                "SELECT cid, owner, ifnull(memo,''), catid, color FROM code_name WHERE lower(name)=lower(?)",
                (name,),
            ).fetchone()
            if existing is not None:
                return {
                    "tool": "codes/create_code",
                    "created": False,
                    "reason": "already_exists",
                    "code": {
                        "cid": int(existing[0]),
                        "name": name,
                        "owner": existing[1],
                        "memo": existing[2],
                        "catid": existing[3],
                        "color": existing[4],
                    },
                }

            cur.execute(
                "INSERT INTO code_name (name, memo, catid, owner, date, color) VALUES (?, ?, ?, ?, ?, ?)",
                (name, memo, catid, self.AI_AGENT_OWNER, now, color),
            )
            cid = int(cur.lastrowid)
            conn.commit()
            if hasattr(self.app, "delete_backup"):
                self.app.delete_backup = False

            self._record_ai_change(
                change_set_id,
                {
                    "type": "create_code",
                    "cid": cid,
                    "name": name,
                    "memo": memo,
                    "catid": catid,
                    "color": color,
                    "owner": self.AI_AGENT_OWNER,
                    "created_at": now,
                },
            )
            return {
                "tool": "codes/create_code",
                "created": True,
                "code": {
                    "cid": cid,
                    "name": name,
                    "memo": memo,
                    "catid": catid,
                    "color": color,
                    "owner": self.AI_AGENT_OWNER,
                    "date": now,
                },
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _tool_create_text_coding(self, arguments: Dict[str, Any], change_set_id: str) -> Dict[str, Any]:
        cid = self._to_int(arguments.get("cid"), -1)
        fid = self._to_int(arguments.get("fid"), -1)
        quote = str(arguments.get("quote", "") if arguments.get("quote", "") is not None else "").strip()
        memo = str(arguments.get("memo", "") if arguments.get("memo", "") is not None else "")

        if cid <= 0:
            raise ValueError("cid must be a positive integer.")
        if fid <= 0:
            raise ValueError("fid must be a positive integer.")
        if quote == "":
            raise ValueError("quote must not be empty.")

        conn = self._connect()
        try:
            cur = conn.cursor()
            now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

            code_row = cur.execute("SELECT cid, name FROM code_name WHERE cid=?", (cid,)).fetchone()
            if code_row is None:
                raise ValueError(f"Code id {cid} not found.")

            source_row = cur.execute(
                "SELECT id, name, ifnull(fulltext,'') FROM source WHERE id=? AND fulltext IS NOT NULL",
                (fid,),
            ).fetchone()
            if source_row is None:
                raise ValueError(f"Text document id {fid} not found.")
            fulltext = str(source_row[2])

            pos0, pos1 = self._quote_search(quote, fulltext)
            if pos0 < 0 or pos1 <= pos0:
                raise ValueError("quote could not be matched in the document text.")
            seltext = fulltext[pos0:pos1]

            existing = cur.execute(
                "SELECT ctid FROM code_text WHERE cid=? AND fid=? AND pos0=? AND pos1=? AND owner=?",
                (cid, fid, pos0, pos1, self.AI_AGENT_OWNER),
            ).fetchone()
            if existing is not None:
                return {
                    "tool": "codes/create_text_coding",
                    "created": False,
                    "reason": "already_exists",
                    "coding": {
                        "ctid": int(existing[0]),
                        "cid": cid,
                        "fid": fid,
                        "pos0": pos0,
                        "pos1": pos1,
                        "owner": self.AI_AGENT_OWNER,
                    },
                }

            cur.execute(
                "INSERT INTO code_text (cid, fid, seltext, pos0, pos1, owner, date, memo) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (cid, fid, seltext, pos0, pos1, self.AI_AGENT_OWNER, now, memo),
            )
            ctid = int(cur.lastrowid)
            conn.commit()
            if hasattr(self.app, "delete_backup"):
                self.app.delete_backup = False

            self._record_ai_change(
                change_set_id,
                {
                    "type": "create_coding_text",
                    "ctid": ctid,
                    "cid": cid,
                    "fid": fid,
                    "code_name": str(code_row[1] if code_row[1] is not None else ""),
                    "source_name": str(source_row[1] if source_row[1] is not None else ""),
                    "pos0": pos0,
                    "pos1": pos1,
                    "seltext": seltext,
                    "owner": self.AI_AGENT_OWNER,
                    "memo": memo,
                    "created_at": now,
                },
            )
            return {
                "tool": "codes/create_text_coding",
                "created": True,
                "coding": {
                    "ctid": ctid,
                    "cid": cid,
                    "fid": fid,
                    "pos0": pos0,
                    "pos1": pos1,
                    "quote": seltext,
                    "owner": self.AI_AGENT_OWNER,
                    "date": now,
                },
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _normalize_hex_color(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", text) is None:
            return ""
        return text.upper()

    def _quote_search(self, quote: str, fulltext: str) -> Tuple[int, int]:
        """Find quote boundaries, preferring ai_llm.ai_quote_search with graceful fallback."""

        try:
            from .ai_llm import ai_quote_search as _ai_quote_search
            return _ai_quote_search(quote, fulltext)
        except Exception:
            quote_text = str(quote).strip()
            if quote_text == "":
                return -1, -1
            start = fulltext.find(quote_text)
            if start < 0:
                return -1, -1
            return start, start + len(quote_text)

    def _record_ai_change(self, change_set_id: str, operation: Dict[str, Any]) -> None:
        ai = getattr(self.app, "ai", None)
        if ai is not None and hasattr(ai, "record_ai_change"):
            ai.record_ai_change(change_set_id, operation)

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
        speaker_prefix = "ðŸ“Œ "
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

    def _fetch_source_name(self, doc_id: int) -> Optional[str]:
        row = self._fetchone("SELECT name FROM source WHERE id=?", (doc_id,))
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

        return {
            "strategy": strategy,
            "max_segments": max_segments,
            "max_chars": max_chars,
            "cursor": cursor,
            "file_ids": file_ids,
        }

    def _parse_vector_search_options(self, query: Dict[str, List[str]]) -> Dict[str, Any]:
        query_strings: List[str] = []
        for key in ("q", "query", "queries"):
            values = query.get(key, [])
            for raw in values:
                raw_text = str(raw).strip()
                if raw_text == "":
                    continue
                for line in raw_text.splitlines():
                    line_clean = " ".join(line.split()).strip()
                    if line_clean != "":
                        query_strings.append(line_clean)

        # Keep order, remove duplicates case-insensitively.
        deduped_queries: List[str] = []
        seen_queries: set[str] = set()
        for q in query_strings:
            q_key = q.lower()
            if q_key in seen_queries:
                continue
            seen_queries.add(q_key)
            deduped_queries.append(q)

        if len(deduped_queries) == 0:
            raise ValueError("Missing vector search query. Use at least one ?q=... parameter.")

        cursor = self._to_int(query.get("cursor", [0])[0], 0)
        cursor = max(0, cursor)

        page_size = self._to_int(query.get("page_size", [self.default_vector_page_size])[0], self.default_vector_page_size)
        page_size = max(1, min(page_size, self.max_vector_page_size))

        score_threshold = self._to_float(
            query.get("score_threshold", [self.default_vector_score_threshold])[0],
            self.default_vector_score_threshold,
        )
        score_threshold = max(0.0, min(score_threshold, 1.0))

        k_per_query = self._to_int(
            query.get("k_per_query", [self.default_vector_k_per_query])[0],
            self.default_vector_k_per_query,
        )
        k_per_query = max(1, min(k_per_query, self.max_vector_k_per_query))

        file_ids: List[int] = []
        for raw in query.get("file_ids", []):
            parts = [p.strip() for p in str(raw).split(",")]
            for part in parts:
                if part == "":
                    continue
                file_ids.append(max(0, self._to_int(part, -1)))
        file_ids = [fid for fid in file_ids if fid > 0]

        return {
            "queries": deduped_queries,
            "cursor": cursor,
            "page_size": page_size,
            "file_ids": file_ids,
            "score_threshold": score_threshold,
            "k_per_query": k_per_query,
        }

    def _parse_regex_search_options(self, query: Dict[str, List[str]]) -> Dict[str, Any]:
        pattern = ""
        for key in ("pattern", "q", "query"):
            values = query.get(key, [])
            if len(values) == 0:
                continue
            pattern = str(values[0]).strip()
            if pattern != "":
                break
        if pattern == "":
            raise ValueError("Missing regex pattern. Use ?pattern=...")

        flags = str(query.get("flags", [""])[0]).strip().lower()
        flags = "".join(ch for ch in flags if ch in ("i", "m", "s", "x"))

        cursor = self._to_int(query.get("cursor", [0])[0], 0)
        cursor = max(0, cursor)

        page_size = self._to_int(query.get("page_size", [self.default_regex_page_size])[0], self.default_regex_page_size)
        page_size = max(1, min(page_size, self.max_regex_page_size))

        context_chars = self._to_int(
            query.get("context_chars", [self.default_regex_context_chars])[0],
            self.default_regex_context_chars,
        )
        context_chars = max(0, min(context_chars, self.max_regex_context_chars))

        file_ids: List[int] = []
        for raw in query.get("file_ids", []):
            parts = [p.strip() for p in str(raw).split(",")]
            for part in parts:
                if part == "":
                    continue
                file_ids.append(max(0, self._to_int(part, -1)))
        file_ids = [fid for fid in file_ids if fid > 0]

        return {
            "pattern": pattern,
            "flags": flags,
            "cursor": cursor,
            "page_size": page_size,
            "context_chars": context_chars,
            "file_ids": file_ids,
        }

    def _read_vector_search(self, options: Dict[str, Any]) -> Dict[str, Any]:
        ai = getattr(self.app, "ai", None)
        if ai is None:
            raise RuntimeError("AI integration is not initialized.")
        vectorstore = getattr(ai, "sources_vectorstore", None)
        if vectorstore is None or getattr(vectorstore, "faiss_db", None) is None:
            raise RuntimeError("Vectorstore is not initialized.")
        is_ready_fn = getattr(vectorstore, "is_ready", None)
        if callable(is_ready_fn) and not is_ready_fn():
            raise RuntimeError("Vectorstore is currently updating. Please try again shortly.")

        queries = options.get("queries", [])
        if not isinstance(queries, list) or len(queries) == 0:
            raise ValueError("Missing vector search query.")
        cursor = max(0, self._to_int(options.get("cursor", 0), 0))
        page_size = max(1, min(self._to_int(options.get("page_size", self.default_vector_page_size),
                                            self.default_vector_page_size),
                               self.max_vector_page_size))
        file_ids = options.get("file_ids", [])
        if not isinstance(file_ids, list):
            file_ids = []
        score_threshold = max(0.0, min(self._to_float(options.get("score_threshold", self.default_vector_score_threshold),
                                                      self.default_vector_score_threshold), 1.0))
        k_per_query = max(1, min(self._to_int(options.get("k_per_query", self.default_vector_k_per_query),
                                              self.default_vector_k_per_query), self.max_vector_k_per_query))

        cache_key = self._vector_search_cache_key(queries, file_ids, score_threshold, k_per_query)

        conn = self._connect_chat_history()
        try:
            self._ensure_vector_search_cache_tables(conn)
            cache_meta = self._get_vector_search_cache(conn, cache_key)
            if cache_meta is None:
                vectorstore_sig = self._vectorstore_signature()
                cache_id, total_hits = self._build_vector_search_cache(
                    conn,
                    cache_key,
                    vectorstore_sig,
                    queries,
                    file_ids,
                    score_threshold,
                    k_per_query,
                )
            else:
                cache_id, total_hits = cache_meta

            if cursor > total_hits:
                cursor = total_hits

            cur = conn.cursor()

            hits: List[Dict[str, Any]] = []
            next_cursor = cursor
            fetch_batch_size = max(page_size * 3, 50)
            while len(hits) < page_size and next_cursor < total_hits:
                cur.execute(
                    "SELECT position, docstore_id, source_id, start_index, text_length, score "
                    "FROM mcp_vector_search_hits "
                    "WHERE cache_id=? AND position>=? "
                    "ORDER BY position ASC LIMIT ?",
                    (cache_id, next_cursor, fetch_batch_size),
                )
                rows = cur.fetchall()
                if len(rows) == 0:
                    next_cursor = total_hits
                    break

                docstore_ids = [str(row[1]).strip() for row in rows if row[1] is not None and str(row[1]).strip() != ""]
                docs_map = self._fetch_cached_documents_by_docstore_id(docstore_ids)

                for row in rows:
                    position = self._to_int(row[0], 0)
                    docstore_id = "" if row[1] is None else str(row[1]).strip()
                    source_id = self._to_int(row[2], -1)
                    start_index = self._to_int(row[3], -1)
                    score = self._to_float(row[5], 0.0)
                    next_cursor = max(next_cursor, position + 1)

                    if docstore_id == "":
                        continue
                    doc_obj = docs_map.get(docstore_id)
                    if doc_obj is None:
                        # stale cache entry: silently skip
                        continue

                    text = str(getattr(doc_obj, "page_content", ""))
                    if text.strip() == "":
                        continue
                    metadata = getattr(doc_obj, "metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    meta_source_id = self._to_int(metadata.get("id"), -1)
                    if meta_source_id > 0:
                        source_id = meta_source_id
                    meta_start_index = self._to_int(metadata.get("start_index"), -1)
                    if meta_start_index >= 0:
                        start_index = meta_start_index
                    source_name = str(metadata.get("name", "")).strip()
                    if source_name == "" and source_id > 0:
                        fetched_name = self._fetch_source_name(source_id)
                        source_name = "" if fetched_name is None else str(fetched_name)

                    hits.append(
                        {
                            "rank": position + 1,
                            "position": position,
                            "docstore_id": docstore_id,
                            "source_id": (source_id if source_id > 0 else None),
                            "source_name": source_name,
                            "start": (start_index if start_index >= 0 else None),
                            "length": len(text),
                            "score": score,
                            "text": text,
                        }
                    )
                    if len(hits) >= page_size:
                        break

            if next_cursor > total_hits:
                next_cursor = total_hits
            truncated = next_cursor < total_hits

            return {
                "selection": {
                    "queries": queries,
                    "cursor": cursor,
                    "page_size": page_size,
                    "file_ids": file_ids,
                    "score_threshold": score_threshold,
                    "k_per_query": k_per_query,
                    "total_hits": total_hits,
                    "returned_hits": len(hits),
                    "next_cursor": next_cursor,
                    "truncated": truncated,
                },
                "hits": hits,
            }
        finally:
            conn.close()

    def _read_regex_search(self, options: Dict[str, Any]) -> Dict[str, Any]:
        pattern_text = str(options.get("pattern", "")).strip()
        if pattern_text == "":
            raise ValueError("Missing regex pattern.")
        flags_text = str(options.get("flags", "")).strip().lower()
        cursor = max(0, self._to_int(options.get("cursor", 0), 0))
        page_size = max(1, min(self._to_int(options.get("page_size", self.default_regex_page_size),
                                            self.default_regex_page_size),
                               self.max_regex_page_size))
        context_chars = max(0, min(self._to_int(options.get("context_chars", self.default_regex_context_chars),
                                                self.default_regex_context_chars),
                                   self.max_regex_context_chars))
        file_ids = options.get("file_ids", [])
        if not isinstance(file_ids, list):
            file_ids = []

        re_flags = self._regex_flags_to_re_flags(flags_text)
        try:
            regex = re.compile(pattern_text, re_flags)
        except re.error as err:
            raise ValueError(f"Invalid regex pattern: {err}")

        cache_key = self._regex_search_cache_key(pattern_text, flags_text, file_ids, context_chars)

        conn = self._connect_chat_history()
        try:
            self._ensure_vector_search_cache_tables(conn)
            cache_meta = self._get_regex_search_cache(conn, cache_key)
            if cache_meta is None:
                cache_id, total_hits = self._build_regex_search_cache(
                    conn,
                    cache_key,
                    pattern_text,
                    flags_text,
                    file_ids,
                    context_chars,
                )
            else:
                cache_id, total_hits = cache_meta

            if cursor > total_hits:
                cursor = total_hits

            cur = conn.cursor()
            hits: List[Dict[str, Any]] = []
            next_cursor = cursor
            fetch_batch_size = max(page_size * 3, 50)

            while len(hits) < page_size and next_cursor < total_hits:
                cur.execute(
                    "SELECT position, source_id, context_start, context_length, match_start, match_length "
                    "FROM mcp_regex_search_hits "
                    "WHERE cache_id=? AND position>=? "
                    "ORDER BY position ASC LIMIT ?",
                    (cache_id, next_cursor, fetch_batch_size),
                )
                rows = cur.fetchall()
                if len(rows) == 0:
                    next_cursor = total_hits
                    break

                source_ids = [self._to_int(row[1], -1) for row in rows]
                source_texts = self._fetch_sources_texts(source_ids)

                for row in rows:
                    position = self._to_int(row[0], 0)
                    source_id = self._to_int(row[1], -1)
                    context_start = self._to_int(row[2], -1)
                    context_length = self._to_int(row[3], 0)
                    match_start = self._to_int(row[4], -1)
                    match_length = self._to_int(row[5], 0)
                    next_cursor = max(next_cursor, position + 1)
                    if source_id <= 0 or context_start < 0 or context_length <= 0:
                        continue

                    source_row = source_texts.get(source_id)
                    if source_row is None:
                        # source vanished or changed unexpectedly; silently skip
                        continue
                    source_name, fulltext = source_row
                    if fulltext == "":
                        continue
                    if context_start >= len(fulltext):
                        continue
                    context_end = min(len(fulltext), context_start + context_length)
                    if context_end <= context_start:
                        continue

                    # Entry may be stale if document text changed since cache build.
                    if match_start < 0 or match_length <= 0:
                        continue
                    if (match_start + match_length) > len(fulltext):
                        continue
                    current_match = regex.match(fulltext, match_start)
                    if current_match is None:
                        continue
                    if current_match.start() != match_start or current_match.end() != (match_start + match_length):
                        continue

                    context_text = fulltext[context_start:context_end]
                    if context_text.strip() == "":
                        continue

                    hits.append(
                        {
                            "order": position + 1,
                            "position": position,
                            "source_id": source_id,
                            "source_name": source_name,
                            "start": context_start,
                            "length": len(context_text),
                            "match_start": match_start,
                            "match_length": match_length,
                            "text": context_text,
                        }
                    )
                    if len(hits) >= page_size:
                        break

            if next_cursor > total_hits:
                next_cursor = total_hits
            truncated = next_cursor < total_hits

            return {
                "selection": {
                    "pattern": pattern_text,
                    "flags": flags_text,
                    "cursor": cursor,
                    "page_size": page_size,
                    "context_chars": context_chars,
                    "file_ids": file_ids,
                    "total_hits": total_hits,
                    "returned_hits": len(hits),
                    "next_cursor": next_cursor,
                    "truncated": truncated,
                },
                "hits": hits,
            }
        finally:
            conn.close()

    def _chat_history_db_path(self) -> str:
        project_path = getattr(self.app, "project_path", "")
        if project_path is None or project_path == "":
            raise RuntimeError("No project open.")
        ai_data_dir = os.path.join(project_path, "ai_data")
        if not os.path.exists(ai_data_dir):
            os.makedirs(ai_data_dir, exist_ok=True)
        return os.path.join(ai_data_dir, "chat_history.sqlite")

    def _connect_chat_history(self) -> sqlite3.Connection:
        return sqlite3.connect(self._chat_history_db_path(), timeout=30)

    def _ensure_vector_search_cache_tables(self, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS mcp_vector_search_cache ("
            "id INTEGER PRIMARY KEY, "
            "cache_key TEXT NOT NULL, "
            "vectorstore_sig TEXT NOT NULL, "
            "query_json TEXT NOT NULL, "
            "file_ids_json TEXT NOT NULL, "
            "score_threshold REAL NOT NULL, "
            "k_per_query INTEGER NOT NULL, "
            "total_hits INTEGER NOT NULL DEFAULT 0, "
            "created_at TEXT NOT NULL, "
            "last_used_at TEXT NOT NULL)"
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mcp_vector_search_cache_key "
            "ON mcp_vector_search_cache(cache_key, vectorstore_sig)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS mcp_vector_search_hits ("
            "id INTEGER PRIMARY KEY, "
            "cache_id INTEGER NOT NULL, "
            "position INTEGER NOT NULL, "
            "docstore_id TEXT, "
            "source_id INTEGER, "
            "start_index INTEGER, "
            "text_length INTEGER, "
            "score REAL, "
            "FOREIGN KEY (cache_id) REFERENCES mcp_vector_search_cache(id))"
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mcp_vector_search_hits_cache_pos "
            "ON mcp_vector_search_hits(cache_id, position)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_mcp_vector_search_cache_key_lookup "
            "ON mcp_vector_search_cache(cache_key)"
        )

        cur.execute(
            "CREATE TABLE IF NOT EXISTS mcp_regex_search_cache ("
            "id INTEGER PRIMARY KEY, "
            "cache_key TEXT NOT NULL, "
            "pattern TEXT NOT NULL, "
            "flags TEXT NOT NULL, "
            "file_ids_json TEXT NOT NULL, "
            "context_chars INTEGER NOT NULL, "
            "total_hits INTEGER NOT NULL DEFAULT 0, "
            "created_at TEXT NOT NULL, "
            "last_used_at TEXT NOT NULL)"
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mcp_regex_search_cache_key "
            "ON mcp_regex_search_cache(cache_key)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS mcp_regex_search_hits ("
            "id INTEGER PRIMARY KEY, "
            "cache_id INTEGER NOT NULL, "
            "position INTEGER NOT NULL, "
            "source_id INTEGER NOT NULL, "
            "context_start INTEGER NOT NULL, "
            "context_length INTEGER NOT NULL, "
            "match_start INTEGER NOT NULL, "
            "match_length INTEGER NOT NULL, "
            "FOREIGN KEY (cache_id) REFERENCES mcp_regex_search_cache(id))"
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mcp_regex_search_hits_cache_pos "
            "ON mcp_regex_search_hits(cache_id, position)"
        )
        conn.commit()

    def _get_vector_search_cache(self, conn: sqlite3.Connection, cache_key: str) -> Optional[Tuple[int, int]]:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, total_hits FROM mcp_vector_search_cache "
            "WHERE cache_key=? ORDER BY id DESC LIMIT 1",
            (cache_key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cache_id = self._to_int(row[0], -1)
        total_hits = max(0, self._to_int(row[1], 0))
        if cache_id <= 0:
            return None
        now = datetime.utcnow().isoformat(timespec="seconds")
        cur.execute(
            "UPDATE mcp_vector_search_cache SET last_used_at=? WHERE id=?",
            (now, cache_id),
        )
        conn.commit()
        return cache_id, total_hits

    def _build_vector_search_cache(self, conn: sqlite3.Connection, cache_key: str,
                                   vectorstore_sig: str, queries: List[str], file_ids: List[int],
                                   score_threshold: float, k_per_query: int) -> Tuple[int, int]:
        ai = getattr(self.app, "ai", None)
        if ai is None:
            raise RuntimeError("AI integration is not initialized.")
        doc_filter = file_ids if len(file_ids) > 0 else None
        chunks = ai._retrieve_from_vectorstore(
            queries,
            doc_ids=doc_filter,
            score_threshold=score_threshold,
            k=k_per_query,
        )
        if not isinstance(chunks, list):
            chunks = []
        total_hits = len(chunks)
        now = datetime.utcnow().isoformat(timespec="seconds")

        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO mcp_vector_search_cache "
                "(cache_key, vectorstore_sig, query_json, file_ids_json, score_threshold, "
                "k_per_query, total_hits, created_at, last_used_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cache_key,
                    vectorstore_sig,
                    json.dumps(queries, ensure_ascii=False),
                    json.dumps(file_ids),
                    float(score_threshold),
                    int(k_per_query),
                    int(total_hits),
                    now,
                    now,
                ),
            )
            cache_id = int(cur.lastrowid)
        except sqlite3.IntegrityError:
            # If another writer inserted the same key first, reuse it.
            conn.rollback()
            existing = self._get_vector_search_cache(conn, cache_key)
            if existing is None:
                raise
            return existing

        rows_to_insert: List[Tuple[Any, ...]] = []
        for pos, chunk_doc in enumerate(chunks):
            docstore_id = getattr(chunk_doc, "id", None)
            if docstore_id is None:
                docstore_id = ""
            else:
                docstore_id = str(docstore_id)

            metadata = getattr(chunk_doc, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            source_id = self._to_int(metadata.get("id"), -1)
            if source_id <= 0:
                source_id = None
            start_index = self._to_int(metadata.get("start_index"), -1)
            if start_index < 0:
                start_index = None
            page_content = str(getattr(chunk_doc, "page_content", ""))
            text_length = len(page_content)
            score = self._to_float(metadata.get("score", 0.0), 0.0)

            rows_to_insert.append(
                (
                    cache_id,
                    pos,
                    docstore_id,
                    source_id,
                    start_index,
                    text_length,
                    score,
                )
            )
        if len(rows_to_insert) > 0:
            cur.executemany(
                "INSERT INTO mcp_vector_search_hits "
                "(cache_id, position, docstore_id, source_id, start_index, text_length, score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows_to_insert,
            )
        conn.commit()
        return cache_id, total_hits

    def _get_regex_search_cache(self, conn: sqlite3.Connection, cache_key: str) -> Optional[Tuple[int, int]]:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, total_hits FROM mcp_regex_search_cache WHERE cache_key=? LIMIT 1",
            (cache_key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cache_id = self._to_int(row[0], -1)
        total_hits = max(0, self._to_int(row[1], 0))
        if cache_id <= 0:
            return None
        now = datetime.utcnow().isoformat(timespec="seconds")
        cur.execute(
            "UPDATE mcp_regex_search_cache SET last_used_at=? WHERE id=?",
            (now, cache_id),
        )
        conn.commit()
        return cache_id, total_hits

    def _build_regex_search_cache(self, conn: sqlite3.Connection, cache_key: str, pattern_text: str,
                                  flags_text: str, file_ids: List[int], context_chars: int) -> Tuple[int, int]:
        re_flags = self._regex_flags_to_re_flags(flags_text)
        try:
            regex = re.compile(pattern_text, re_flags)
        except re.error as err:
            raise ValueError(f"Invalid regex pattern: {err}")

        where_sql = " WHERE fulltext is not null"
        params: List[Any] = []
        norm_file_ids: List[int] = []
        for fid in file_ids:
            try:
                fid_i = int(fid)
            except (TypeError, ValueError):
                continue
            if fid_i > 0:
                norm_file_ids.append(fid_i)
        norm_file_ids = sorted(set(norm_file_ids))
        if len(norm_file_ids) > 0:
            placeholders = ",".join(["?"] * len(norm_file_ids))
            where_sql += f" AND id IN ({placeholders})"
            params.extend(norm_file_ids)

        rows = self._fetchall(
            "SELECT id, ifnull(name,''), ifnull(fulltext,'') FROM source"
            + where_sql
            + " ORDER BY lower(name)",
            tuple(params),
        )

        now = datetime.utcnow().isoformat(timespec="seconds")
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO mcp_regex_search_cache "
                "(cache_key, pattern, flags, file_ids_json, context_chars, total_hits, created_at, last_used_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cache_key,
                    pattern_text,
                    flags_text,
                    json.dumps(norm_file_ids),
                    int(context_chars),
                    0,
                    now,
                    now,
                ),
            )
            cache_id = int(cur.lastrowid)
        except sqlite3.IntegrityError:
            conn.rollback()
            existing = self._get_regex_search_cache(conn, cache_key)
            if existing is None:
                raise
            return existing

        match_rows: List[Tuple[Any, ...]] = []
        for row in rows:
            source_id = self._to_int(row[0], -1)
            fulltext = "" if row[2] is None else str(row[2])
            if source_id <= 0 or fulltext == "":
                continue
            for match in regex.finditer(fulltext):
                match_start = int(match.start())
                match_end = int(match.end())
                if match_end <= match_start:
                    continue
                context_start = max(0, match_start - context_chars)
                context_end = min(len(fulltext), match_end + context_chars)
                context_length = max(0, context_end - context_start)
                if context_length <= 0:
                    continue
                match_rows.append(
                    (
                        source_id,
                        context_start,
                        context_length,
                        match_start,
                        match_end - match_start,
                    )
                )
                if len(match_rows) >= self.max_regex_hits:
                    break
            if len(match_rows) >= self.max_regex_hits:
                break

        # Regex has no relevance score. Mix matches deterministically to avoid over-weighting early documents.
        rng = random.Random(cache_key)
        rng.shuffle(match_rows)

        insert_rows: List[Tuple[Any, ...]] = []
        for pos, row_data in enumerate(match_rows):
            insert_rows.append((cache_id, pos, *row_data))

        if len(insert_rows) > 0:
            cur.executemany(
                "INSERT INTO mcp_regex_search_hits "
                "(cache_id, position, source_id, context_start, context_length, match_start, match_length) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                insert_rows,
            )
        cur.execute(
            "UPDATE mcp_regex_search_cache SET total_hits=? WHERE id=?",
            (len(insert_rows), cache_id),
        )
        conn.commit()
        return cache_id, len(insert_rows)

    def _vector_search_cache_key(self, queries: List[str], file_ids: List[int],
                                 score_threshold: float, k_per_query: int) -> str:
        norm_file_ids: List[int] = []
        for fid in file_ids:
            try:
                fid_i = int(fid)
            except (TypeError, ValueError):
                continue
            if fid_i > 0:
                norm_file_ids.append(fid_i)
        payload = {
            "queries": queries,
            "file_ids": sorted(set(norm_file_ids)),
            "score_threshold": round(float(score_threshold), 4),
            "k_per_query": int(k_per_query),
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _regex_search_cache_key(self, pattern_text: str, flags_text: str, file_ids: List[int],
                                context_chars: int) -> str:
        norm_file_ids: List[int] = []
        for fid in file_ids:
            try:
                fid_i = int(fid)
            except (TypeError, ValueError):
                continue
            if fid_i > 0:
                norm_file_ids.append(fid_i)
        payload = {
            "version": "regex_order_shuffle_v1",
            "pattern": pattern_text,
            "flags": flags_text,
            "file_ids": sorted(set(norm_file_ids)),
            "context_chars": int(context_chars),
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _regex_flags_to_re_flags(self, flags_text: str) -> int:
        flags = 0
        text = str(flags_text).lower()
        if "i" in text:
            flags |= re.IGNORECASE
        if "m" in text:
            flags |= re.MULTILINE
        if "s" in text:
            flags |= re.DOTALL
        if "x" in text:
            flags |= re.VERBOSE
        return flags

    def _vectorstore_signature(self) -> str:
        ai = getattr(self.app, "ai", None)
        vectorstore = getattr(ai, "sources_vectorstore", None) if ai is not None else None
        faiss_path = getattr(vectorstore, "faiss_db_path", None) if vectorstore is not None else None
        if faiss_path is None or str(faiss_path).strip() == "":
            project_path = getattr(self.app, "project_path", "")
            if project_path is None or project_path == "":
                return "missing"
            faiss_path = os.path.join(project_path, "ai_data", "vectorstore", "faiss_store.bin")
        if not os.path.exists(faiss_path):
            return "missing"
        try:
            stat = os.stat(faiss_path)
            return f"{int(stat.st_mtime)}:{int(stat.st_size)}"
        except OSError:
            return "unknown"

    def _fetch_cached_documents_by_docstore_id(self, docstore_ids: List[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        ai = getattr(self.app, "ai", None)
        vectorstore = getattr(ai, "sources_vectorstore", None) if ai is not None else None
        faiss_db = getattr(vectorstore, "faiss_db", None) if vectorstore is not None else None
        docstore = getattr(faiss_db, "docstore", None) if faiss_db is not None else None
        if docstore is None:
            return result

        for docstore_id in docstore_ids:
            key = str(docstore_id).strip()
            if key == "" or key in result:
                continue
            try:
                doc = docstore.search(key)
            except Exception:
                doc = None
            if doc is None:
                continue
            if not hasattr(doc, "page_content"):
                continue
            result[key] = doc
        return result

    def _fetch_sources_texts(self, source_ids: List[int]) -> Dict[int, Tuple[str, str]]:
        normalized_ids: List[int] = []
        for source_id in source_ids:
            try:
                source_id_i = int(source_id)
            except (TypeError, ValueError):
                continue
            if source_id_i > 0:
                normalized_ids.append(source_id_i)
        normalized_ids = sorted(set(normalized_ids))
        if len(normalized_ids) == 0:
            return {}
        placeholders = ",".join(["?"] * len(normalized_ids))
        rows = self._fetchall(
            "SELECT id, ifnull(name,''), ifnull(fulltext,'') FROM source "
            f"WHERE id IN ({placeholders})",
            tuple(normalized_ids),
        )
        result: Dict[int, Tuple[str, str]] = {}
        for row in rows:
            source_id = self._to_int(row[0], -1)
            if source_id <= 0:
                continue
            source_name = "" if row[1] is None else str(row[1])
            fulltext = "" if row[2] is None else str(row[2])
            result[source_id] = (source_name, fulltext)
        return result

    def _read_code_segments(self, cid: int, options: Dict[str, Any]) -> Dict[str, Any]:
        code_name = self._fetch_code_name(cid)
        if code_name is None:
            raise ValueError(f"Code id {cid} not found.")

        strategy = str(options.get("strategy", "diverse_by_document"))
        max_segments = int(options.get("max_segments", self.default_segments_max_segments))
        max_chars = int(options.get("max_chars", self.default_segments_max_chars))
        cursor = int(options.get("cursor", 0))
        file_ids = options.get("file_ids", [])
        if not isinstance(file_ids, list):
            file_ids = []

        if not self._view_exists("code_text_visible"):
            raise RuntimeError("Required view 'code_text_visible' not found.")
        table_name = "code_text_visible"

        where_parts = ["ct.cid=?"]
        where_params: List[Any] = [cid]
        if len(file_ids) > 0:
            placeholders = ",".join(["?"] * len(file_ids))
            where_parts.append(f"ct.fid IN ({placeholders})")
            where_params.extend(file_ids)
        where_sql = " WHERE " + " AND ".join(where_parts)

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
                "visible_filter_applied": True,
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

    def _to_int(self, value: Any, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _to_float(self, value: Any, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        try:
            return float(value)
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
