# -*- coding: utf-8 -*-

"""
Internal MCP server for QualCoder (read-only phase).

This module uses the official MCP Python SDK (low-level server) and exposes
an in-process JSON-RPC bridge (`handle_request`) so the current chat flow can
call it without transport setup.
"""

import asyncio
import hashlib
import json
import os
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
    STATUS_REVIEW_AVAILABLE_MATERIALS = "status.review.available_materials"
    STATUS_REVIEW_DOCUMENT_LIST = "status.review.document_list"
    STATUS_REVIEW_CODE_TREE = "status.review.code_tree"
    STATUS_REVIEW_DOCUMENT = "status.review.document"
    STATUS_REVIEW_RESOURCE = "status.review.resource"
    STATUS_REVIEW_CODE_SEGMENTS = "status.review.code_segments"
    STATUS_REVIEW_VECTOR_SEARCH = "status.review.vector_search"
    STATUS_REVIEW_REGEX_SEARCH = "status.review.regex_search"
    STATUS_PLAN_MCP_STEPS = "status.plan.mcp_steps"
    STATUS_EXECUTE_MCP_STEPS = "status.execute.mcp_steps"
    STATUS_REFLECT_MCP_RESULTS = "status.reflect.mcp_results"
    STATUS_FORMULATE_RESPONSE = "status.final.formulate_response"

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
            "QualCoder internal MCP server (read-only). "
            "Use resources/list, resources/read, prompts/list, and prompts/get. "
            "Available resources: text documents list (qualcoder://documents), document text by id "
            "(qualcoder://documents/text/{id}), code tree (qualcoder://codes/tree), and coded text segments by code id "
            "(qualcoder://codes/segments/{cid}), semantic vector search "
            "(qualcoder://vector/search?q=...), and regular-expression search "
            "(qualcoder://search/regex?pattern=...). "
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
        if uri_base == "qualcoder://vector/search":
            return {
                **base_event,
                "status_code": "vector_search",
                "entity_type": "vector_search",
                "message_id": self.STATUS_REVIEW_VECTOR_SEARCH,
            }
        if uri_base == "qualcoder://search/regex":
            return {
                **base_event,
                "status_code": "regex_search",
                "entity_type": "regex_search",
                "message_id": self.STATUS_REVIEW_REGEX_SEARCH,
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

        return {
            **base_event,
            "status_code": "resource_read",
            "entity_type": "resource",
            "message_id": self.STATUS_REVIEW_RESOURCE,
        }

    def describe_host_status_event(self, phase: str) -> Optional[Dict[str, Any]]:
        """Describe non-MCP but related host-side status events."""

        if phase == "planning":
            return {
                "status_code": "planning",
                "phase": "start",
                "message_id": self.STATUS_PLAN_MCP_STEPS,
                "message_args": {},
            }
        if phase == "execution":
            return {
                "status_code": "execution",
                "phase": "start",
                "message_id": self.STATUS_EXECUTE_MCP_STEPS,
                "message_args": {},
            }
        if phase == "reflection":
            return {
                "status_code": "reflection",
                "phase": "start",
                "message_id": self.STATUS_REFLECT_MCP_RESULTS,
                "message_args": {},
            }
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
        templates = {
            # Keep literal msgids in _() calls so gettext extraction can discover them.
            self.STATUS_REVIEW_AVAILABLE_MATERIALS: _('Reviewing available project materials...'),
            self.STATUS_REVIEW_DOCUMENT_LIST: _('Reviewing the list of text documents...'),
            self.STATUS_REVIEW_CODE_TREE: _('Reviewing the current code structure...'),
            self.STATUS_REVIEW_DOCUMENT: _('Reviewing text document "{name}"...'),
            self.STATUS_REVIEW_RESOURCE: _('Reviewing project material...'),
            self.STATUS_REVIEW_CODE_SEGMENTS: _('Reviewing coded text segments for "{name}"...'),
            self.STATUS_REVIEW_VECTOR_SEARCH: _('Running semantic search in the project data...'),
            self.STATUS_REVIEW_REGEX_SEARCH: _('Running keyword search in the project data...'),
            self.STATUS_PLAN_MCP_STEPS: _('Planning how to gather project evidence...'),
            self.STATUS_EXECUTE_MCP_STEPS: _('Executing MCP retrieval steps...'),
            self.STATUS_REFLECT_MCP_RESULTS: _('Reflecting on retrieved evidence and revising the plan...'),
            self.STATUS_FORMULATE_RESPONSE: _('Formulating a response based on the selected materials...'),
        }
        translated = templates.get(message_id)
        if translated is None:
            # Backward compatibility if older serialized events contain literal text IDs.
            translated = _(message_id)
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
            return types.ListToolsResult(tools=[])

        @self._sdk_server.call_tool()
        async def _call_tool(_name: str, _arguments: Dict[str, Any]) -> Dict[str, Any]:
            raise RuntimeError("No MCP tools are enabled yet. Current phase is read-only resources.")

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
                            "rank": position + 1,
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

        insert_rows: List[Tuple[Any, ...]] = []
        pos = 0
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
                insert_rows.append(
                    (
                        cache_id,
                        pos,
                        source_id,
                        context_start,
                        context_length,
                        match_start,
                        match_end - match_start,
                    )
                )
                pos += 1
                if pos >= self.max_regex_hits:
                    break
            if pos >= self.max_regex_hits:
                break

        if len(insert_rows) > 0:
            cur.executemany(
                "INSERT INTO mcp_regex_search_hits "
                "(cache_id, position, source_id, context_start, context_length, match_start, match_length) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                insert_rows,
            )
        cur.execute(
            "UPDATE mcp_regex_search_cache SET total_hits=? WHERE id=?",
            (pos, cache_id),
        )
        conn.commit()
        return cache_id, pos

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
