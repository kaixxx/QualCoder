import json
import os
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest import TestCase

from qualcoder.ai_mcp_server import AiMcpServer


class TestAiMcpServer(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_path = self.temp_dir.name
        self.db_path = os.path.join(self.project_path, "data.qda")
        self.conn = sqlite3.connect(self.db_path)
        cur = self.conn.cursor()

        cur.execute(
            "CREATE TABLE project (databaseversion text, date text, memo text, about text, "
            "bookmarkfile integer, bookmarkpos integer, codername text, recently_used_codes text)"
        )
        cur.execute(
            "CREATE TABLE source (id integer primary key, name text, fulltext text, mediapath text, "
            "memo text, owner text, date text, av_text_id integer, risid integer, unique(name))"
        )
        cur.execute(
            "CREATE TABLE code_cat (catid integer primary key, name text, owner text, date text, memo text, "
            "supercatid integer, unique(name))"
        )
        cur.execute(
            "CREATE TABLE code_name (cid integer primary key, name text, memo text, catid integer, owner text, "
            "date text, color text, unique(name))"
        )
        cur.execute(
            "CREATE TABLE journal (jid integer primary key, name text, jentry text, date text, owner text, unique(name))"
        )
        cur.execute("INSERT INTO project VALUES(?,?,?,?,?,?,?,?)", ("v11", "2026-02-13", "memo text", "about", 0, 0, "default", ""))
        cur.execute(
            "INSERT INTO source (id, name, fulltext, mediapath, memo, owner, date, av_text_id, risid) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (1, "doc one", "abcdefghij", None, "doc memo", "default", "2026-02-13", None, None),
        )
        cur.execute(
            "INSERT INTO code_cat (catid, name, owner, date, memo, supercatid) VALUES (?,?,?,?,?,?)",
            (1, "cat one", "default", "2026-02-13", "cat memo", None),
        )
        cur.execute(
            "INSERT INTO code_name (cid, name, memo, catid, owner, date, color) VALUES (?,?,?,?,?,?,?)",
            (1, "code one", "code memo", 1, "default", "2026-02-13", "#AAAAAA"),
        )
        cur.execute(
            "INSERT INTO journal (jid, name, jentry, date, owner) VALUES (?,?,?,?,?)",
            (1, "journal one", "journal body", "2026-02-13", "default"),
        )
        self.conn.commit()

        self.app = SimpleNamespace(
            project_path=self.project_path,
            project_name="test_project.qda",
            settings={"codername": "default"},
        )
        self.server = AiMcpServer(self.app)

    def tearDown(self):
        self.conn.close()
        self.temp_dir.cleanup()

    def test_initialize(self):
        res = self.server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertIn("result", res)
        self.assertEqual("2025-06-18", res["result"]["protocolVersion"])
        self.assertIn("resources", res["result"]["capabilities"])

    def test_resources_list_contains_only_top_level_resources(self):
        res = self.server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "resources/list", "params": {}})
        self.assertIn("result", res)
        uris = [r["uri"] for r in res["result"]["resources"]]
        self.assertIn("qualcoder://project/memo", uris)
        self.assertIn("qualcoder://codes/tree", uris)
        self.assertIn("qualcoder://documents", uris)
        self.assertIn("qualcoder://journals", uris)
        self.assertNotIn("qualcoder://documents/text/1", uris)
        self.assertNotIn("qualcoder://journals/1", uris)

    def test_resources_read_document_slice(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/read",
            "params": {"uri": "qualcoder://documents/text/1", "start": 2, "length": 4},
        }
        res = self.server.handle_request(req)
        self.assertIn("result", res)
        payload = json.loads(res["result"]["contents"][0]["text"])
        self.assertEqual(1, payload["id"])
        self.assertEqual("cdef", payload["text"])
        self.assertEqual(2, payload["start"])

    def test_unknown_method_returns_jsonrpc_error(self):
        res = self.server.handle_request({"jsonrpc": "2.0", "id": 4, "method": "unknown/method", "params": {}})
        self.assertIn("error", res)
        self.assertEqual(-32601, res["error"]["code"])

    def test_status_event_for_document_read_contains_id_and_name(self):
        event = self.server.describe_status_event("resources/read", {"uri": "qualcoder://documents/text/1"})
        self.assertIsNotNone(event)
        self.assertEqual("document_read", event["status_code"])
        self.assertEqual(1, event["entity_id"])
        self.assertEqual("doc one", event["entity_name"])
        self.assertEqual({"id": 1, "name": "doc one"}, event["message_args"])

    def test_status_event_to_text_uses_document_name(self):
        event = self.server.describe_status_event("resources/read", {"uri": "qualcoder://documents/text/1"})
        txt = self.server.status_event_to_text(event)
        self.assertIn("doc one", txt)

    def test_status_event_for_documents_list(self):
        event = self.server.describe_status_event("resources/read", {"uri": "qualcoder://documents"})
        self.assertIsNotNone(event)
        self.assertEqual("documents_list", event["status_code"])
        self.assertEqual("documents", event["entity_type"])

    def test_host_status_event_for_final_response(self):
        event = self.server.describe_host_status_event("final_response")
        self.assertIsNotNone(event)
        self.assertEqual("final_response", event["status_code"])
        txt = self.server.status_event_to_text(event)
        self.assertTrue(len(txt) > 0)
