import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, call, patch

import numpy as np
from PyQt6 import QtCore

from qualcoder.ai_vectorstore import AiVectorstore, E5SentenceTransformerEmbeddings


class TestE5SentenceTransformerEmbeddings(TestCase):
    def test_preserves_preprocessing_for_existing_search_indexes(self):
        with patch("qualcoder.ai_vectorstore.sentence_transformers.SentenceTransformer") as model_class:
            encode = model_class.return_value.encode
            encode.side_effect = [np.array([[1.0], [2.0]]), np.array([[3.0]])]

            embeddings = E5SentenceTransformerEmbeddings("cached-model")
            documents = embeddings.embed_documents(["first\nline", "second\r\nline"])
            query = embeddings.embed_query("first\nquery")

        model_class.assert_called_once_with("cached-model")
        self.assertEqual([[1.0], [2.0]], documents)
        self.assertEqual([3.0], query)
        self.assertEqual(
            [
                call(["passage: first line", "passage: second\r line"],
                     show_progress_bar=False, normalize_embeddings=False),
                call(["query: first query"], show_progress_bar=False, normalize_embeddings=False),
            ],
            encode.call_args_list,
        )


class TestVectorstoreUpgrade(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qt_app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def test_rebuilds_old_cache_from_project_text_and_reuses_sqlite_index(self):
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        project_dir = Path(temp_dir.name)
        with closing(sqlite3.connect(project_dir / "data.qda")) as conn:
            conn.execute(
                "CREATE TABLE source (id INTEGER PRIMARY KEY, name TEXT, fulltext TEXT, "
                "memo TEXT, owner TEXT, date TEXT, mediapath TEXT)"
            )
            conn.execute("INSERT INTO source (id, name, fulltext) VALUES (1, 'interview.txt', 'Interview text')")
            conn.commit()
        old_cache = project_dir / "ai_data" / "vectorstore" / "faiss_store.bin"
        old_cache.parent.mkdir(parents=True)
        old_cache.write_bytes(b"old cache")

        embeddings = Mock()
        embeddings.embed_documents.return_value = [[1.0, 0.0]]
        embeddings.embed_query.return_value = [1.0, 0.0]
        app = SimpleNamespace(project_path=str(project_dir), ai_embedding_function=embeddings)
        store = AiVectorstore(app, SimpleNamespace(append=Mock()), "documents")
        self.addCleanup(store.close)

        with patch("qualcoder.ai_vectorstore._", lambda message: message, create=True), \
                patch("qualcoder.ai_vectorstore.open", create=True) as open_file:
            store._open_db()
            embeddings.embed_documents.assert_called_once_with(["Interview text"])
            store.close()
            store._open_db()

        open_file.assert_not_called()
        self.assertEqual(b"old cache", old_cache.read_bytes())
        self.assertTrue((project_dir / "ai_data" / "search.sqlite").is_file())
        embeddings.embed_documents.assert_called_once_with(["Interview text"])
        matches = store.similarity_search_with_relevance_scores("Interview")
        self.assertEqual([("Interview text", 1.0)], [(doc.page_content, score) for doc, score in matches])
