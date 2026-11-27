from unittest import TestCase
from unittest.mock import call, patch

import numpy as np

from qualcoder.ai_vectorstore import E5SentenceTransformerEmbeddings


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
