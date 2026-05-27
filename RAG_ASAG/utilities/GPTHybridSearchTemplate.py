from dataclasses import dataclass
from typing import List
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import minmax_scale


@dataclass
class Document:
    id: str
    text: str


class HybridSearchASAG:
    def __init__(
        self,
        docs: List[Document],
        bert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        alpha_bm25: float = 0.45,
        alpha_bert: float = 0.40,
        alpha_asag: float = 0.15,
    ):
        self.docs = docs
        self.alpha_bm25 = alpha_bm25
        self.alpha_bert = alpha_bert
        self.alpha_asag = alpha_asag

        self.tokenized_docs = [self._tokenize(d.text) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        self.model = SentenceTransformer(bert_model)
        self.doc_embeddings = self.model.encode(
            [d.text for d in docs],
            normalize_embeddings=True
        )

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _asag_score(self, query: str, doc_text: str) -> float:
        """
        Platzhalter für ASAG = Automatic Short Answer Grading.
        Hier simpel: semantische Ähnlichkeit zwischen Anfrage und Dokument.
        Kann durch ein feinjustiertes ASAG-BERT-Regressionsmodell ersetzt werden.
        """
        q_emb = self.model.encode(query, normalize_embeddings=True)
        d_emb = self.model.encode(doc_text, normalize_embeddings=True)
        return float(util.cos_sim(q_emb, d_emb)[0][0])

    def search(self, query: str, top_k: int = 5):
        # 1. Fulltext / BM25
        bm25_scores = np.array(
            self.bm25.get_scores(self._tokenize(query)),
            dtype=float
        )

        # 2. BERT semantic search
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        bert_scores = util.cos_sim(
            query_embedding,
            self.doc_embeddings
        )[0].cpu().numpy()

        # 3. ASAG score
        asag_scores = np.array([
            self._asag_score(query, doc.text)
            for doc in self.docs
        ])

        # Normalize scores
        bm25_norm = minmax_scale(bm25_scores) if bm25_scores.max() > 0 else bm25_scores
        bert_norm = minmax_scale(bert_scores)
        asag_norm = minmax_scale(asag_scores)

        # Hybrid weighted score
        final_scores = (
            self.alpha_bm25 * bm25_norm +
            self.alpha_bert * bert_norm +
            self.alpha_asag * asag_norm
        )

        ranked_indices = np.argsort(final_scores)[::-1][:top_k]

        return [
            {
                "id": self.docs[i].id,
                "text": self.docs[i].text,
                "bm25": float(bm25_norm[i]),
                "bert": float(bert_norm[i]),
                "asag": float(asag_norm[i]),
                "score": float(final_scores[i]),
            }
            for i in ranked_indices
        ]


if __name__ == "__main__":
    docs = [
        Document("1", "BERT can be used for semantic search and text similarity."),
        Document("2", "BM25 is a classic fulltext search ranking algorithm."),
        Document("3", "Hybrid search combines lexical and semantic retrieval."),
        Document("4", "ASAG means automatic short answer grading using NLP models."),
    ]

    engine = HybridSearchASAG(docs)

    results = engine.search("How does hybrid search with BERT and fulltext work?")

    for r in results:
        print("\nID:", r["id"])
        print("Score:", round(r["score"], 3))
        print("BM25:", round(r["bm25"], 3))
        print("BERT:", round(r["bert"], 3))
        print("ASAG:", round(r["asag"], 3))
        print("Text:", r["text"])