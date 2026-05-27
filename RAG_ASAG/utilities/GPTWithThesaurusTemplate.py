# hybrid_thesaurus_search.py
# pip install pymupdf python-docx pandas rank-bm25 sentence-transformers scikit-learn

import os
import re
import json
import fitz
import docx
import pandas as pd
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 1. Dokumente laden
# -----------------------------

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path):
    text = []
    pdf = fitz.open(path)
    for page in pdf:
        text.append(page.get_text())
    return "\n".join(text)


def read_docx(path):
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def read_csv(path):
    df = pd.read_csv(path)
    return "\n".join(df.astype(str).fillna("").agg(" ".join, axis=1))


def load_documents(folder):
    docs = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        ext = file.lower().split(".")[-1]

        if ext == "txt":
            text = read_txt(path)
        elif ext == "pdf":
            text = read_pdf(path)
        elif ext == "docx":
            text = read_docx(path)
        elif ext == "csv":
            text = read_csv(path)
        else:
            continue

        docs.append({
            "filename": file,
            "text": text
        })

    return docs


# -----------------------------
# 2. Einfacher Thesaurus
# -----------------------------

class Thesaurus:
    def __init__(self):
        self.terms = defaultdict(set)

    def add_group(self, words):
        words = [w.lower().strip() for w in words if w.strip()]
        for w in words:
            self.terms[w].update(words)
            self.terms[w].discard(w)

    def expand_query(self, query):
        tokens = tokenize(query)
        expanded = list(tokens)

        for token in tokens:
            expanded.extend(self.terms.get(token.lower(), []))

        return " ".join(expanded)

    def save(self, path):
        data = {k: sorted(v) for k, v in self.terms.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.terms = defaultdict(set, {k: set(v) for k, v in data.items()})


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# -----------------------------
# 3. Thesaurus aus CSV befüllen
# Format:
# term,synonyms
# car,"auto,vehicle,automobile"
# cancer,"tumor,carcinoma,malignancy"
# -----------------------------

def fill_thesaurus_from_csv(thesaurus, path):
    df = pd.read_csv(path)

    for _, row in df.iterrows():
        term = str(row["term"])
        synonyms = str(row["synonyms"]).split(",")
        thesaurus.add_group([term] + synonyms)


# -----------------------------
# 4. Thesaurus automatisch aus Text befüllen
# sehr einfache Variante:
# häufige ähnliche Begriffe müssten normalerweise
# mit NLP/Embeddings/WordNet/LLM geprüft werden.
# Hier werden manuelle Marker unterstützt

# hybrid_search.py
# pip install sentence-transformers rank-bm25 numpy scikit-learn

import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


class Thesaurus:
    def __init__(self):
        self.synonyms = {}

    def add_group(self, words):
        words = [w.lower().strip() for w in words]
        for w in words:
            self.synonyms.setdefault(w, set()).update(set(words) - {w})

    def expand_query(self, query: str):
        tokens = tokenize(query)
        expanded = list(tokens)

        for token in tokens:
            expanded.extend(self.synonyms.get(token, []))

        return " ".join(expanded)


class HybridSearch:
    def __init__(self, documents, thesaurus=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.documents = documents
        self.thesaurus = thesaurus or Thesaurus()

        self.texts = [d["text"] for d in documents]
        self.tokenized_docs = [tokenize(t) for t in self.texts]

        self.bm25 = BM25Okapi(self.tokenized_docs)

        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def search(self, query, top_k=5, bm25_weight=0.5, semantic_weight=0.5):
        expanded_query = self.thesaurus.expand_query(query)

        bm25_scores = np.array(
            self.bm25.get_scores(tokenize(expanded_query)),
            dtype=float
        )

        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        semantic_scores = cosine_similarity(
            query_embedding,
            self.doc_embeddings
        )[0]

        hybrid_scores = (
            bm25_weight * bm25_scores +
            semantic_weight * semantic_scores
        )

        ranked = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in ranked:
            results.append({
                "id": self.documents[idx].get("id", idx),
                "title": self.documents[idx].get("title", ""),
                "text": self.documents[idx]["text"][:500],
                "bm25_score": float(bm25_scores[idx]),
                "semantic_score": float(semantic_scores[idx]),
                "hybrid_score": float(hybrid_scores[idx]),
                "expanded_query": expanded_query
            })

        return results


if __name__ == "__main__":
    docs = [
        {
            "id": 1,
            "title": "Medizin Krebs",
            "text": "Krebs ist eine Erkrankung, bei der Zellen unkontrolliert wachsen. Tumore können gutartig oder bösartig sein."
        },
        {
            "id": 2,
            "title": "Auto Technik",
            "text": "Ein Fahrzeug besitzt Motor, Getriebe, Räder und elektronische Steuergeräte."
        },
        {
            "id": 3,
            "title": "KI Suche",
            "text": "Hybrid Search kombiniert Volltextsuche mit semantischer Suche über Embeddings."
        },
        {
            "id": 4,
            "title": "Onkologie",
            "text": "Karzinome und maligne Tumoren werden in der Onkologie diagnostiziert und behandelt."
        }
    ]

    thesaurus = Thesaurus()
    thesaurus.add_group(["krebs", "tumor", "karzinom", "malignom", "onkologie"])
    thesaurus.add_group(["auto", "fahrzeug", "wagen", "automobil", "pkw"])
    thesaurus.add_group(["suche", "retrieval", "search", "volltextsuche"])

    engine = HybridSearch(docs, thesaurus)

    query = "Behandlung von Krebs"
    results = engine.search(
        query=query,
        top_k=3,
        bm25_weight=0.45,
        semantic_weight=0.55
    )

    for r in results:
        print("\n---")
        print("ID:", r["id"])
        print("Titel:", r["title"])
        print("Hybrid:", round(r["hybrid_score"], 4))
        print("BM25:", round(r["bm25_score"], 4))
        print("Semantic:", round(r["semantic_score"], 4))
        print("Expanded Query:", r["expanded_query"])
        print("Text:", r["text"])