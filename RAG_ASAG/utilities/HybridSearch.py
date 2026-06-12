#######################################################################
##Hybrid search utility including Thesaurus
## (c) 2026 Harald Glab-Plhak
## email: hglabplhak@gmail.com
## MIT License
#######################################################################

import pypdf
import os
import json
import fitz
import pandas as pd
from collections import defaultdict
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Plus
from RAG_ASAG.utilities.RAGUtils import set_api_env_and_keys_in_parent
from  RAG_ASAG.utilities.FullTextSearch import get_collections_path, get_pdf_files, extract_keywords,sort_by_score,print_results,get_key_strict


# 1. Extract Text using PyMuPDF
def extract_pdf_content(pdf_path, split_size=20):
    doc = pypdf.PdfReader(pdf_path)

    chunks = []
    for page in doc.pages:
        # Extract text and split into smaller chunks for better search granularity
        text = page.extract_text()
        # Simple chunking: splitting by double newlines or fixed length
        if len(text) !=  0:
            page_chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > split_size]
            chunks.extend(page_chunks)
    if len(chunks) == 0:
        chunks.append("##dummy##")
    return chunks

def extract_and_search(pdf_path, split_size=20):
    # 2. Setup Hybrid Search Engines
    documents  = extract_pdf_content(pdf_path, split_size=split_size)
    tokenized_docs = [doc.lower().split() for doc in documents]

    bm25 =BM25Plus(tokenized_docs, b=0.75, k1=2)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Semantic engine
    doc_embeddings = model.encode(documents)

    return model, bm25, doc_embeddings, documents

def normalize(score):
    if score.size > 0:
        norm_score = (score - min(score) / max(score) - min(score))
    else:
        norm_score = 0
    print(norm_score)
    return norm_score

# 3. Perform Hybrid Search
def hybrid_search(pdf_path, query, keywords, k=3, split_size=20):
    # Keyword scores
    model, bm25, doc_embeddings, documents = extract_and_search(pdf_path, split_size=split_size)
    bm25_scores = bm25.get_scores(keywords.split())
    # Semantic scores (cosine similarity)
    query_vec = model.encode([query])
    semantic_scores = np.dot(doc_embeddings, query_vec.T).flatten()

    # Combine scores (simple weighted average)
    hybrid_scores = ((0.5 * max(bm25_scores)) + (0.5 * semantic_scores))

    # Get top k results
    top_indices = np.argsort(hybrid_scores)[-k:][::-1]
    hybrid_scores_sorted = sorted(hybrid_scores, reverse=True)
    print('=' * 80)
    print('Hybrid Search Results')
    print(hybrid_scores[:10])
    hybrid_result = []
    for index in top_indices:
        doc = documents[index]
        score = hybrid_scores_sorted[index]
        hybrid_result.append({
            "score": score,
            "doc": doc,
        })
    print(hybrid_result)
    print('#' * 80)

    return hybrid_result

def search_all_pdfs(dir_path, query, keywords, k=3, split_size=20):
    files = get_pdf_files(dir_path)
    overall_result = []
    for file in files:
        print(file)
        findings =  hybrid_search(file, query,  keywords)
        overall_result.append(findings)
    sorted_result = sort_by_score(overall_result)
    return sorted_result

def print_user_friendly(results, num_entries=5):
    index = 0
    for result in results:
        if index < num_entries:
            print(result.get("doc"))
        index += 1





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




def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())




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



if __name__ == '__main__':

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

