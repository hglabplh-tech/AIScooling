import os
import pypdf
import  random

from langchain_core.documents import Document
from rank_bm25 import BM25Plus
#from langchain_community.retrievers import  BM25Retriever
from sentence_transformers import SentenceTransformer
import numpy as np
from RAG_ASAG.utilities.BM25PlusDerived import BM25PlusDerived
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

if __name__ == '__main__':
    query = input("Enter query or exit: ")
    set_api_env_and_keys_in_parent()
    strict =  get_key_strict(query)
    num_hits = input("Enter number of hits: ")
    num_results = input("Enter number of results: ")
    while query != "exit":
        keywords, _ = extract_keywords(query, strict=strict)
        sorted_results = search_all_pdfs(get_collections_path('compscience'), query, keywords)
        print_results(sorted_results, num_hits=int(num_hits))
        print_user_friendly(sorted_results, num_entries=int(num_results))
        query = input("Enter query or exit: ")
        strict = get_key_strict(query)
        num_hits = input("Enter number of hits: ")
        num_results = input("Enter number of results: ")

