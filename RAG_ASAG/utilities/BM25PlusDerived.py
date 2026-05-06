from rank_bm25 import BM25Plus


class BM25PlusDerived(BM25Plus):
    def __init__(self, documents, b=0.75, k1= 2, delta=0):
        tokenized_docs = []
        for document in documents:
            text = document.get("page_content").lower()
            splitted = text.split()
            tokenized_docs.append(splitted)
        self.bm25 = BM25Plus.__init__(self, tokenized_docs, b=b, k1=k1, delta=delta)

    def get_scores(self, keywords):
        return self.bm25.get_scores(keywords)