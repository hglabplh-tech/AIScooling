"""
LSTM Text Processing Framework mit:
- Dynamischem Vokabular
- Tokenizer
- Embedding + LSTM + Logits Output
- Dynamischem Padding/Shaping
- Save/Load von:
    - vocab
    - state_dict
    - vollständigem Modell
- Pretrained Loader
- ChromaDB Integration
- Retrieval-Augmented Chat Client
- Relevance Score
- Erweiterbares Vokabular für weiteres Training

Benötigt:
pip install torch chromadb
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import chromadb
from chromadb.config import Settings


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# TOKENIZER
# =========================================================

class DynamicTokenizer:
    """
    Dynamisch erweiterbarer Tokenizer.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self):
        self.token_to_id = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }

        self.id_to_token = {
            v: k for k, v in self.token_to_id.items()
        }

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def normalize(self, text: str) -> str:
        return text.lower().strip()

    def tokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        return text.split()

    def add_tokens_from_text(self, text: str):
        tokens = self.tokenize(text)

        for token in tokens:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        expand_vocab: bool = True
    ) -> List[int]:

        if expand_vocab:
            self.add_tokens_from_text(text)

        tokens = self.tokenize(text)

        ids = []

        if add_special_tokens:
            ids.append(self.token_to_id[self.BOS_TOKEN])

        for token in tokens:
            ids.append(
                self.token_to_id.get(
                    token,
                    self.token_to_id[self.UNK_TOKEN]
                )
            )

        if add_special_tokens:
            ids.append(self.token_to_id[self.EOS_TOKEN])

        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []

        for idx in ids:
            token = self.id_to_token.get(idx, self.UNK_TOKEN)

            if token in {
                self.PAD_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
            }:
                continue

            tokens.append(token)

        return " ".join(tokens)

    def save_vocab(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_vocab(cls, path: str):
        tok = cls()

        with open(path, "r", encoding="utf-8") as f:
            tok.token_to_id = json.load(f)

        tok.id_to_token = {
            int(v): k for k, v in tok.token_to_id.items()
        }

        return tok


# =========================================================
# DATASET
# =========================================================

class TextDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        texts: List[str],
        tokenizer: DynamicTokenizer
    ):
        self.texts = texts
        self.tokenizer = tokenizer

        for text in texts:
            tokenizer.add_tokens_from_text(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        ids = self.tokenizer.encode(self.texts[idx])

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        return x, y


def collate_fn(batch):

    xs, ys = zip(*batch)

    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = pad_sequence(ys, batch_first=True, padding_value=0)

    return xs, ys


# =========================================================
# MODEL
# =========================================================

class LSTMTextModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):

        """
        x shape:
        [batch, seq_len]

        output logits:
        [batch, seq_len, vocab_size]
        """

        emb = self.embedding(x)

        lstm_out, _ = self.lstm(emb)

        logits = self.fc(lstm_out)

        return {
            "embeddings": emb,
            "logits": logits
        }

    def resize_token_embeddings(self, new_vocab_size: int):

        old_embedding = self.embedding
        old_fc = self.fc

        old_vocab_size, emb_dim = old_embedding.weight.shape

        if new_vocab_size <= old_vocab_size:
            return

        # Neues Embedding
        new_embedding = nn.Embedding(
            new_vocab_size,
            emb_dim,
            padding_idx=0
        )

        new_embedding.weight.data[:old_vocab_size] = (
            old_embedding.weight.data
        )

        self.embedding = new_embedding.to(DEVICE)

        # Neues Output Layer
        hidden_size = old_fc.in_features

        new_fc = nn.Linear(hidden_size, new_vocab_size)

        new_fc.weight.data[:old_vocab_size] = (
            old_fc.weight.data
        )

        new_fc.bias.data[:old_vocab_size] = (
            old_fc.bias.data
        )

        self.fc = new_fc.to(DEVICE)

    def save(
        self,
        model_dir: str,
        tokenizer: DynamicTokenizer
    ):

        os.makedirs(model_dir, exist_ok=True)

        # state dict
        torch.save(
            self.state_dict(),
            os.path.join(model_dir, "model_state.pt")
        )

        # komplettes Modell
        torch.save(
            self,
            os.path.join(model_dir, "full_model.pt")
        )

        # vocab
        tokenizer.save_vocab(
            os.path.join(model_dir, "vocab.json")
        )

        # config
        config = {
            "vocab_size": tokenizer.vocab_size
        }

        with open(
            os.path.join(model_dir, "config.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(config, f, indent=2)

    @classmethod
    def pretrained(
        cls,
        model_dir: str,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
    ):

        tokenizer = DynamicTokenizer.load_vocab(
            os.path.join(model_dir, "vocab.json")
        )

        model = cls(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        state = torch.load(
            os.path.join(model_dir, "model_state.pt"),
            map_location=DEVICE
        )

        model.load_state_dict(state)

        model.to(DEVICE)
        model.eval()

        return model, tokenizer


# =========================================================
# TRAINING
# =========================================================

def train_model(
    model: LSTMTextModel,
    dataset: TextDataset,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3
):

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x)

            logits = out["logits"]

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1} "
            f"Loss: {total_loss / len(loader):.4f}"
        )


# =========================================================
# CHROMA DATABASE
# =========================================================

class ChromaTextDB:

    def __init__(
        self,
        persist_dir: str = "./chroma_db"
    ):

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name="knowledge"
        )

    def add_documents(
        self,
        docs: List[str]
    ):

        ids = [
            f"doc_{i}"
            for i in range(self.collection.count(),
                           self.collection.count() + len(docs))
        ]

        self.collection.add(
            documents=docs,
            ids=ids
        )

    def search(
        self,
        query: str,
        top_k: int = 3
    ):

        result = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        return result


# =========================================================
# TEXT GENERATION
# =========================================================

@torch.no_grad()
def generate_text(
    model: LSTMTextModel,
    tokenizer: DynamicTokenizer,
    prompt: str,
    max_new_tokens: int = 30
):

    model.eval()

    ids = tokenizer.encode(prompt)

    x = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    generated = ids.copy()

    for _ in range(max_new_tokens):

        out = model(x)

        logits = out["logits"]

        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1
        ).item()

        generated.append(next_token)

        if next_token == tokenizer.token_to_id[
            tokenizer.EOS_TOKEN
        ]:
            break

        x = torch.tensor(
            [generated],
            dtype=torch.long
        ).to(DEVICE)

    return tokenizer.decode(generated)


# =========================================================
# CHAT CLIENT
# =========================================================

@dataclass
class ChatRequest:
    instruction: str
    question: str


class RetrievalChatClient:

    def __init__(
        self,
        model: LSTMTextModel,
        tokenizer: DynamicTokenizer,
        db: ChromaTextDB
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.db = db

    def ask(
        self,
        data: Dict[str, str]
    ) -> Dict[str, Any]:

        instruction = data["instruction"]
        question = data["question"]

        # Retrieval
        result = self.db.search(question)

        docs = result["documents"][0]
        distances = result["distances"][0]

        context = "\n".join(docs)

        relevance = 1.0

        if len(distances) > 0:
            relevance = float(1 / (1 + distances[0]))

        # Generationsprompt
        prompt = f"""
ANWEISUNG:
{instruction}

KONTEXT:
{context}

FRAGE:
{question}

ANTWORT:
"""

        generated = generate_text(
            self.model,
            self.tokenizer,
            prompt
        )

        return {
            "relevance": relevance,
            "generated_text": generated,
            "retrieved_docs": docs
        }


# =========================================================
# EXAMPLE
# =========================================================

if __name__ == "__main__":

    texts = [
        "PyTorch ist ein Deep Learning Framework",
        "LSTM Modelle verarbeiten Sequenzen",
        "ChromaDB ist eine Vektor Datenbank",
        "Embeddings repräsentieren Text numerisch",
    ]

    tokenizer = DynamicTokenizer()

    dataset = TextDataset(
        texts,
        tokenizer
    )

    model = LSTMTextModel(
        vocab_size=tokenizer.vocab_size
    ).to(DEVICE)

    # TRAIN
    train_model(
        model,
        dataset,
        epochs=5
    )

    # SAVE
    model.save(
        "./saved_model",
        tokenizer
    )

    # LOAD PRETRAINED
    model, tokenizer = (
        LSTMTextModel.pretrained(
            "./saved_model"
        )
    )

    # Dynamisches Vokabular erweitern
    new_text = "Transformer Modelle nutzen Attention"

    tokenizer.add_tokens_from_text(new_text)

    model.resize_token_embeddings(
        tokenizer.vocab_size
    )

    # CHROMA DB
    db = ChromaTextDB()

    db.add_documents([
        "LSTM kann Text generieren",
        "ChromaDB speichert Embeddings",
        "PyTorch unterstützt GPU Training"
    ])

    # CHAT CLIENT
    client = RetrievalChatClient(
        model,
        tokenizer,
        db
    )

    response = client.ask({
        "instruction":
            "Beantworte technisch präzise",
        "question":
            "Wie funktioniert ChromaDB?"
    })

    print("\n=== RESPONSE ===")
    print(json.dumps(
        response,
        indent=2,
        ensure_ascii=False
    ))