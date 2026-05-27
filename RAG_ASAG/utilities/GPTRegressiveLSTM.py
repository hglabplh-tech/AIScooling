# autoregressive_lstm_train.py

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


TEXTS = [
    "user: hallo bot: hallo wie geht es dir",
    "user: wie geht es dir bot: mir geht es gut",
    "user: wer bist du bot: ich bin ein lstm chatbot",
    "user: was kannst du bot: ich kann einfache antworten geben",
]

SEQ_LEN = 12
BATCH_SIZE = 4
EPOCHS = 300
LR = 0.003


def build_vocab(texts, path="vocab.json"):
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
    }

    for text in texts:
        for token in text.lower().split():
            if token not in vocab:
                vocab[token] = len(vocab)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    return vocab


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.pad_id = vocab["<PAD>"]
        self.unk_id = vocab["<UNK>"]
        self.bos_id = vocab["<BOS>"]
        self.eos_id = vocab["<EOS>"]

    def encode(self, text):
        ids = [self.bos_id]
        ids += [self.vocab.get(t, self.unk_id) for t in text.lower().split()]
        ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        tokens = []

        for idx in ids:
            token = self.id_to_token.get(int(idx), "<UNK>")

            if token in ["<PAD>", "<BOS>"]:
                continue
            if token == "<EOS>":
                break

            tokens.append(token)

        return " ".join(tokens)


class AutoregressiveDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.samples = []
        self.pad_id = tokenizer.pad_id

        for text in texts:
            ids = tokenizer.encode(text)

            for i in range(1, len(ids)):
                context = ids[max(0, i - seq_len):i]
                target = ids[i]

                while len(context) < seq_len:
                    context.insert(0, self.pad_id)

                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        context, target = self.samples[index]

        attention_mask = [0 if x == self.pad_id else 1 for x in context]

        return {
            "input_ids": torch.tensor(context, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.long),
        }


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)

        lstm_out, _ = self.lstm(embedded)

        mask = attention_mask.unsqueeze(-1)
        lstm_out = lstm_out * mask

        lengths = attention_mask.sum(dim=1).long()
        lengths = torch.clamp(lengths, min=1)

        last_indices = lengths - 1

        batch_indices = torch.arange(
            input_ids.size(0),
            device=input_ids.device
        )

        last_hidden = lstm_out[batch_indices, last_indices]

        logits = self.output(last_hidden)

        return logits


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = build_vocab(TEXTS)
    tokenizer = Tokenizer(vocab)

    dataset = AutoregressiveDataset(TEXTS, tokenizer, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMLanguageModel(vocab_size=len(vocab)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        model.train()

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)

            logits = model(input_ids, attention_mask)

            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 25 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "autoregressive_lstm.pt")

    print("Gespeichert: autoregressive_lstm.pt")
    print("Gespeichert: vocab.json")


def generate(model, tokenizer, prompt, max_new_tokens=30, temperature=0.8):
    device = next(model.parameters()).device

    model.eval()

    ids = tokenizer.encode(prompt)
    ids = ids[:-1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-SEQ_LEN:]

            while len(context) < SEQ_LEN:
                context.insert(0, tokenizer.pad_id)

            attention_mask = [0 if x == tokenizer.pad_id else 1 for x in context]

            input_ids = torch.tensor([context], dtype=torch.long).to(device)
            mask = torch.tensor([attention_mask], dtype=torch.float32).to(device)

            logits = model(input_ids, mask)
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            ids.append(next_id)

            if next_id == tokenizer.eos_id:
                break

    return tokenizer.decode(ids)


if __name__ == "__main__":
    train()