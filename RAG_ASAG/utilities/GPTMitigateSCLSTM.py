# mitigate_shortcut_lstm_chat.py
import random
import re
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1. Data augmentation against shortcuts
# -----------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def mask_spurious_tokens(text: str, p: float = 0.15) -> str:
    """
    Randomly masks superficial tokens so the model cannot rely too much on
    single trigger words like 'always', 'because', 'yes', 'no', etc.
    """
    tokens = text.split()
    out = []
    for tok in tokens:
        if random.random() < p:
            out.append("<mask>")
        else:
            out.append(tok)
    return " ".join(out)


def shuffle_lightly(text: str, p: float = 0.10) -> str:
    """
    Light token shuffling reduces dependence on fixed phrase position.
    Use lightly, because too much destroys meaning.
    """
    tokens = text.split()
    for i in range(len(tokens) - 1):
        if random.random() < p:
            tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
    return " ".join(tokens)


def augment_question(question: str) -> str:
    question = normalize_text(question)
    question = mask_spurious_tokens(question, p=0.10)
    question = shuffle_lightly(question, p=0.05)
    return question


# -----------------------------
# 2. Simple tokenizer / vocab
# -----------------------------

class SimpleTokenizer:
    def __init__(self):
        self.special = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]
        self.word2id = {w: i for i, w in enumerate(self.special)}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def build_vocab(self, texts: List[str], min_freq: int = 1):
        freq = {}
        for text in texts:
            for tok in normalize_text(text).split():
                freq[tok] = freq.get(tok, 0) + 1

        for word, count in freq.items():
            if count >= min_freq and word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = ["<bos>"] + normalize_text(text).split() + ["<eos>"]
        ids = [self.word2id.get(t, self.word2id["<unk>"]) for t in tokens]
        ids = ids[:max_len]
        ids += [self.word2id["<pad>"]] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            w = self.id2word.get(int(i), "<unk>")
            if w in ["<pad>", "<bos>", "<eos>"]:
                continue
            words.append(w)
        return " ".join(words)


# -----------------------------
# 3. Dataset with counterfactual examples
# -----------------------------

class ChatDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer: SimpleTokenizer,
        max_q_len: int = 64,
        max_a_len: int = 64,
        augment: bool = True
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, a = self.pairs[idx]

        if self.augment:
            q = augment_question(q)

        q_ids = self.tokenizer.encode(q, self.max_q_len)
        a_ids = self.tokenizer.encode(a, self.max_a_len)

        return {
            "question": torch.tensor(q_ids, dtype=torch.long),
            "answer": torch.tensor(a_ids, dtype=torch.long)
        }


# -----------------------------
# 4. LSTM encoder-decoder chat model
# -----------------------------

class LSTMChatModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.30,
        pad_id: int = 0
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        self.encoder = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.decoder = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, question_ids, answer_ids):
        q_emb = self.embedding(question_ids)
        _, hidden = self.encoder(q_emb)

        a_emb = self.embedding(answer_ids[:, :-1])
        dec_out, _ = self.decoder(a_emb, hidden)

        logits = self.output(dec_out)
        return logits


# -----------------------------
# 5. Shortcut mitigation losses
# -----------------------------

def consistency_loss(logits_clean, logits_aug):
    """
    Forces model to produce similar predictions for clean and augmented inputs.
    This discourages reliance on superficial wording.
    """
    p_clean = torch.softmax(logits_clean.detach(), dim=-1)
    log_p_aug = torch.log_softmax(logits_aug, dim=-1)
    return nn.functional.kl_div(log_p_aug, p_clean, reduction="batchmean")


def confidence_penalty(logits):
    """
    Penalizes overconfident predictions.
    Shortcut learning often appears as brittle overconfidence.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return -entropy


# -----------------------------
# 6. Training
# -----------------------------

@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    max_q_len: int = 64
    max_a_len: int = 64
    consistency_weight: float = 0.30
    confidence_weight: float = 0.01
    grad_clip: float = 1.0


def train_model(pairs: List[Tuple[str, str]], config: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = SimpleTokenizer()
    all_text = []
    for q, a in pairs:
        all_text.append(q)
        all_text.append(a)

    tokenizer.build_vocab(all_text)

    dataset_clean = ChatDataset(
        pairs,
        tokenizer,
        config.max_q_len,
        config.max_a_len,
        augment=False
    )

    dataset_aug = ChatDataset(
        pairs,
        tokenizer,
        config.max_q_len,
        config.max_a_len,
        augment=True
    )

    loader_clean = DataLoader(dataset_clean, batch_size=config.batch_size, shuffle=True)
    loader_aug = DataLoader(dataset_aug, batch_size=config.batch_size, shuffle=True)

    model = LSTMChatModel(
        vocab_size=len(tokenizer.word2id),
        pad_id=tokenizer.word2id["<pad>"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.word2id["<pad>"])

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0

        for clean_batch, aug_batch in zip(loader_clean, loader_aug):
            q_clean = clean_batch["question"].to(device)
            a_clean = clean_batch["answer"].to(device)

            q_aug = aug_batch["question"].to(device)
            a_aug = aug_batch["answer"].to(device)

            logits_clean = model(q_clean, a_clean)
            logits_aug = model(q_aug, a_aug)

            target = a_clean[:, 1:].contiguous()

            main_loss = ce_loss(
                logits_clean.reshape(-1, logits_clean.size(-1)),
                target.reshape(-1)
            )

            c_loss = consistency_loss(logits_clean, logits_aug)
            conf_loss = confidence_penalty(logits_clean)

            loss = (
                main_loss
                + config.consistency_weight * c_loss
                + config.confidence_weight * conf_loss
            )

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: loss={total_loss / len(loader_clean):.4f}")

    return model, tokenizer


# -----------------------------
# 7. Generation
# -----------------------------

@torch.no_grad()
def generate_answer(model, tokenizer, question: str, max_len: int = 50):
    device = next(model.parameters()).device
    model.eval()

    q_ids = torch.tensor(
        [tokenizer.encode(question, max_len=64)],
        dtype=torch.long,
        device=device
    )

    q_emb = model.embedding(q_ids)
    _, hidden = model.encoder(q_emb)

    current = torch.tensor(
        [[tokenizer.word2id["<bos>"]]],
        dtype=torch.long,
        device=device
    )

    generated = []

    for _ in range(max_len):
        emb = model.embedding(current)
        out, hidden = model.decoder(emb, hidden)
        logits = model.output(out[:, -1])

        # temperature sampling avoids always choosing shortcut answers
        probs = torch.softmax(logits / 0.9, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        token_id = int(next_id.item())

        if token_id == tokenizer.word2id["<eos>"]:
            break

        generated.append(token_id)
        current = next_id

    return tokenizer.decode(generated)


# -----------------------------
# 8. Example
# -----------------------------

if __name__ == "__main__":
    training_pairs = [
        (
            "What is shortcut learning?",
            "Shortcut learning is when a model learns superficial patterns instead of the intended reasoning."
        ),
        (
            "How can shortcut learning be reduced?",
            "It can be reduced with balanced data, counterfactual examples, augmentation, regularization and consistency training."
        ),
        (
            "Why is biased training data dangerous?",
            "Biased data can cause the model to rely on accidental correlations and fail on new examples."
        ),
        (
            "What helps a chat model generalize?",
            "Diverse examples, paraphrases, hard negatives and evaluation on unseen data help generalization."
        ),
    ]

    config = TrainConfig(epochs=20, batch_size=2)
    model, tokenizer = train_model(training_pairs, config)

    print(generate_answer(model, tokenizer, "How do I prevent shortcut learning?"))

    torch.save(
        {
            "model_state": model.state_dict(),
            "word2id": tokenizer.word2id,
            "id2word": tokenizer.id2word,
        },
        "lstm_chat_shortcut_mitigated.pt"
    )