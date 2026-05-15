import os
from typing import Mapping, Any, Optional

import torch.nn as nn
import torch
import json
import re
from collections import Counter
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from RAG_ASAG.utilities.RAGUtils import get_model_path, extract_csv_data
from transformers import PretrainedConfig
from RAG_ASAG.utilities.RAGUtils import extract_doc_from_pdf
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, unpack_sequence

#Some constant values
LR     = 2e-5
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class YALLSTMModel(nn.Module):
    def __init__(self,
                 input_ids=[],
                 attention_mask=[],
                 labels=[],
                 input_dim=10,
                 embed_dim = 10,
                 hidden_dim=64,
                 vocab_size=7000,
                 layer_dim=4,
                 output_dim=1,
                 softmax=False):
        super(YALLSTMModel, self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        # LSTM Layer
        # batch_first=True ensures input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)

        # Fully connected layer to convert hidden state to final output
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = softmax
        self.log_softmax = nn.LogSoftmax(dim=2)


    @classmethod
    def split_data(cls, dataset):
        total = len(dataset)
        train_size = int(0.7 * total)
        val_size = int(0.15 * total)
        test_size = total - train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        return train_ds, val_ds, test_ds

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None
                ):

        fc_lin = nn.Linear(self.hidden_dim, self.output_dim)
        if self.softmax:
            out, _ = self.lstm(x)
            # Take the last time step
            logits = fc_lin(out[:, -1, :])

            # Convert raw logits to probabilities [0, 1]
            probs = self.log_softmax(logits)
            return probs
        else:
            #x = self.embedding(x)
            #out, _ = self.lstm(x)
            # We give only back the last logits of the epoch
            #logits = self.fc(out[:, -1, :])
            #return logits
            # Initializing hidden state (h0) and cell state (c0) with zeros
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

            b1 = []
            b2 = []
            size = int(len(x) /2)
            size_2 = len(x) - size
            for i in range(len(x)):
                if (i <= size):
                    b1.append(x[i])
                else:
                    b2.append(x[i])
            print(b1)
            print(b2)
            b1_tensor = self.make_list_tensors(b1, dtype=torch.float)
            b2_tensor = self.make_list_tensors(b2, dtype=torch.float)
            if (len(b2) == 0):
                input_to_lstm = [b1_tensor]
            else:
                input_to_lstm = [b1_tensor, b2_tensor]
            lstm_input = self.get_packed_tensors(input_to_lstm)
            print(lstm_input)
            fc_lin = nn.Linear(len(x), 1)
            self.fc = fc_lin
            #out: tensor of shape (batch_size, seq_length, hidden_dim)
            lstm_call = nn.LSTM(input_size=1, hidden_size=len(x), batch_first=True, bidirectional=False)
            packed_output, (hn, cn) = lstm_call(lstm_input)
            output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
            # We only need the last time step's output for the final prediction
            # out[:, -1, :] extracts (self.batch_size, self.hidden_dim)
            out = fc_lin(output[:,-1,  :])
            return out

    @classmethod
    def get_packed_tensors(cls, sequences):

        # Track actual sequence lengths as a CPU integer tensor or list
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int64)

        # 2. PAD THE SEQUENCES FIRST
        # Puts them into a regular 3D tensor tensor of shape: (batch_size, max_seq_len, features)
        # Here features = 1 for simple scalar lists
        padded_seqs =nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
        padded_seqs = padded_seqs.unsqueeze(-1)  # Shape becomes: (3, 5, 1)

        # 3. PACK THE PADDED SEQUENCES
        # enforce_sorted=False allows PyTorch to handle unsorted sequence lengths automatically
        packed_input = pack_padded_sequence(
            padded_seqs,
            lengths,
            batch_first=True,
            enforce_sorted=False)
        return packed_input

    @classmethod
    def make_list_tensors(cls, raw_list, dtype=torch.long):
        list_tensor = torch.tensor(raw_list, dtype=dtype)
        indexes = []
        for i in range(len(raw_list)):
            indexes.append(i)
        list_tensor = list_tensor[torch.tensor(indexes, dtype=torch.long)]
        print(list_tensor)
        return list_tensor

    @classmethod
    def prepare_data_for_train(cls,  pt_model_path,  dataset):
        train_ds, val_ds, test_ds = cls.split_data(dataset)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        label_size = 0
        for batch in train_loader:
            ids, mask, lbl = batch
            print("\n── Sample Batch ──────────────────────────")
            print("input_ids     :", ids.shape)  # (B, MAX_LEN)
            print("attention_mask:", mask.shape)  # (B, MAX_LEN)
            print("labels        :", lbl)  # (B,)
            label_size = lbl.size
            break
        print(f"Label size: {label_size}")
        model, config =YALLSTMModel.load_model(base_model_path, pt_model_path)
        return model, train_loader, val_loader, test_loader

    def save_model_conf(self, model_base_path, num_classes= 1):
        # 1. Define the dictionary structure
        lstm_config = YALLSTMModelConfig(vocab_size=self.vocab_size, embed_dim=self.embed_dim, num_layers=self.layer_dim, dropout=0.2,
                                         hidden_dim=self.hidden_dim,num_classes=num_classes, bidirectional=False)
        lstm_config.save_pretrained(model_base_path)

    def train_model(self, model, train_loader: DataLoader[tuple[Tensor, ...]],val_loader: DataLoader[tuple[Tensor, ...]], epochs=5):

        # ─────────────────────────────────────────
        # 6. Inspect a Batch
        # ─────────────────────────────────────────

        criterion = nn.MSELoss()  # Waits for Logits!
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR)
        losses = []
        # Training Loop (simplified)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for ids, mask, lbl in train_loader:
                ids, mask, lbl = ids.to(DEVICE), mask.to(DEVICE), lbl.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(ids, mask, lbl)
                print(outputs)
                target = [[1]] * len(outputs)
                #unpacked.append(items2)
                target_tens = self.make_list_tensors(target, dtype=torch.float)
                print(target_tens)
                loss = criterion(outputs, target_tens)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # ── Validate ──
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for ids, mask, lbl in val_loader:
                    ids, mask, lbl = ids.to(DEVICE), mask.to(DEVICE), lbl.to(DEVICE)
                    logits = model(ids, mask)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == lbl).sum().item()
                    total += lbl.size(0)

            val_acc = correct / total if total > 0 else 0.0
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")

    def save_state_dict(self, pkl_path):
        torch.save(self.state_dict(), pkl_path)

    def save_model(self, model_base_path, model_path):
        self.save_model_conf(model_base_path, num_classes=1)
        torch.save(self, model_path)

    @classmethod
    def load_model(cls, model_base_path, model_path):
        config = YALLSTMModelConfig.from_pretrained(model_base_path)
        return torch.load(model_path, map_location="cpu", weights_only=False), config


# 1. Define the custom configuration class
class  YALLSTMModelConfig(PretrainedConfig):
    model_type = "custom_lstm"  # Must match your model's identifier

    def __init__(
        self,
        vocab_size=30522,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        num_classes=3,
        dropout=0.2,
        bidirectional=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional

class YATokenizer:
    def __init__(self,  vocab=None, special_tokens=["<PAD>", "<UNK>"],max_length=10, pad_token="<PAD>", unk_token="<UNK>"):
        self.max_length = max_length
        self.vocab = vocab
        self.id_to_word = {}
        self.special_tokens = special_tokens
        self.max_length = max_length
        self.pad_token = pad_token
        self.unk_token = unk_token

    def build_vocab(self, corpus, append=False):
        # Flatten all text and count word frequencies
        words = " ".join(corpus).split()
        counts = Counter(words)
        unique_words = self.special_tokens + sorted(list(counts.keys()))
        if append:
            all_ids = list(self.vocab.values())
            next_id = max(all_ids) + 1 if all_ids else 0

            new_tokens = []

            for token, _ in self.vocab.items():
                if token not in self.vocab:
                    self.vocab[token] = counts.get(token)
                    new_tokens.append(token)
                    next_id += 1
        else:
            # Add special tokens first, then unique words
            self.vocab = {word: i for i, word in enumerate(unique_words)}
            self.id_to_word = {i: word for word, i in self.vocab.items()}

    # Assuming 'tokenizer' is the object from the previous example
    def to_json(self, file_path):
        tokenizer_data = {
            "vocab": self.vocab,
            "special_tokens": self.special_tokens
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)
            print("Tokenizer saved to {}".format(file_path))

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(vocab=data["vocab"], special_tokens=data["special_tokens"])
        print('Tokenizer created from {}'.format(file_path))
        return instance

    def encode(self, text, return_tensors="pt"):
        unk_id = self.vocab.get("<UNK>")
        ids = [self.vocab.get(word, unk_id) for word in text.split()]
        if return_tensors == "pt":
            # Returns a 1D tensor: [seq_len]
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def batch_encode(self, texts, max_len=10):
        batch_ids = []
        pad_id = self.vocab.get("<PAD>", 0)

        for text in texts:
            ids = self.encode(text, return_tensors='none')  # get list

            # Padding/Truncating
            if len(ids) < max_len:
                ids += [pad_id] * (max_len - len(ids))
            else:
                ids = ids[:max_len]

            batch_ids.append(ids)

        # Returns a 2D tensor: [batch_size, max_len]
        return torch.tensor(batch_ids, dtype=torch.long)

    def decode(self, token_ids, skip_special_tokens=True):
        if torch.is_tensor(token_ids):
            token_ids = token_ids.cpu().tolist()

        # Helper to filter tokens
        def clean(ids):
            words = []
            for i in ids:
                word = self.id_to_word.get(i, "<UNK>")
                if skip_special_tokens and word in self.special_tokens:
                    continue
                words.append(word)
            return " ".join(words)

        # Handle Batch or Single
        if isinstance(token_ids[0], int):
            return clean(token_ids)
        return [clean(seq) for seq in token_ids]

    def _clean_and_tokenize(self, text):
        """Lowercases text and splits it into words/punctuation fragments."""
        text = text.lower()
        # Splits by spaces and isolates punctuation marks
        return re.findall(r"\w+|[^\w\s]", text)

    def __call__(self, text):
        """Tokenizes a single string, pads/truncates it, and generates masks."""
        raw_tokens = self._clean_and_tokenize(text)

        # 1. Convert text tokens to unique Vocabulary IDs
        input_ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in raw_tokens]

        # 2. Create the raw attention mask (1 for real data tokens)
        attention_mask = [1] * len(input_ids)

        # 3. Handle Truncation (if sequence is too long)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # 4. Handle Padding (if sequence is too short)
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids += [self.vocab[self.pad_token]] * padding_length
            attention_mask += [0] * padding_length
            attention_mask = [attention_mask] # 0 marks padded positions to be ignored

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class YAMultDataset:
    def __init__(self, corpus, tokenizer_path):
        self.corpus = corpus
        self.tokenizer = get_tokenizer(tokenizer_path)
        self.encoded = tokenizer(self.corpus)


    def tensor_dataset(self):
        return TensorDataset(self.encoded)

class YALLMInferencePipeline:
    def __init__(self, model_path, tokenizer_path, config_path):
        # 1. Device bestimmen (GPU falls verfügbar)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Konfiguration laden
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # 3. Tokenizer laden
        # (Nutzt die .from_json Methode aus den vorherigen Schritten)
        self.tokenizer = YATokenizer.from_json(tokenizer_path)

        # 4. Modell laden und Gewichte zuweisen
        # (Nutzt die LSTMModel-Klasse aus den vorherigen Schritten)
        self.model = YALLSTMModel(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config["hidden_dim"]
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Wichtig: Schaltet Dropout/Batchnorm für Inferenz aus

    def __call__(self, text, return_probabilities=False):
        """
        Ermöglicht es, die Pipeline wie eine Funktion aufzurufen: pipeline("text")
        """
        # 1. Text zu Tensor konvertieren und auf das richtige Device schieben
        input_tensor = self.tokenizer.batch_encode(
            [text],
            max_len=self.config["max_len"]
        ).to(self.device)

        # 2. Inferenz ohne Gradientenberechnung
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1)

        # 3. ID zurück in Text decodieren
        predicted_text = self.tokenizer.decode(predicted_id)[0]

        # 4. Rückgabe formatieren
        if return_probabilities:
            confidence = probs[0, predicted_id.item()].item()
            return {"prediction": predicted_text, "confidence": confidence}

        return predicted_text


class YATextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=10):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx, return_tensors="pt"):
        # Tokenize the text
        token_ids = self.tokenizer.encode(self.texts[idx])

        # Simple Padding/Truncation to fixed length
        if len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.vocab["<PAD>"]] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]

        if return_tensors == "pt":
            # Returns a 1D tensor: [seq_len]
            return torch.tensor(token_ids, dtype=torch.long)

        return token_ids


 # 2. Define a processing function
def tokenize_function(examples):
   # This uses a pre-built tokenizer (like BERT) or your custom one
   # If using your custom tokenizer, ensure it supports list inputs
    return {"input_ids": [tokenizer.encode(text) for text in examples["text"]]}

def tokenize_function_tensors(examples):
   # This uses a pre-built tokenizer (like BERT) or your custom one
   # If using your custom tokenizer, ensure it supports list inputs
    return {"input_ids": [tokenizer.encode(text, return_tensors='pt') for text in examples["text"]]}

def get_tokenizer(tokenizer_path):
    if os.path.exists(tokenizer_path):
        tokenizer = YATokenizer.from_json(tokenizer_path)
    else:
        tokenizer = YATokenizer()
    return tokenizer

def load_create_model(base_model_path, pt_model_path):
    if (os.path.exists(pt_model_path)):
        model, config = YALLSTMModel.load_model(model_base_path=base_model_path, model_path=pt_model_path)
        print(f"Model loaded from: {pt_model_path}")
    else:
        model = YALLSTMModel(input_dim=10, hidden_dim=64, layer_dim=3, output_dim=1)
        model.save_model(os.path.join(pt_model_path))
        print("Model instance created")
    return model


if __name__ == '__main__':
    # --- Setup and Usage ---
    _, base_model_path = get_model_path()
    pt_model_path = os.path.join(base_model_path, 'YALSTMModel.pt')
    # Params: 10 input features, 32 hidden units, 2 stacked layers, 1 output value
    model = load_create_model(base_model_path, pt_model_path)

    # Dummy Input: (Batch Size=8, Sequence Length=5, Features=10)

    dummy = torch.randn(3, 8, 5)
    print(dummy)
    # Forward pass

    # Usage of tokenizer
    vocab_data = []
    raw_data = extract_doc_from_pdf(file_path="./test.pdf", as_doc=False)
    vocab_data.append(raw_data)
    raw_data = extract_doc_from_pdf(file_path="./test_two.pdf", as_doc=False)
    vocab_data.append(raw_data)
    raw_data = extract_doc_from_pdf(file_path="./test_three.pdf", as_doc=False)
    vocab_data.append(raw_data)
    raw_data = extract_doc_from_pdf(file_path="./test_four.pdf", as_doc=False)
    vocab_data.append(raw_data)
    input_vocab = vocab_data[0] + vocab_data[1] + vocab_data[2] + vocab_data[3]
    print(input_vocab)
    tokenizer = get_tokenizer("./YALLMModelSuite.json")
    tokenizer.build_vocab(raw_data)
    dataset = YATextDataset(raw_data, tokenizer)
    ds = load_dataset("csv", data_files="./lang_words.csv")
    ds_str = ds.data.__str__()
    encoded = tokenizer(ds_str)
    input_ids = encoded["input_ids"]  # (N, MAX_LEN)
    attention_mask = encoded["attention_mask"]
    print(input_ids)
    print(attention_mask)
    labels = [0] * len(input_ids)
    label_tensor = torch.tensor(labels)  # (N,)
    ids_tensor = torch.tensor(input_ids)
    attn_tensor = torch.tensor(attention_mask)
    print(ids_tensor)
    print(attn_tensor)
    print(label_tensor)
    tokenized_ds =  TensorDataset(ids_tensor, attn_tensor, label_tensor)
    training_model, train_loader, val_loader, test_loader = YALLSTMModel.prepare_data_for_train(pt_model_path, tokenized_ds)
    model.train_model(training_model,train_loader, val_loader, epochs=15)

    model.save_model(base_model_path, os.path.join(pt_model_path))
    model.save_state_dict(os.path.join(base_model_path, 'YALSTMModel.pkl'))
    # 1. Load your local data or a hub dataset




    tokenizer.to_json(file_path="./YALLMModelSuite.json")
    # Usage
    tokenizer = get_tokenizer("./YALLMModelSuite.json")
    encoded = tokenizer.encode("hello world here I am walking like a hurricane with ice in my eyes")
    decoded = tokenizer.decode(encoded)
    tokenized_ds = ds.map(tokenize_function, batched=True)
    tokenized_ds.set_format(type="torch", columns=["input_ids"])
    print(decoded)
