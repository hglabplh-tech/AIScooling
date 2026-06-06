import os
from pathlib import Path
from typing import Mapping, Any, Optional

#######################################################################
## Yet another chat and LLM model
## (c) 2026 Harald Glab-Plhak
## email: hglabplhak@gmail.com
## MIT License
#######################################################################

import torch.nn as nn
import torch
import json
import re
from collections import Counter
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from tqdm import tqdm

from RAG_ASAG.utilities.RAGUtils import get_model_path, get_chat_model_basepath
from transformers import PretrainedConfig
from RAG_ASAG.utilities.RAGUtils import extract_doc_from_pdf
from torch.nn.utils.rnn import pack_padded_sequence

#Some constant values
LR     = 2e-5
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # select the device

class YALLSTMModel(nn.Module):
    def __init__(self,
                 input_ids=[],
                 attention_mask=[],
                 embedding_dim=10,
                 labels=[],
                 input_dim=10,
                 embed_dim = 1,
                 hidden_dim=256,
                 vocab_size=7000,
                 layer_dim=256,
                 num_classes=1,
                 padding_idx=0,
                 softmax=False):
        super(YALLSTMModel, self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        # LSTM Layer
        # batch_first=True ensures input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)

        # Fully connected layer to convert hidden state to final output
        self.softmax = softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

    def inject_vocab_size(self, vocab_size):
        debug_print(f"injecting vocab size is: {vocab_size}")
        self.vocab_size = vocab_size
        return self.get_vocab_size()

    def get_vocab_size(self):
         return self.vocab_size

    @classmethod
    def split_data(cls, dataset, train_part_size=0.8, val_part_size=0.1, test_part_size=0.1):
        total = len(dataset)
        train_size = int(train_part_size * total)

        val_size = int(val_part_size * total)
        test_size = int(test_part_size * total)

        if (train_size + val_size + test_size > total):
            train_part_size , val_part_size , test_part_size = 0.8, 1.0,1.0
            train_size , val_size , test_size = int(train_part_size * total), int(val_part_size * total), int(test_part_size * total)


        train_ds, val_ds, test_ds = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),  # greetings from 'hitchhikers from galaxy'
        )
        return train_ds, val_ds, test_ds

    # ==========================================
    # 4. RE-CONFIGURED TRAINING LOOP
    # ==========================================



    def forward(self, text_tensors, attention_mask):
        # 1. Sum up attention mask horizontally to find true length of each sequence
        lengths = attention_mask.sum().cpu()
        debug_print(f"FWD:Text Tensors -> {text_tensors}")
        debug_print(f"FWD:Attention Mask -> {attention_mask}")

        # 2. Transform token indices to word embeddings
        embedded = self.embedding(text_tensors).cpu()
        debug_print(f"Embedded Values: {embedded}")
        debug_print(f"Embedded Values Shape: {embedded.size()}")
        embedded_size = embedded.size()
        # 3. Pack sequence tightly based on lengths to prevent LSTM from updating on padding zeroes
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(
         #   embedded, lengths, batch_first=True, enforce_sorted=False
        #)
        packed_embedded = self.get_packed_tensors(embedded)
        debug_print(f"Packed embedded: {packed_embedded}")
        # in between initialize LSTM again
        self.lstm = nn.LSTM(self.num_classes, embedded_size[0], embedded_size[1], batch_first=True, bidirectional=False)
        # 4. Process sequence with LSTM
        packed_out, (hidden, cell) = self.lstm(packed_embedded)

        # 5. Restore full original padded context array
        out, lens = nn.utils.rnn.pad_packed_sequence(
                    packed_out, batch_first=True #, total_length=packed_out.size(-1)
            )
        print(lens)
        # 6. Target the exact index step where real words finished for every batch item
        batch_indices = torch.arange(text_tensors.size(0), device=DEVICE)
        last_word_indices = lengths.to(DEVICE) - 1
        last_hidden_state = out[batch_indices, last_word_indices, :]
        debug_print(f"batch indices: {batch_indices}")
        debug_print(f"last word indices: {last_word_indices}")
        debug_print(f"last hidden state: {last_hidden_state}")
        # 7. Convert sequence memory to logits mapping
        x = batch_indices.size()[0]
        y = lengths.to(DEVICE)
        debug_print(f"batch len: {x} - wordslen: {y}")
        self.fc = nn.Linear(x, y)
        return self.fc(last_hidden_state)

    @classmethod
    def get_packed_tensors(cls, sequences):

        # Track actual sequence lengths as a CPU integer tensor or list
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

        # 2. PAD THE SEQUENCES FIRST
        # Puts them into a regular 3D tensor tensor of shape: (batch_size, max_seq_len, features)
        # Here features = 1 for simple scalar lists
        padded_seqs =nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
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
        debug_print(list_tensor)
        return list_tensor

    @classmethod
    def prepare_data_for_train(cls,  pt_model_path,  dataset):
        train_ds, val_ds, test_ds = cls.split_data(dataset, train_part_size=0.7, val_part_size=0.1, test_part_size=0.2)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        label_size = 0
        for batch in train_loader:
            ids, mask, lbl = batch
            debug_print("\n── Sample Batch ──────────────────────────")
            debug_print(f"input_ids shape :{ids.shape}")  # (B, MAX_LEN)
            debug_print(f"attention_mask shape : {mask.shape}")  # (B, MAX_LEN)
            debug_print(f"attention_mask shape : {lbl.shape}")  # (B, MAX_LEN)
            label_size = lbl.shape[0]
            break
        debug_print(f"Label size: {label_size}")
        model, config =YALLSTMModel.load_model(base_model_path, pt_model_path)
        return model, train_loader, val_loader, test_loader

    def save_model_conf(self, model_base_path, num_classes= 1):
        # 1. Define the dictionary structure
        lstm_config = YALLSTMModelConfig(vocab_size=self.vocab_size, embed_dim=self.embed_dim, num_layers=self.layer_dim, dropout=0.2,
                                         hidden_dim=self.hidden_dim,num_classes=num_classes, bidirectional=False)
        lstm_config.save_pretrained(model_base_path)

    def train_model(self, model, train_loader: DataLoader[tuple[Tensor, ...]],
                    val_loader: DataLoader[tuple[Tensor, ...]], epochs=5):

        # ─────────────────────────────────────────
        # 6. Inspect a Batch
        # ─────────────────────────────────────────

        criterion = nn.CrossEntropyLoss() # Waits for Logits!
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        losses = []

        # ==========================================
        # 4. RE-CONFIGURED TRAINING LOOP
        # ==========================================

        model.train()
        for epoch in range(epochs):
            loop = tqdm(loader, total=len(train_loader), leave=True)
            total_loss = 0.0
            for batch_text, batch_mask, labels in loop:
                batch_text, batch_mask, labels = batch_text.to(DEVICE), batch_mask.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                logits = model(batch_text, batch_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_text.size(0)
                loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
                loop.set_postfix(loss=loss.item())

            print(f"Epoch {epoch + 1} | Loss: {total_loss / len(dataset):.4f}")

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
        num_classes=1,
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
    def __init__(self,  vocab=None,
                 max_length=10,
                 pad_token="<PAD>",
                 unk_token="<UNK>",
                 eos_token="<EOS>",
                 bos_token="<BOS>",
                 special_tokens = ["<PAD>", "<UNK>","<EOS>", "<BOS>"]):
        self.max_length = max_length
        self.vocab = vocab
        self.id_to_word = {}
        self.special_tokens = special_tokens
        self.max_length = max_length


        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

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
        return len(self.vocab)

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
        unk_id = self.vocab.get(self.unk_token)
        ids = [self.vocab.get(word, unk_id) for word in text.split()]
        if return_tensors == "pt":
            # Returns a 1D tensor: [seq_len]
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def batch_encode(self, texts, max_len=10):
        batch_ids = []
        pad_id = self.vocab.get(self.pad_token, 0)

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

    def clean_and_tokenize(self, text):
        """Lowercases text and splits it into words/punctuation fragments."""
        if isinstance(text, str):
            text = text.lower()
        elif isinstance(text, list):
            text = " ".join(text).lower()
        # Splits by spaces and isolates punctuation marks
        return re.findall(r"\w+|[^\w\s]", text)

    def __call__(self, text):
    #    debug_print(self.vocab.keys())
        """Tokenizes a single string, pads/truncates it, and generates masks."""
        raw_tokens = self.clean_and_tokenize(text)

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
            attention_mask = [attention_mask]
        # 0 marks padded positions to be ignored
        labels = torch.randint(0, 2, (len(input_ids),))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": labels
        }


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



class YALSTMChatModel(nn.Module):
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

    @classmethod
    def train_model(cls, tokenizer, dataset, chat_model_path, epochs=300):
        device = "cuda" if torch.cuda.is_available() else "cpu"


        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = YALSTMChatModel(vocab_size=len(tokenizer.vocab)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0

            model.train()
            loop = tqdm(loader, total=len(loader), leave=True)
            for batch in loop:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["target"].to(device)

                logits = model(input_ids, attention_mask)

                loss = loss_fn(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                # Update the progress bar with metrics
                loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
                loop.set_postfix(loss=loss.item())

           #if epoch % 25 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss / len(loader):.4f}")

        model.save_model(chat_model_path)


        print("Gespeichert: autoregressive_lstm.pt")
        print("Gespeichert: vocab.json")

    @classmethod
    def generate(cls, model, tokenizer, prompt, max_new_tokens=30, temperature=0.8, seq_len=12):
        device = next(model.parameters()).device

        model.eval()

        encoded = tokenizer(prompt)
        ids = encoded["input_ids"]
        debug_print(f"ids: {ids}")
        ids = ids[:-1]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = ids[-seq_len:]#TODO: SEQ_LEN

                while len(context) < seq_len: #TODO: SEQ_LEN
                    context.insert(0,  tokenizer.vocab[tokenizer.pad_token])

                attention_mask = [0 if x == tokenizer.vocab[tokenizer.pad_token] else 1 for x in context]

                input_ids = torch.tensor([context], dtype=torch.long).to(device)
                mask = torch.tensor([attention_mask], dtype=torch.float32).to(device)

                logits = model(input_ids, mask)
                logits = logits / temperature

                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

                ids.append(next_id)

                if next_id ==  tokenizer.vocab[tokenizer.eos_token]:
                    break

        return tokenizer.decode(ids)

    def save_state_dict(self, pkl_path):
        torch.save(self.state_dict(), pkl_path)

    def save_model(self, model_path):
       # self.save_model_conf(model_base_path, num_classes=1)
        torch.save(self, model_path)

    @classmethod
    def load_model(cls, model_path):
        return torch.load(model_path, map_location="cpu", weights_only=False)



class YAAutoregressiveDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=12):
        self.samples = []
        self.pad_id = tokenizer.vocab[tokenizer.pad_token]

        for text in texts:
            ids = tokenizer.encode(text, return_tensors="none")

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
            return torch.tensor(token_ids, dtype=torch.long).to(DEVICE)

        return token_ids


 # 2. Define a processing function
def tokenize_function(values):
   # This uses a pre-built tokenizer (like BERT) or your custom one
   # If using your custom tokenizer, ensure it supports list inputs
    return {"input_ids": [tokenizer.encode(text) for text in values["text"]]}

def tokenize_function_tensors(values):
   # This uses a pre-built tokenizer (like BERT) or your custom one
   # If using your custom tokenizer, ensure it supports list inputs
   if isinstance(values, list):
       return {"input_ids": [tokenizer.encode(text, return_tensors='pt') for text in values["text"]]}
   else:
       return {"input_ids": [tokenizer.encode(text, return_tensors='pt') for text in values]}


def get_tokenizer(tokenizer_path):
    if os.path.exists(tokenizer_path):
        debug_print('tokenizer loaded from {}'.format(tokenizer_path))
        tokenizer = YATokenizer.from_json(tokenizer_path)
    else:
        debug_print('tokenizer created new')
        tokenizer = YATokenizer()
    return tokenizer

def load_create_model(base_model_path, pt_model_path, vocab_size=7000):
    if (os.path.exists(pt_model_path)):
        model, config = YALLSTMModel.load_model(model_base_path=base_model_path, model_path=pt_model_path)
        model.inject_vocab_size(vocab_size)
        print(f"Model loaded from: {pt_model_path}")
    else:
        model = YALLSTMModel(input_dim=10, hidden_dim=64, layer_dim=3, embed_dim=2,
                             num_classes=1, vocab_size=vocab_size)
        model.save_model(base_model_path, pt_model_path)
        print("Model instance created")
    return model

def load_create_chat_model(pt_model_path, vocab_size=7000):
    if (os.path.exists(pt_model_path)):
        model = YALSTMChatModel.load_model(model_path=pt_model_path)
        #model.inject_vocab_size(vocab_size)
        print(f"Model loaded from: {pt_model_path}")
    else:
        model = YALSTMChatModel(vocab_size=vocab_size)
        model.save_model(pt_model_path)
        print("Model instance created")
    return model



def read_csv_as_plaintext(csv_file):
    with open(csv_file, "r", encoding="utf-8") as f:
        content = f.readlines().__str__()
        f.close()
    return content

def read_csv_as_plainbytes(csv_file):
    with open(csv_file, "rb") as f:
        content = f.readlines().__str__()
        f.close()
    return content

def concat_csv_files(csv_file, csv_target):
    content = read_csv_as_plaintext(csv_file)
    with open(csv_target, "a", encoding="utf-8") as f:
        f.write(content)
        f.close()


def load_ds_and_tok(type, file_path, tokenizer):
    debug_print("Loading dataset")
    debug_print("file_path: {}".format(file_path))
    ds = load_dataset(type, data_files=file_path)
    ds_str = ds.data.__str__()
    encoded = tokenizer(ds_str)
    input_str = ''
    for e in raw_data:
        input_str += e
    return ds, encoded, input_str


def prepare_for_ds(input_ids, attention_mask, label_tensor):
    ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    attn_tensor = torch.tensor(attention_mask, dtype=torch.long)
    debug_print(f"Input IDS Tensor: {ids_tensor}")
    debug_print(f"Attention Mask Tensor: {attn_tensor}")
    debug_print(f"Labels Tensor: {label_tensor}")
    tokenized_ds = TensorDataset(ids_tensor, attn_tensor, label_tensor)
    return tokenized_ds

DEBUG = False
def debug_print(message):
    if DEBUG:
        print(message)


def get_model_paths(model_postfix, base_model_path):
    chat_model_path = get_chat_model_basepath()
    pt_model_name = f"YALSTMModel_{model_postfix}.pt"
    chat_model_name = f"YALSTMModel_{model_postfix}_chat.pt"
    dictionary_name = f"YALSTMDictionary_{model_postfix}.pt"
    tok_save_model_name = f"YALSTMTokenizer_{model_postfix}.json"
    pt_model_path = os.path.join(base_model_path, pt_model_name)
    chat_model_path = os.path.join(chat_model_path, chat_model_name)
    dictionary_path = os.path.join(base_model_path, dictionary_name)
    tok_save_path = os.path.join(base_model_path, tok_save_model_name)
    lang_words_path = os.path.join(base_model_path, 'vocab_input', 'lang_words.csv')
    csv_data_ds_path = os.path.join(base_model_path, 'vocab_input_batch', '*.csv')
    debug_print(f"The paths\nThe model path: {pt_model_path}\nThe dictionary path:{dictionary_path}\nThe tokenizer path: {tok_save_path}\nThe language words path: {lang_words_path}")
    return  pt_model_path, dictionary_path, tok_save_path, lang_words_path, csv_data_ds_path, chat_model_path

if __name__ == '__main__':
    # --- Setup and Usage ---

    _, base_model_path = get_model_path()
    pt_model_path, dictionary_path, tok_save_path, lang_words_path, csv_data_ds_path, chat_model_path = get_model_paths(model_postfix="tiny", base_model_path=base_model_path)
    # Params: 10 input features, 32 hidden units, 2 stacked layers, 1 output value


    # Dummy Input: (Batch Size=8, Sequence Length=5, Features=10)

    # Usage of tokenizer
    tokenizer = get_tokenizer(tokenizer_path=tok_save_path)
    vocab_data = []
    vocabs_lists = []
    raw_data, raw_text = extract_doc_from_pdf(file_path="./test.pdf", as_doc=False)
    raw_text = tokenizer.clean_and_tokenize(raw_text)
    raw_data = tokenizer.clean_and_tokenize(raw_data)
    vocab_data.extend(raw_text)
    vocabs_lists.append(raw_data)

    raw_data, raw_text  = extract_doc_from_pdf(file_path="./test_two.pdf", as_doc=False)
    raw_text = tokenizer.clean_and_tokenize(raw_text)
    raw_data = tokenizer.clean_and_tokenize(raw_data)
    vocab_data.extend(raw_text)
    vocabs_lists.append(raw_data)

    raw_data, raw_text = extract_doc_from_pdf(file_path="./test_three.pdf", as_doc=False)
    raw_text = tokenizer.clean_and_tokenize(raw_text)
    raw_data = tokenizer.clean_and_tokenize(raw_data)
    vocab_data.extend(raw_text)
    vocabs_lists.append(raw_data)

    raw_data, raw_text = extract_doc_from_pdf(file_path="./test_four.pdf", as_doc=False)
    raw_text = tokenizer.clean_and_tokenize(raw_text)
    raw_data = tokenizer.clean_and_tokenize(raw_data)
    vocab_data.extend(raw_text)
    vocabs_lists.append(raw_data)

    raw_data, raw_text = extract_doc_from_pdf(file_path="./test_five.pdf", as_doc=False)
    raw_text = tokenizer.clean_and_tokenize(raw_text)
    raw_data = tokenizer.clean_and_tokenize(raw_data)
    vocab_data.extend(raw_text)
    vocabs_lists.append(raw_data)

    raw_data, raw_text = extract_doc_from_pdf(file_path="./test_six.pdf", as_doc=False)
    raw_text = tokenizer.clean_and_tokenize(raw_text)
    raw_data = tokenizer.clean_and_tokenize(raw_data)
    vocab_data.extend(raw_text)
    vocabs_lists.append(raw_data)

    csv_content = read_csv_as_plaintext(lang_words_path)
    csv_content = tokenizer.clean_and_tokenize(csv_content)
    vocab_data.extend(csv_content)
    vocabs_lists.append([csv_content])
    input_vocab = (vocabs_lists[0] + vocabs_lists[1] + vocabs_lists[2] + vocabs_lists[3] +
                   vocabs_lists[4] + vocabs_lists[5] + vocabs_lists[6])
    complete_text = " ".join(vocab_data).lower()

    build_vocab = input("build vocab loaded new")
    if (build_vocab == "y"):
        vocab_size = tokenizer.build_vocab(vocab_data, append=False)
    else:
       # vocab_size = tokenizer.build_vocab(vocab_data, append=True)
        vocab_size = len(tokenizer.vocab)

    model = load_create_model(base_model_path, pt_model_path, vocab_size=vocab_size)

    dataset = YATextDataset(raw_data, tokenizer)
    # ds, encoded, input_str = load_ds_and_tok('csv', lang_words_path, tokenizer)
    ds, encoded, input_str = load_ds_and_tok('csv', csv_data_ds_path, tokenizer)
    input_ids = encoded["input_ids"]  # (N, MAX_LEN)
    attention_mask = encoded["attention_mask"]
    labels_tensor = encoded["target"]
    debug_print(f"Input IDS: {input_ids}")
    debug_print(f"Attention Mask: {attention_mask}")
    debug_print(f"Labels: {labels_tensor}")
    tokenized_ds = prepare_for_ds(input_ids,
                                  attention_mask,
                                  labels_tensor)
    training_model, train_loader, val_loader, test_loader = YALLSTMModel.prepare_data_for_train(pt_model_path, tokenized_ds)
    model.train_model(training_model,train_loader, val_loader, epochs=5)

    model.save_model(base_model_path, os.path.join(pt_model_path))
    model.save_state_dict(dictionary_path)
    write_to_json = input('Write tokenizer to disk (y/n): ')
    if write_to_json == 'y':
        write_tok_to_json = True
    else:
        write_tok_to_json = False
    if write_tok_to_json:
        tokenizer.to_json(file_path=tok_save_path)
    encoded = tokenizer("hello world here I am walking like a hurricane with ice in my eyes".lower())
    decoded = tokenizer.decode(encoded["input_ids"])
    print(decoded)
    base = os.path.join(Path.home(), 'collections', 'chat_dialog')
    docs = []
    for file in os.listdir(base):
        filep = os.path.join(base, file)
        docs.append(read_csv_as_plainbytes(filep))
    #print(docs)
    chat_dataset  = YAAutoregressiveDataset(docs, tokenizer)
    # TODO: correct this all
    chat = load_create_chat_model(chat_model_path)
    chat.train_model(tokenizer, chat_dataset, chat_model_path=chat_model_path, epochs=30)
    result = YALSTMChatModel.generate(chat, tokenizer, "Who was Winston Churchill ?", max_new_tokens=40)
    print(result)
