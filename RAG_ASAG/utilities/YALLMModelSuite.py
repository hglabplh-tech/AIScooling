import os
from typing import Mapping, Any

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import AutoTokenizer
import json
from collections import Counter
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm, trange
from transformers.generation.continuous_batching.utils import build_attention_mask
from RAG_ASAG.utilities.RAGUtils import get_model_path, extract_csv_data

from RAG_ASAG.utilities.RAGUtils import extract_doc_from_pdf

class YALLMModel(nn.Module):
    def __init__(self, input_dim=10,
                 embed_dim = 10,
                 hidden_dim=64,
                 vocab_size=7000,
                 layer_dim=4,
                 output_dim=1,
                 softmax=False):
        super(YALLMModel, self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        # LSTM Layer
        # batch_first=True ensures input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Fully connected layer to convert hidden state to final output
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = softmax
        self.log_softmax = nn.LogSoftmax(dim=2)


    def forward(self, x):
        x =  torch.randn(self.layer_dim, self.embed_dim, self.input_dim)
        if self.softmax:
            out, _ = self.lstm(x)
            # Take the last time step
            logits = self.fc(out[:, -1, :])

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

            #out: tensor of shape (batch_size, seq_length, hidden_dim)
            out, (hn, cn) = self.lstm(x, (h0, c0))

            # We only need the last time step's output for the final prediction
            # out[:, -1, :] extracts (self.batch_size, self.hidden_dim)
            out = self.fc(out[:,-1,  :])
            return out

    def train_model(self, loader: DataLoader[tuple[Tensor, ...]], epochs=5):
        criterion = nn.CrossEntropyLoss()  # Waits for Logits!
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        losses = []
        # Training Loop (simplified)
        self.train()
        step = 0
        for epoch in range(epochs):
            for batch in loader: ##TODO: really use data next step !!!! ERROR
                # Move batch to GPU
                input_tensor = torch.randn(self.layer_dim, self.embed_dim, self.input_dim, device='cpu', requires_grad=False)
                input_tensor.to('cpu')
                # Clear previous gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(input_tensor)

                logits = self(input_tensor)
                loss = criterion(logits, outputs)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                print("Epoch {}, Step {}, Loss: {}".format(epoch, step, loss.item()))
                step += 1

    def save_state_dict(self, pkl_path):
        torch.save(self.state_dict(), pkl_path)

    def save_model(self, model_path):
        torch.save(self, model_path)

    @classmethod
    def load_model(cls, model_path):
        return torch.load(model_path, map_location="cpu", weights_only=False)

class YATokenizer:
    def __init__(self,  vocab=None, special_tokens=["<PAD>", "<UNK>"]):
        self.vocab = vocab
        self.id_to_word = {}
        self.special_tokens = special_tokens

    def build_vocab(self, corpus, is_text=False):
        # Flatten all text and count word frequencies
        words = " ".join(corpus).split()
        counts = Counter(words)

        # Add special tokens first, then unique words
        unique_words = self.special_tokens + sorted(list(counts.keys()))
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

class YAMultDataset:
    def __init__(self, corpus, tokenizer_path):
        self.corpus = corpus
        self.tokenizer = YATokenizer.from_json(tokenizer_path)
        self.encoded = tokenizer.encode(self.corpus, return_tensors="pt")
        self.tokenized = torch.tensor(self.encoded, dtype=torch.long)
    def tensor_dataset(self):
        return TensorDataset(self.tokenized)

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
        self.model = YALLMModel(
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

if __name__ == '__main__':
    # --- Setup and Usage ---
    _, model_base_path = get_model_path()
    pt_model_path = os.path.join(model_base_path, 'YALLModel.pt')
    # Params: 10 input features, 32 hidden units, 2 stacked layers, 1 output value
    if (os.path.exists(pt_model_path)):
        model = YALLMModel.load_model(model_path=pt_model_path)
        print(f"Model loaded from: {pt_model_path}")
    else:
        model = YALLMModel(input_dim=10, hidden_dim=64, layer_dim=3, output_dim=1)
        print("Model instance created")

    # Dummy Input: (Batch Size=8, Sequence Length=5, Features=10)
    random_input = torch.randn(8, 5, 10)

    # Forward pass
    prediction = model(random_input)
    print(f"Input Shape: {random_input.shape}")  # [8, 5, 10]
    print(f"Output Shape: {prediction.shape}")  # [8, 1]

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
    tokenizer = YATokenizer()
    tokenizer.build_vocab(raw_data)
    dataset = YATextDataset(raw_data, tokenizer)
    ds = load_dataset("csv", data_files="./lang_words.csv")
    ds_str = ds.data.__str__()
    tokenized_ds = YAMultDataset(ds_str, "./YALLMModelSuite.json", )
    t_ds = tokenized_ds.tensor_dataset()
    loader = DataLoader(t_ds, batch_size=3)

    model.train_model(loader)


    model.save_model(os.path.join(pt_model_path))
    model.save_state_dict(os.path.join(model_base_path, 'YALLModel.pkl'))
    # 1. Load your local data or a hub dataset




    tokenizer.to_json(file_path="./YALLMModelSuite.json")
    # Usage
    tokenizer = YATokenizer.from_json(file_path="./YALLMModelSuite.json")
    encoded = tokenizer.encode("hello world here I am walking like a hurricane with ice in my eyes")
    decoded = tokenizer.decode(encoded)
    tokenized_ds = ds.map(tokenize_function, batched=True)
    tokenized_ds.set_format(type="torch", columns=["input_ids"])
    print(decoded)
