import torch.nn as nn
import torch
import json
from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class YALLMModel:
    def __init__(self, input_dim=10, hidden_dim=64, layer_dim=4, output_dim=1):
        super(YALLMModel, self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # LSTM Layer
        # batch_first=True ensures input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Fully connected layer to convert hidden state to final output
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)


    def forward(self, x = torch.randn(8, 5, 10)):
        # Initializing hidden state (h0) and cell state (c0) with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        # out: tensor of shape (batch_size, seq_length, hidden_dim)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # We only need the last time step's output for the final prediction
        # out[:, -1, :] extracts (batch_size, hidden_dim)
        out = self.fc(out[:, -1, :])
        return out


class YATokenizer:
    def __init__(self,  vocab=None, special_tokens=["<PAD>", "<UNK>"]):
        self.vocab = vocab
        self.id_to_word = {}
        self.special_tokens = special_tokens

    def build_vocab(self, corpus):
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

if __name__ == '__main__':
    # --- Setup and Usage ---

    # Params: 10 input features, 32 hidden units, 2 stacked layers, 1 output value
    model = YALLMModel(input_dim=10, hidden_dim=32, layer_dim=2, output_dim=1)

    # Dummy Input: (Batch Size=8, Sequence Length=5, Features=10)
    random_input = torch.randn(8, 5, 10)

    # Forward pass
    prediction = model(random_input)
    print(f"Input Shape: {random_input.shape}")  # [8, 5, 10]
    print(f"Output Shape: {prediction.shape}")  # [8, 1]

    # Usage of tokenizer
    raw_data = ["hello world", "pytorch is great", "lstm models are powerful"]
    tokenizer = YATokenizer()
    tokenizer.build_vocab(raw_data)

    dataset = YATextDataset(raw_data, tokenizer)
    loader = DataLoader(dataset, batch_size=2)

    # 1. Load your local data or a hub dataset
    # ds = load_dataset("csv", data_files="my_data.csv")
    ds = load_dataset("imdb", split="train[:100]")  # Example with built-in data

    # 3. Apply the mapping
    tokenized_ds = ds.map(tokenize_function, batched=True)
    tokenized_ds.set_format(type="torch", columns=["input_ids"])

    tokenizer.to_json(file_path="./YALLMModelSuite.json")
    # Usage
    tokenizer = YATokenizer.from_json(file_path="./YALLMModelSuite.json")
    print(tokenizer.encode("hello world"))