# app_tf_full.py

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import tensorflow as tf

import chromadb
from PIL import Image


# ============================================================
# OUTPUT CONFIG
# ============================================================

@dataclass
class OutputConfig:
    answer: bool = True
    asag: bool = True
    context: bool = True
    code: bool = False
    image_search: bool = False
    youtube: bool = True
    literature: bool = True


# ============================================================
# TOKENIZER WITH NEW / APPEND VOCAB
# ============================================================

class SimpleTokenizer:

    def __init__(self, vocab_path="vocab.json"):
        self.vocab_path = vocab_path
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }

        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = dict(self.special_tokens)

        self.rebuild_reverse_vocab()

    def rebuild_reverse_vocab(self):
        self.rev = {int(v): k for k, v in self.vocab.items()}

    def build_vocab(self, texts, mode="append", min_freq=1, max_vocab=None):
        if mode not in ["new", "append"]:
            raise ValueError("mode must be 'new' or 'append'")

        if mode == "new":
            self.vocab = dict(self.special_tokens)

        counter = {}

        for text in texts:
            for tok in text.lower().split():
                counter[tok] = counter.get(tok, 0) + 1

        tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        for tok, freq in tokens:
            if freq < min_freq:
                continue

            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

            if max_vocab and len(self.vocab) >= max_vocab:
                break

        self.rebuild_reverse_vocab()

    def encode(self, text, add_special=False):
        ids = [
            self.vocab.get(tok, self.vocab["<unk>"])
            for tok in text.lower().split()
        ]

        if add_special:
            ids = [self.vocab["<bos>"]] + ids + [self.vocab["<eos>"]]

        return ids

    def decode(self, ids):
        return " ".join(
            self.rev.get(int(i), "<unk>")
            for i in ids
        )

    def save(self):
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)


# ============================================================
# TENSORFLOW CHAT MODEL
# RNN / GRU / LSTM
# ============================================================

def build_chat_model(
    vocab_size,
    model_type="lstm",
    embedding_dim=256,
    hidden_dim=512,
    num_layers=2,
    dropout=0.2
):
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)

    x = tf.keras.layers.Embedding(
        vocab_size,
        embedding_dim,
        name="token_embedding"
    )(inputs)

    for i in range(num_layers):
        return_sequences = True

        if model_type == "rnn":
            x = tf.keras.layers.SimpleRNN(
                hidden_dim,
                return_sequences=return_sequences,
                dropout=dropout,
                name=f"rnn_{i}"
            )(x)

        elif model_type == "gru":
            x = tf.keras.layers.GRU(
                hidden_dim,
                return_sequences=return_sequences,
                dropout=dropout,
                name=f"gru_{i}"
            )(x)

        elif model_type == "lstm":
            x = tf.keras.layers.LSTM(
                hidden_dim,
                return_sequences=return_sequences,
                dropout=dropout,
                name=f"lstm_{i}"
            )(x)

        else:
            raise ValueError("model_type must be rnn, gru or lstm")

    outputs = tf.keras.layers.Dense(
        vocab_size,
        name="lm_head"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.config_dict = {
        "vocab_size": vocab_size,
        "model_type": model_type,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout
    }

    return model


# ============================================================
# PRETRAINED MODEL WITH GROWN VOCAB
# ============================================================

def load_pretrained_with_grown_vocab_tf(
    new_model,
    old_model_path
):
    """
    Lädt ein altes Keras-Modell in ein neues Modell.
    Embedding und Output-Layer werden teilweise kopiert,
    wenn das Vocab gewachsen ist.
    """

    old_model = tf.keras.models.load_model(old_model_path)

    old_layers = {
        layer.name: layer
        for layer in old_model.layers
    }

    for new_layer in new_model.layers:

        if new_layer.name not in old_layers:
            continue

        old_layer = old_layers[new_layer.name]

        old_weights = old_layer.get_weights()
        new_weights = new_layer.get_weights()

        if not old_weights or not new_weights:
            continue

        copied = []

        for old_w, new_w in zip(old_weights, new_weights):

            if old_w.shape == new_w.shape:
                copied.append(old_w)

            else:
                tmp = new_w.copy()

                slices = tuple(
                    slice(0, min(o, n))
                    for o, n in zip(old_w.shape, new_w.shape)
                )

                tmp[slices] = old_w[slices]
                copied.append(tmp)

        new_layer.set_weights(copied)

    return new_model


# ============================================================
# DATASET CREATION
# ============================================================

def make_language_dataset(tokenized_texts, seq_len=32, batch_size=8):
    xs = []
    ys = []

    for ids in tokenized_texts:
        if len(ids) <= seq_len:
            continue

        for i in range(len(ids) - seq_len):
            xs.append(ids[i:i + seq_len])
            ys.append(ids[i + 1:i + seq_len + 1])

    if not xs:
        return None

    x = np.array(xs, dtype=np.int32)
    y = np.array(ys, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(2048).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


# ============================================================
# TRAINING
# ============================================================

def train_model_tf(
    model,
    tokenizer,
    texts,
    seq_len=32,
    lr=5e-4,
    epochs=3,
    batch_size=8
):
    tokenized = [
        tokenizer.encode(t, add_special=True)
        for t in texts
    ]

    ds = make_language_dataset(
        tokenized,
        seq_len=seq_len,
        batch_size=batch_size
    )

    if ds is None:
        raise ValueError("Not enough text for training dataset")

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
    )

    model.fit(ds, epochs=epochs)

    return model


# ============================================================
# WEIGHTED ASAG
# ============================================================

class ASAGScorer:

    def __init__(self, weights=None):
        self.weights = weights or {
            "jaccard": 0.25,
            "semantic": 0.50,
            "context": 0.25
        }

        self.normalize_weights()

    def normalize_weights(self):
        total = sum(self.weights.values())

        if total <= 0:
            raise ValueError("ASAG weights must sum to > 0")

        for k in self.weights:
            self.weights[k] = self.weights[k] / total

    def jaccard(self, a, b):
        sa = set(a.lower().split())
        sb = set(b.lower().split())

        if not sa or not sb:
            return 0.0

        return len(sa & sb) / len(sa | sb)

    def hash_embedding(self, text, dim=384):
        vec = np.zeros(dim, dtype=np.float32)

        for tok in text.lower().split():
            idx = abs(hash(tok)) % dim
            vec[idx] += 1.0

        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def cosine(self, a, b):
        e1 = self.hash_embedding(a)
        e2 = self.hash_embedding(b)

        return float(np.dot(e1, e2))

    def score(self, question, answer, context=""):
        metrics = {
            "jaccard": self.jaccard(question, answer),
            "semantic": self.cosine(question, answer),
            "context": self.cosine(context, answer) if context else 0.0
        }

        total = 0.0

        for metric, weight in self.weights.items():
            total += weight * metrics.get(metric, 0.0)

        return {
            "total": total,
            "metrics": metrics,
            "weights": self.weights
        }


# ============================================================
# TEXT RETRIEVAL WITH CHROMADB
# TensorFlow-friendly hash embeddings
# ============================================================

class TextRetriever:

    def __init__(self, db_path="./chroma_db", dim=384):
        self.dim = dim

        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            "knowledge"
        )

    def embed(self, text):
        vec = np.zeros(self.dim, dtype=np.float32)

        for tok in text.lower().split():
            vec[abs(hash(tok)) % self.dim] += 1.0

        vec = vec / (np.linalg.norm(vec) + 1e-8)

        return vec

    def index(self, texts):
        for i, text in enumerate(texts):
            emb = self.embed(text)

            self.collection.upsert(
                ids=[str(i)],
                documents=[text],
                embeddings=[emb.tolist()]
            )

    def search(self, query, n_results=5):
        emb = self.embed(query)

        result = self.collection.query(
            query_embeddings=[emb.tolist()],
            n_results=n_results
        )

        return result.get("documents", [[]])[0]


# ============================================================
# TENSORFLOW CNN IMAGE RETRIEVAL
# ============================================================

class CNNImageRetriever:

    def __init__(self, image_dir="data/images"):
        self.image_dir = Path(image_dir)

        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )

        self.model = base

        self.index_vectors = []
        self.index_paths = []

    def load_image(self, image_path):
        img = tf.keras.utils.load_img(
            image_path,
            target_size=(224, 224)
        )

        arr = tf.keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)

        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

        return arr

    def image_embedding(self, image_path):
        arr = self.load_image(image_path)

        emb = self.model.predict(arr, verbose=0)[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        return emb

    def build_index(self):
        self.index_vectors = []
        self.index_paths = []

        if not self.image_dir.exists():
            return

        for path in self.image_dir.glob("*"):
            if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                emb = self.image_embedding(path)
                self.index_vectors.append(emb)
                self.index_paths.append(str(path))

    def search_by_image(self, query_image, top_k=5):
        if not self.index_vectors:
            self.build_index()

        q = self.image_embedding(query_image)

        results = []

        for emb, path in zip(self.index_vectors, self.index_paths):
            sim = float(np.dot(q, emb))
            results.append((sim, path))

        results.sort(reverse=True)

        return results[:top_k]


# ============================================================
# CSV YOUTUBE + LITERATURE
# ============================================================

class CSVReferenceRetriever:

    def __init__(
        self,
        youtube_csv="data/youtube_links.csv",
        literature_csv="data/literature.csv"
    ):
        self.youtube = pd.read_csv(youtube_csv)
        self.literature = pd.read_csv(literature_csv)

    def score_row(self, query, row):
        q = set(query.lower().split())

        text = " ".join(str(v).lower() for v in row.values)
        words = set(text.split())

        if not q or not words:
            return 0.0

        return len(q & words) / len(q | words)

    def query_df(self, df, query, top_k=3):
        scored = []

        for _, row in df.iterrows():
            score = self.score_row(query, row)
            scored.append((score, row.to_dict()))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            item
            for score, item in scored[:top_k]
            if score > 0
        ]

    def query_youtube(self, query, top_k=3):
        return self.query_df(self.youtube, query, top_k)

    def query_literature(self, query, top_k=3):
        return self.query_df(self.literature, query, top_k)


# ============================================================
# CODE GENERATOR
# Python / Java / PL1 CICS VSAM / REXX
# ============================================================

class CodeGenerator:

    def generate_code(self, task, language="python"):
        language = language.lower()

        if language == "python":
            return self.python_template(task)

        if language == "java":
            return self.java_template(task)

        if language in ["pl1", "pli", "pl/i"]:
            return self.pl1_cics_vsam_template(task)

        if language == "rexx":
            return self.rexx_template(task)

        raise ValueError("language must be python, java, pl1 or rexx")

    def python_template(self, task):
        return f'''# Generated Python code
# Task: {task}

def main():
    print("Task: {task}")

if __name__ == "__main__":
    main()
'''

    def java_template(self, task):
        return f'''// Generated Java code
// Task: {task}

public class GeneratedApp {{

    public static void main(String[] args) {{
        System.out.println("Task: {task}");
    }}
}}
'''

    def pl1_cics_vsam_template(self, task):
        return r'''
/********************************************************************/
/* PL/I CICS VSAM EXAMPLE                                            */
/********************************************************************/

CICSVSAM: PROC OPTIONS(MAIN);

   DCL CUSTKEY        CHAR(10);
   DCL CUSTNAME       CHAR(40);
   DCL CUSTADDR       CHAR(60);
   DCL RESP           FIXED BIN(31);
   DCL RESP2          FIXED BIN(31);

   DCL WS_RECORD      CHAR(110);

   DCL 1 CUSTOMER_RECORD,
         05 CR_KEY    CHAR(10),
         05 CR_NAME   CHAR(40),
         05 CR_ADDR   CHAR(60);

   EXEC CICS RECEIVE
        INTO(WS_RECORD)
        RESP(RESP)
        RESP2(RESP2)
   END-EXEC;

   CUSTKEY = SUBSTR(WS_RECORD, 1, 10);

   EXEC CICS READ
        FILE('CUSTVSAM')
        INTO(CUSTOMER_RECORD)
        RIDFLD(CUSTKEY)
        RESP(RESP)
        RESP2(RESP2)
   END-EXEC;

   IF RESP = DFHRESP(NORMAL) THEN DO;

      EXEC CICS SEND TEXT
           FROM('CUSTOMER FOUND')
           RESP(RESP)
      END-EXEC;

   END;
   ELSE DO;

      CR_KEY  = CUSTKEY;
      CR_NAME = 'NEW CUSTOMER';
      CR_ADDR = 'UNKNOWN ADDRESS';

      EXEC CICS WRITE
           FILE('CUSTVSAM')
           FROM(CUSTOMER_RECORD)
           RIDFLD(CR_KEY)
           RESP(RESP)
           RESP2(RESP2)
      END-EXEC;

      EXEC CICS SEND TEXT
           FROM('CUSTOMER CREATED')
           RESP(RESP)
      END-EXEC;

   END;

   EXEC CICS RETURN END-EXEC;

END CICSVSAM;
'''

    def rexx_template(self, task):
        return r'''
/* REXX INTEGRATION EXAMPLE */

ADDRESS TSO

SAY "Starting REXX integration"

CUSTKEY  = "0000000001"
CUSTNAME = "TEST CUSTOMER"
CUSTADDR = "MAIN STREET 1"

LINE.1 = LEFT(CUSTKEY,10) || LEFT(CUSTNAME,40) || LEFT(CUSTADDR,60)

"ALLOC FI(CICSIN) DA('USER.CICS.INPUT') SHR REUSE"

IF RC <> 0 THEN DO
   SAY "ALLOC FAILED RC=" RC
   EXIT RC
END

"EXECIO 1 DISKW CICSIN (STEM LINE. FINIS"

"FREE FI(CICSIN)"

SAY "Record prepared for CICS bridge, EXCI, MQ or batch gateway"

EXIT 0
'''


# ============================================================
# NAS CONFIG
# ============================================================

@dataclass
class NASConfig:
    model_type: str
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    seq_len: int


class NASChatSelector:

    def __init__(
        self,
        tokenizer,
        texts,
        population_size=4,
        generations=2
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.population_size = population_size
        self.generations = generations

        self.search_space = {
            "model_type": ["rnn", "gru", "lstm"],
            "embedding_dim": [128, 256, 384],
            "hidden_dim": [256, 512, 768],
            "num_layers": [1, 2, 3],
            "dropout": [0.1, 0.2, 0.3],
            "lr": [1e-4, 5e-4, 1e-3],
            "seq_len": [24, 32, 48]
        }

    def random_config(self):
        return NASConfig(
            model_type=random.choice(self.search_space["model_type"]),
            embedding_dim=random.choice(self.search_space["embedding_dim"]),
            hidden_dim=random.choice(self.search_space["hidden_dim"]),
            num_layers=random.choice(self.search_space["num_layers"]),
            dropout=random.choice(self.search_space["dropout"]),
            lr=random.choice(self.search_space["lr"]),
            seq_len=random.choice(self.search_space["seq_len"])
        )

    def mutate(self, cfg):
        values = asdict(cfg)
        key = random.choice(list(values.keys()))
        values[key] = random.choice(self.search_space[key])
        return NASConfig(**values)

    def crossover(self, a, b):
        values = {}

        for key in asdict(a).keys():
            values[key] = random.choice(
                [getattr(a, key), getattr(b, key)]
            )

        return NASConfig(**values)

    def evaluate(self, cfg, epochs=1):
        model = build_chat_model(
            vocab_size=len(self.tokenizer.vocab),
            model_type=cfg.model_type,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout
        )

        tokenized = [
            self.tokenizer.encode(t, add_special=True)
            for t in self.texts
        ]

        ds = make_language_dataset(
            tokenized,
            seq_len=cfg.seq_len,
            batch_size=8
        )

        if ds is None:
            return -9999.0, None

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=cfg.lr
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )
        )

        history = model.fit(
            ds.take(25),
            epochs=epochs,
            verbose=0
        )

        loss = float(history.history["loss"][-1])

        return -loss, model

    def genetic_search(self):
        population = [
            self.random_config()
            for _ in range(self.population_size)
        ]

        best_score = -9999.0
        best_cfg = None
        best_model = None

        for generation in range(self.generations):
            scored = []

            for cfg in population:
                score, model = self.evaluate(cfg)
                scored.append((score, cfg, model))

                if score > best_score:
                    best_score = score
                    best_cfg = cfg
                    best_model = model

            scored.sort(key=lambda x: x[0], reverse=True)

            elites = scored[:2]

            new_population = [
                elites[0][1],
                elites[1][1]
            ]

            while len(new_population) < self.population_size:
                a = random.choice(elites)[1]
                b = random.choice(elites)[1]

                child = self.crossover(a, b)

                if random.random() < 0.4:
                    child = self.mutate(child)

                new_population.append(child)

            population = new_population

            print(
                "NAS generation",
                generation,
                "best score",
                best_score,
                "best cfg",
                best_cfg
            )

        return best_cfg, best_model

    def reinforcement_refine(self, cfg, steps=3):
        best_cfg = cfg
        best_score, best_model = self.evaluate(best_cfg)

        for _ in range(steps):
            candidate = self.mutate(best_cfg)
            score, model = self.evaluate(candidate)

            if score > best_score:
                best_score = score
                best_cfg = candidate
                best_model = model

        return best_cfg, best_model


# ============================================================
# SMART CHAT ENGINE
# ============================================================

class SmartChatEngine:

    def __init__(
        self,
        model,
        tokenizer,
        text_retriever,
        ref_retriever,
        image_retriever=None,
        asag_weights=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.text_retriever = text_retriever
        self.ref_retriever = ref_retriever
        self.image_retriever = image_retriever

        self.asag = ASAGScorer(weights=asag_weights)
        self.code_generator = CodeGenerator()

    def generate(self, prompt, max_tokens=60):
        ids = self.tokenizer.encode(prompt, add_special=True)

        if not ids:
            ids = [self.tokenizer.vocab["<bos>"]]

        generated = list(ids)

        for _ in range(max_tokens):
            x = np.array([generated], dtype=np.int32)

            logits = self.model.predict(x, verbose=0)

            next_id = int(np.argmax(logits[0, -1]))

            generated.append(next_id)

            if next_id == self.tokenizer.vocab.get("<eos>"):
                break

        return self.tokenizer.decode(
            generated[len(ids):]
        )

    def ask(
        self,
        question,
        output_config=None,
        code_language="python",
        code_task=None,
        query_image_path=None
    ):
        if output_config is None:
            output_config = OutputConfig()

        result = {
            "question": question
        }

        docs = []

        if output_config.answer or output_config.context or output_config.asag:
            docs = self.text_retriever.search(question)

        context = "\n".join(docs)

        answer = None

        if output_config.answer:
            prompt = (
                "question: "
                + question
                + "\ncontext: "
                + context
                + "\nanswer:"
            )

            answer = self.generate(prompt)
            result["answer"] = answer

        if output_config.asag and answer is not None:
            result["asag"] = self.asag.score(
                question=question,
                answer=answer,
                context=context
            )

        if output_config.context:
            result["context"] = docs

        if output_config.code:
            result["code"] = self.code_generator.generate_code(
                task=code_task or question,
                language=code_language
            )

        if output_config.image_search:
            if self.image_retriever and query_image_path:
                result["images"] = self.image_retriever.search_by_image(
                    query_image_path,
                    top_k=5
                )
            else:
                result["images"] = []

        if output_config.youtube:
            result["youtube"] = self.ref_retriever.query_youtube(question)

        if output_config.literature:
            result["literature"] = self.ref_retriever.query_literature(question)

        return result


# ============================================================
# DATA LOADING
# ============================================================

def load_texts(data_dir="data"):
    texts = []

    for path in Path(data_dir).glob("**/*.txt"):
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    if not texts:
        texts = [
            "lstm networks are useful for sequential language modeling",
            "bert style embeddings are useful for semantic search",
            "cnn networks are useful for image retrieval",
            "asag scores evaluate generated answers with configurable weights",
            "neural architecture search optimizes chat model configuration",
            "cics vsam programs can be generated in pl1",
            "rexx can integrate batch tso and transaction gateway logic"
        ]

    return texts


def create_example_csvs():
    os.makedirs("data", exist_ok=True)

    youtube_path = "data/youtube_links.csv"
    literature_path = "data/literature.csv"

    if not os.path.exists(youtube_path):
        pd.DataFrame([
            {
                "topic": "lstm",
                "title": "LSTM explained",
                "url": "https://youtube.com/example-lstm",
                "tags": "lstm rnn nlp chatbot"
            },
            {
                "topic": "cnn",
                "title": "CNN image retrieval",
                "url": "https://youtube.com/example-cnn",
                "tags": "cnn image retrieval tensorflow"
            },
            {
                "topic": "cics",
                "title": "CICS VSAM programming",
                "url": "https://youtube.com/example-cics",
                "tags": "cics vsam pl1 rexx mainframe"
            }
        ]).to_csv(youtube_path, index=False)

    if not os.path.exists(literature_path):
        pd.DataFrame([
            {
                "topic": "lstm",
                "title": "Long Short-Term Memory",
                "author": "Hochreiter and Schmidhuber",
                "year": 1997,
                "url": "https://example.org/lstm",
                "tags": "lstm rnn sequence modeling"
            },
            {
                "topic": "cnn",
                "title": "Deep Learning",
                "author": "Goodfellow Bengio Courville",
                "year": 2016,
                "url": "https://example.org/deep-learning",
                "tags": "cnn neural networks vision"
            },
            {
                "topic": "cics",
                "title": "CICS Application Programming",
                "author": "IBM",
                "year": 2024,
                "url": "https://example.org/cics",
                "tags": "cics vsam pl1 transaction"
            }
        ]).to_csv(literature_path, index=False)


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    data_dir="data",
    vocab_path="vocab.json",
    model_dir="models_tf",
    vocab_mode="append",
    use_pretrained=True,
    asag_weights=None
):
    os.makedirs(model_dir, exist_ok=True)

    texts = load_texts(data_dir)

    tokenizer = SimpleTokenizer(vocab_path)

    tokenizer.build_vocab(
        texts,
        mode=vocab_mode,
        min_freq=1
    )

    tokenizer.save()

    text_retriever = TextRetriever("./chroma_db_tf")
    text_retriever.index(texts)

    config_path = f"{model_dir}/best_config.json"
    model_path = f"{model_dir}/best_chat_model.keras"

    if use_pretrained and os.path.exists(config_path) and os.path.exists(model_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = NASConfig(**json.load(f))

        model = build_chat_model(
            vocab_size=len(tokenizer.vocab),
            model_type=cfg.model_type,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout
        )

        model = load_pretrained_with_grown_vocab_tf(
            model,
            model_path
        )

        model = train_model_tf(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            seq_len=cfg.seq_len,
            lr=cfg.lr,
            epochs=3
        )

    else:
        nas = NASChatSelector(
            tokenizer=tokenizer,
            texts=texts,
            population_size=4,
            generations=2
        )

        cfg, model = nas.genetic_search()
        cfg, model = nas.reinforcement_refine(cfg)

        model = train_model_tf(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            seq_len=cfg.seq_len,
            lr=cfg.lr,
            epochs=3
        )

    model.save(model_path)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    with open(f"{model_dir}/asag_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": asag_weights or {
                    "jaccard": 0.25,
                    "semantic": 0.50,
                    "context": 0.25
                }
            },
            f,
            indent=2
        )

    return model, tokenizer, text_retriever, cfg


def load_engine(
    vocab_path="vocab.json",
    model_dir="models_tf",
    youtube_csv="data/youtube_links.csv",
    literature_csv="data/literature.csv"
):
    tokenizer = SimpleTokenizer(vocab_path)

    model = tf.keras.models.load_model(
        f"{model_dir}/best_chat_model.keras"
    )

    text_retriever = TextRetriever("./chroma_db_tf")

    ref_retriever = CSVReferenceRetriever(
        youtube_csv=youtube_csv,
        literature_csv=literature_csv
    )

    image_retriever = CNNImageRetriever(
        image_dir="data/images"
    )

    asag_weights = None
    asag_path = f"{model_dir}/asag_config.json"

    if os.path.exists(asag_path):
        with open(asag_path, "r", encoding="utf-8") as f:
            asag_weights = json.load(f).get("weights")

    return SmartChatEngine(
        model=model,
        tokenizer=tokenizer,
        text_retriever=text_retriever,
        ref_retriever=ref_retriever,
        image_retriever=image_retriever,
        asag_weights=asag_weights
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_example_csvs()

    asag_weights = {
        "jaccard": 0.20,
        "semantic": 0.55,
        "context": 0.25
    }

    if not os.path.exists("models_tf/best_chat_model.keras"):
        model, tokenizer, text_retriever, cfg = run_pipeline(
            data_dir="data",
            vocab_path="vocab.json",
            model_dir="models_tf",
            vocab_mode="append",
            use_pretrained=True,
            asag_weights=asag_weights
        )

        ref_retriever = CSVReferenceRetriever(
            youtube_csv="data/youtube_links.csv",
            literature_csv="data/literature.csv"
        )

        image_retriever = CNNImageRetriever("data/images")

        engine = SmartChatEngine(
            model=model,
            tokenizer=tokenizer,
            text_retriever=text_retriever,
            ref_retriever=ref_retriever,
            image_retriever=image_retriever,
            asag_weights=asag_weights
        )

    else:
        engine = load_engine()

    cfg = OutputConfig(
        answer=True,
        asag=True,
        context=True,
        code=True,
        image_search=False,
        youtube=True,
        literature=True
    )

    result = engine.ask(
        question="Explain CICS VSAM and generate PL1 code",
        output_config=cfg,
        code_language="pl1"
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))