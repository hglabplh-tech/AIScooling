import os
import uuid
from typing import Any
import torch as to
import pandas as pd
from boto3.dynamodb.types import LIST
from torch import Tensor
from torch.optim._multi_tensor import AdamW
from torch.utils.data import TensorDataset, DataLoader

from RAG_ASAG.utilities.RAGUtils import load_vector_db, get_db_base_path, get_model_path
from RAG_ASAG.utilities.RAGUtils import get_app_key
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertModel, DataCollatorWithPadding,
    BertTokenizer, BertForSequenceClassification)

import csv
from sklearn.utils import Bunch
import numpy as np  # For computations


def create_BERTBase():
    # 1. Initialize tokenizer and model for regression (outputting a score)
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4) # 1 for regression
    text  = load_csv('temp_data')
    tokenizer.tokenize(text)
    tensors = DataCollatorWithPadding(tokenizer=tokenizer,  return_tensors='pt', padding=True)
    print(tensors)
    return tokenizer, model, tensors

def add_to_dataset(dataset_name, vector_db, query):
    texts = get_all_data_from_db(vector_db, query)
    with open('../' + dataset_name + '.csv','a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\\')
        csv_writer.writerow(['question', 'answer'])
        index ='question'
        for text in texts:
            if not text:
                text  =  'none'
            csv_writer.writerow([index,text])
            #index = uuid.uuid4().hex
        csv_file.close()

def load_csv(csv_name):
    texts = []
    with open('../' + csv_name + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\\')
        line_count = 0
        for row in csv_reader:
            line_count += 1
            for element in row:
                texts.append(str(line_count) + ':' + element)
    return texts

id  = uuid.uuid4().hex
def load_dataset(dataset_name):
    suffix = '.csv'

    with open(r'../' + dataset_name + suffix) as csv_file:
        data_reader = csv.reader(csv_file, delimiter='\\')
        feature_names = next(data_reader)[:-1]
        data = []
        target = []
        for row in data_reader:
            features = row[:-1]
            label = row[-1]
            data.append([num for num in features])
            target.append(label)

        data = np.array(data)
        target = np.array(target)
    return Bunch(data=data, target=target, feature_names=feature_names)


def get_all_data_from_db(vector_db,  query):
    print("Get datas")
    docs = vector_db.similarity_search(query,k=20)
    texts = [doc.page_content for doc in docs]
    print(texts)
    print("End get datas")
    return texts


def tokenize_function(tokenizer_bert, texts):
    return [tokenizer_bert(text, truncation=True, max_length=512, return_tensors='pt') for text in texts]


## TODO: work on this tomorrow 28.04.2026
def train_untrained_model(model_path):
    # Initialize the BERT classifier
    model_name = "bert-base-uncased"

    tokenizer_bert, model, _ = create_BERTBase()
    texts =load_csv('temp_data')
    model.train_batch_size  =  24
    encoded = tokenize_function(tokenizer_bert, texts)
    for encode in encoded:
        labels = to.tensor([1])
        print(encoded)
        dataset = TensorDataset(encode['input_ids'], encode['attention_mask'], labels)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        epochs = 5
        train_model(loader, model, optimizer, epochs)
    # Save your fine-tuned model
    save_model(model, model_path)


def train_model(loader: DataLoader[tuple[Tensor, ...]], model: BertForSequenceClassification, optimizer, epochs):


    # 7. Start training
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            # Move batch to GPU
            b_input_ids, b_attn_mask, b_labels = [t.to('cpu') for t in batch]

            # Clear previous gradients
            model.zero_grad()

            # Forward pass
            outputs = model(b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

        print(f"Epoch {epoch + 1} complete. Loss: {loss.item()}")
    print("end train -> save model")


def save_model(bert_model,  model_path):
    to.save(bert_model, model_path)

if  __name__ == '__main__':
    base_path = get_db_base_path()
    vector_db = load_vector_db(base_path)

    prompt  = input("Prompt: ")
    while prompt !='exit':
        add_to_dataset('temp_data',vector_db, prompt)
        prompt = input("Prompt: ")
    model_path, base_path =  get_model_path()
    train_untrained_model(model_path)