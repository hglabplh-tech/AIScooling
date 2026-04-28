import os
import uuid
import pandas as pd
from RAG_ASAG.utilities.RAGUtils import load_vector_db, get_db_base_path, get_model_path
from RAG_ASAG.utilities.RAGUtils import get_app_key
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertModel, DataCollatorWithPadding,
    BertTokenizer)

import csv
from sklearn.utils import Bunch
import numpy as np  # For computations


def create_BERTBase():
    # 1. Initialize tokenizer and model for regression (outputting a score)
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, num_labels=4) # 1 for regression
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
    with open('../' + csv_name + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\\')
        line_count = 0
        text  = ""
        for row in csv_reader:
            line_count += 1
            for element in row:
                text =  text + ' ' + str(line_count)  +  ' ' + element
    return text

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
    return tokenizer_bert.tokenize(texts, padding="max_length", truncation=True, max_length=128
    )


## TODO: work on this tomorrow 28.04.2026
def train_untrained_model(model_path):
    # Initialize the BERT classifier
    model_name = "bert-base-uncased"

    tokenizer_bert, model, _ = create_BERTBase()
    texts =load_csv('temp_data')
    model.train_batch_size  =  24
    tokenized_datasets = tokenize_function(tokenizer_bert, texts)
    mo
    # 5. Define training parameters
    training_args = TrainingArguments(
        output_dir=model_path,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # 6. Initialize the Trainer
    trainer = Trainer(
        eval_dataset=tokenized_datasets,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets
    )

    # 7. Start training
    trainer.train()
    print("end train -> save model")
    # Save your fine-tuned model
    model.save(model_path)


if  __name__ == '__main__':
    base_path = get_db_base_path()
    vector_db = load_vector_db(base_path)

    prompt  = input("Prompt: ")
    while prompt !='exit':
        add_to_dataset('temp_data',vector_db, prompt)
        prompt = input("Prompt: ")
    _, base_path =  get_model_path()
    train_untrained_model(base_path)