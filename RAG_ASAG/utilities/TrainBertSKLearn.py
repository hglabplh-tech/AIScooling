import os
import uuid
import pandas as pd
from RAG_ASAG.utilities.RAGUtils import load_vector_db, get_db_base_path, get_model_path
from RAG_ASAG.utilities.RAGUtils import get_app_key
from bert_sklearn import BertClassifier, load_model
from transformers import BertTokenizer, BertForSequenceClassification,  BertModel
import csv
from sklearn.utils import Bunch
import numpy as np  # For computations
def create_BERTBase():
    # 1. Initialize tokenizer and model for regression (outputting a score)
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, num_labels=4) # 1 for regression
    return tokenizer, model

def create_dataset(dataset_name, vector_db, query):
    texts, labels = get_all_data_from_db(vector_db, query)
    with open('../' + dataset_name + '.csv','w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\\')
        csv_writer.writerow(['question', 'answer'])
        index ='question'
        for text in texts:
            if not text:
                text  =  'none'
            csv_writer.writerow([index,text])
            #index = uuid.uuid4().hex
        csv_file.close()


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
   # docs = [(key, value) for key, value in docs if not key.startswith('__')]
    #docs =  vector_db._texts
    #print(vector_db._ids)
   # print(vector_db._texts)
   # print(docs[0])
    #print(docs[0].metadata)
    texts = [doc.page_content for doc in docs]

  #  tok = BertTokenizer.from_pretrained('bert-base-uncased')
   # result =  tok.tokenize(texts,pair=None, add_special_tokens=False)

    print("End get datas")
    return texts

def train_untrained_model(vector_db, query,model_path):
    # Initialize the BERT classifier
    model_name = "bert-base-uncased"
    tokenizer_bert, model = create_BERTBase()
    textData = pd.read_csv("../temp_data.csv", delimiter='\\')
    texts = get_all_data_from_db(vector_db, query)
    texts = f'{texts}::{query}'
    print(texts)
    model.train_batch_size  =  24
    inputs = tokenizer_bert(texts,num_labels=4, truncation=True, padding='max_length', return_tensors="pt")
    print(inputs)
    # 'page', 'page_label',  'source', 'producer', 'total_pages', 'subject']
    print("end train -> save model")

    # Save your fine-tuned model
   # model.save(model_path)


if  __name__ == '__main__':
    base_path = get_db_base_path()
    model_path= os.path.join(base_path, get_model_path())
    vector_db = load_vector_db(True)
    prompt  = input("Prompt: ")
    while prompt !='exit':
       # create_dataset(dataset_name='temp_data', vector_db=vector_db, query=prompt)
        train_untrained_model(vector_db, prompt, model_path)
        prompt = input("Prompt: ")