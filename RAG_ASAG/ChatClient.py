import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model

from langchain_classic.chains import LLMChain, SimpleSequentialChain
from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain, example_generator

from utilities.RAGUtils import load_vector_db, printout_results, query_execute, get_app_key, get_model_path
from  RAG_ASAG.TrainBertModel import load_model_and_train,  save_model, BERT, BertPositionalEmbedding, MultiHeadedAttention
from  RAG_ASAG.TrainBertModel import TransformerEncoderLayer,MLMHead, MLMDataset, AdamW,Dataset, DataLoader, Pooler, BertWordPieceTokenizer
from tokenizers import BertWordPieceTokenizer


def set_api_env_and_keys(mode):
    app_key = get_app_key()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_API_KEY'] = app_key
    os.environ['OPENAI_API_KEY'] = app_key
    return



def train_the_model(question,  answer):
    model_path, _ =  get_model_path()
    load_model_and_train(model_path, question, answer)

    print(f"Model saved to '{model_path}'")


def print_answer(result):
    print(result)

def inputPrompt(prompt, index):
    query = input(f"{prompt}({index}) : ")
    return  query


def trainig_question():
    train_str = input("Do / Continue training for Bert Model  y/n: ")
    if train_str.lower() == 'y':
        train = True
    else:
        train = False
    return train


if __name__ == '__main__':
    index = 0
    train = trainig_question()
    vector_db  = load_vector_db(False)
    query = inputPrompt('Prompt', index)
    while  query != 'exit':
        answer, result, relevance, q_result = query_execute(vector_db, query)
        print(f"====================== {result[0].page_content} ======================================")
        print(f"=======================================================================")
        if query != 'exit':
            if train:
                query_result = result[0].page_content
                train_the_model(query, query_result)
            printout_results(answer, result, relevance, q_result)
            index = index + 1
            train = trainig_question()
            query = inputPrompt('Next Prompt', index)