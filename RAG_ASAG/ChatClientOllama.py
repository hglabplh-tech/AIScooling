import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model

from langchain_classic.chains import LLMChain, SimpleSequentialChain
from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain, example_generator

from utilities.RAGUtils import load_vector_db, printout_results, query_execute, get_app_key

def set_api_env_and_keys(mode):
    app_key = get_app_key()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_API_KEY'] = app_key
    os.environ['OPENAI_API_KEY'] = app_key
    return





def print_answer(result):
    print(result)

def inputPrompt(prompt, index):
    query = input(f"{prompt}({index}) : ")
    return  query

if __name__ == '__main__':
    index = 0
    vector_db  = load_vector_db()

    query = inputPrompt('Prompt', index)
    while  query != 'exit':
        answer, result, relevance,q_result = query_execute(vector_db, query)
        if query != 'exit':
            printout_results(answer, result, relevance, q_result)
            index = index + 1
            query = inputPrompt('Next Prompt', index)