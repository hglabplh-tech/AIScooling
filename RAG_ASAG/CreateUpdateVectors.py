# %% [markdown]
# 
# %%
import kagglehub
import os
import uuid
import bs4

from pathlib import Path

from lxml.etree import DocumentInvalid
from numpy.f2py.auxfuncs import throw_error
from oauthlib.oauth2.rfc6749.endpoints import metadata
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, WebBaseLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings


from utilities.RAGUtils import build_vectors, get_db_lit_path, get_db_inf_path, get_db_history_path, actual_time
from utilities.RAGUtils import CHUNK_SIZE, add_documents, read_all_docs, get_db_base_path

def get_embedding(key: str):
    if key == 'openai':
        set_api_env_and_keys()
        return OpenAIEmbeddings()
    if key == 'ollama':
        return OllamaEmbeddings(model="llama3.1")
    elif key == 'huggingface':
        return HuggingFaceEmbeddings()
    else:
        return DeterministicFakeEmbedding(size=4096)


def get_app_key():
    fname = 'app_keyid.sec'
    with open(fname) as f:
        app_key = f.read()
        f.close()
        return app_key


def write_index_row(id, path):
    with open(get_db_inf_path() + "/id_index", 'w') as csv:
        row = id + ',' + path
        csv.writelines([row])

def read_lines(file_path):
    with open(file_path, 'r') as infile:
        lines = infile.readlines()
        return lines


def set_api_env_and_keys():
    app_key = get_app_key()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://withpersona.com/verify?inquiry-id=inq_NMrSJeR6Aiv2XciLNSjc5qcsv2vn'
    os.environ['LANGCHAIN_API_KEY'] = app_key
    os.environ['OPENAI_API_KEY'] = app_key
    return

def update_vectors(complete_content):
    chunk_size, chunk_overlap = CHUNK_SIZE()
    splitter =  RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    embeddings = get_embedding('openai')
    chunks = splitter.split_text(complete_content)

    return vector_db

if __name__ == '__main__':
    set_api_env_and_keys()

    mode = input("Select mode create / createhist / createlit / add / addhist/ addlit / update(**later**): ")
    print(f"start collecting pdf datas at {actual_time()}....")
    print(get_db_base_path())
    if mode == 'create':
        ret_code, complete_content = read_all_docs(
            ['compscience'])

        print("build vector")
        vector_db = build_vectors(complete_content, get_db_inf_path(), False)
    elif mode == 'createhist':
        ret_code, complete_content = read_all_docs(
            ['history'])

        print("build vector")
        vector_db = build_vectors(complete_content, get_db_history_path(), False)
    elif mode == 'createlit':
        ret_code, complete_content = read_all_docs(
            ['literature'])

        print("build vector")
        vector_db = build_vectors(complete_content, get_db_lit_path(), False)
    elif mode == 'add':
        ret_code, complete_content = read_all_docs(['comp-add3'])
        print(complete_content[0])
        vector_db = add_documents(complete_content, get_db_inf_path(), False)
    elif mode == 'addhist':
        ret_code, complete_content = read_all_docs(['/Users/hglabplhak/collections/history_edu/more_ger'])
        print(complete_content[0])
        vector_db = add_documents(complete_content, get_db_history_path(), False)
    elif mode == 'addlit':
        ret_code, complete_content = read_all_docs(['/Users/hglabplhak/collections/history_edu/more_ger'])
        print(complete_content[0])
        vector_db = add_documents(complete_content, get_db_lit_path(), False)
    print(f"ready at {actual_time()}")