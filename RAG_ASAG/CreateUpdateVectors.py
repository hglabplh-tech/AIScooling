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
from utilities.RAGUtils import CHUNK_SIZE, add_documents, read_all_docs, get_db_base_path, get_embedding


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
    embeddings = get_embedding('huggingface', parent=False)
    chunks = splitter.split_text(complete_content)

    return vector_db

def processInputAndImport(data_rel_path, db_path):
    first = True
    data_path = os.path.join(Path.home(), 'collections', data_rel_path)
    filenames = os.listdir(data_path)

    for filename in filenames:
        ret_code, complete_content = read_all_docs(data_path, filename)
        if first:
            print("build vector-db")
            build_vectors(complete_content, db_path, False)
            first = False
        else:
            print("add to vector-db")
            add_documents(complete_content, db_path, False)
    return vector_db

if __name__ == '__main__':
    set_api_env_and_keys()

    mode = input("Select mode create / createhist / createlit / add / addhist/ addlit / update(**later**): ")
    print(f"start collecting pdf datas at {actual_time()}....")
    print(get_db_base_path())
    if mode == 'create':
        vector_db = processInputAndImport('compscience', get_db_inf_path())
    elif mode == 'createhist':
        vector_db = processInputAndImport('history', get_db_history_path())
    elif mode == 'createlit':
        vector_db = processInputAndImport('literature', get_db_lit_path())
    elif mode == 'add':
        print("not yet implemented renewed later")
    elif mode == 'addhist':
        print("not yet implemented renewed later")
    elif mode == 'addlit':
        print("not yet implemented renewed later")
    print(f"ready at {actual_time()}")