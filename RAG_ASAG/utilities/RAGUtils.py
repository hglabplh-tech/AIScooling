import os
import time
import bs4
import datetime
from pathlib import Path
from pandas import read_csv
from pandas import DataFrame, read_csv
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, WebBaseLoader, TextLoader, \
    UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import LLMChain, SimpleSequentialChain
from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from RAG_ASAG.utilities.HuggingChat import HuggingChat
import pandas as pd

CSIZE_CONST = 1024


def extract_doc_from_web_html(url):
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(url),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    return docs


def extract_csv_data(filepath: str):
    path =  Path(filepath)
    path =  path.absolute()
    csv_data_frame = read_csv(path,  sep=',')
    return csv_data_frame


def extract_doc_from_html(file_path):
    html_loader = UnstructuredHTMLLoader(file_path)
    docs = html_loader.load()
    return docs


def extract_doc_from_text(file_path):
    text_loader = TextLoader(file_path)
    docs = text_loader.load()
    return docs


def extract_doc_from_markdown(file_path):
    md_loader = UnstructuredMarkdownLoader(file_path)
    docs = md_loader.load()
    return docs


def extract_doc_from_word(file_path):
    docx_loader = UnstructuredWordDocumentLoader(file_path)
    docs = docx_loader.load()
    return docs


def extract_doc_from_pdf(file_path, as_doc=True):
    # creating a pdf reader object

    loader = PyPDFLoader(
        file_path,
        mode="page",
        #images_parser=RapidOCRBlobParser(),
    )
    documents = loader.load()
    # printing number of pages in pdf file
    page_count = len(documents)
    print(f'Number of pages: {page_count}')
    if as_doc:
        # getting a specific page from the pdf file
        return documents
    else:
        text = []
        for page in range(page_count):
            text.append(documents[page].page_content)
        return text

def analyze_CSV(csv_file_path, query,parent):
    if parent:
        set_api_env_and_keys_in_parent()
    else:
        set_api_env_and_keys()

    client = HuggingChat()
# Load CSV data
    df = pd.read_csv(csv_file_path)
    csv_string = df.to_string()

# Send to OpenAI
    response = client.execute_query("You are a data researcher.",
            "where criteria is   {query}  in this data\n{csv_string}" )
    return response.content


def read_lines(file_path):
    with open(file_path, 'r') as infile:
        lines = infile.readlines()
        return lines


def get_suffix(f, suffix: str):
    _, ext = os.path.splitext(f)
    return (ext == '.' + suffix)


# %% [markdown]
#
# %%
def read_all_docs(data_path_start, sub_path):
    content_array = []
    data_path = os.path.join(data_path_start, sub_path)
    filenames = os.listdir(data_path)
    for filename in filenames:
        content_path = os.path.join(data_path, filename)
        print(f"Collecting: {content_path}.... ")
        if get_suffix(content_path, 'pdf'):
            documents = extract_doc_from_pdf(content_path)
            content_array = content_array + documents
        elif get_suffix(content_path, 'txt'):
            documents = extract_doc_from_text(content_path)
            content_array = content_array + documents
        elif get_suffix(content_path, 'md'):
            documents = extract_doc_from_markdown(content_path)
            content_array = content_array + documents
        elif get_suffix(content_path, 'docx'):
            documents = extract_doc_from_word(content_path)
            content_array = content_array + documents
        elif get_suffix(content_path, 'html') or get_suffix(content_path, 'htm'):
            documents = extract_doc_from_html(content_path)
            content_array = content_array + documents
        elif get_suffix(content_path, 'wbx'):
            lines = read_lines(content_path)
            for url in lines:
                documents = extract_doc_from_web_html(url)
                content_array = content_array + documents
        else:
            print(f"The {content_path} cannot pe processed.... go on with next entry")
    return 0, content_array


def CHUNK_SIZE():
    chunk_size = CSIZE_CONST
    chunk_overlap = ((CSIZE_CONST / 100) * 15)
    return chunk_size, chunk_overlap


def get_app_key():
    fname = 'app_keyid.sec'
    with open(fname) as f:
        app_key = f.read()
        f.close()
        return app_key


def get_hugg_key():
    fname = 'hugg_keyid.sec'
    with open(fname) as f:
        app_key = f.read()
        f.close()
        return app_key


def get_app_key_in_parent():
    fname = '../app_keyid.sec'
    with open(fname) as f:
        app_key = f.read()
        f.close()
        return app_key


def get_hugg_key_in_parent():
    fname = '../hugg_keyid.sec'
    with open(fname) as f:
        app_key = f.read()
        f.close()
        return app_key


def actual_time():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


def actual_classw_ts():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d[%H:%M]')


def build_vectors(complete_content, db_path, parent):
    # 2. Embed and Store in Vector DB (Chroma)
    chunk_size, chunk_overlap = CHUNK_SIZE()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    embeddings = get_embedding('huggingface', parent)
    print(f"The complete count of documents is: {len(complete_content)}")
    print(f"The first element is : {complete_content[0]}")
    chunks = splitter.split_documents(complete_content)
    print(f"Split pages post into {len(chunks)} sub-documents.")
    vector_db = SKLearnVectorStore.from_documents(chunks, embedding=embeddings,
                                                  persist_path=db_path,
                                                  serializer="parquet")
    vector_db.persist()
    return vector_db


def add_documents(complete_content, db_path, parent):
    embeddings = get_embedding('huggingface', parent)
    chunk_size, chunk_overlap = CHUNK_SIZE()
    vector_db = SKLearnVectorStore(embedding=embeddings,
                                   persist_path=db_path,
                                   serializer="parquet")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    print(f"The complete count of documents is: {len(complete_content)}")
    print(f"The first element is : {complete_content[0]}")
    chunks = splitter.split_documents(complete_content)
    print(f"Split documents into {len(chunks)} sub-documents.")
    print(f"build vector with chunk-size: {chunk_size} and chunk-overlap: {chunk_overlap}")

    vector_db.add_documents(documents=chunks, embedding=embeddings)
    vector_db.persist()
    return vector_db


def get_db_base_path():
    base = Path.home()
    db_base_path = os.path.join(base, 'sklearn_vectordb')
    if not os.path.exists(db_base_path):
        os.makedirs(db_base_path)
    return db_base_path

def  get_model_path():
    base_model_path = os.path.join(get_db_base_path(), "sklearn_model")
    if not os.path.exists(base_model_path):
        os.makedirs(base_model_path)
    absolute_model_path = os.path.join(base_model_path, "sklearn_trained_model.pt")
    return absolute_model_path, base_model_path

def get_rag_config_path():
    home = Path.home()
    conf_base_path = os.path.join(home, 'RAGConf')
    if not os.path.exists(conf_base_path):
        os.makedirs(conf_base_path)
    config_path = os.path.join(conf_base_path, "config.ini")
    return config_path, conf_base_path


def get_db_inf_path():
    db_path = os.path.join(get_db_base_path(), "info_store")
    return db_path


def get_db_temp_path():
    db_path = os.path.join(get_db_base_path(), "temp_store")
    return db_path


def get_db_lit_path():
    db_path = os.path.join(get_db_base_path(), "lit_store")
    return db_path


def get_db_history_path():
    db_path = os.path.join(get_db_base_path(), "history_store")
    return db_path


def set_api_env_and_keys():
    app_key = get_app_key()
    hugg_key = get_hugg_key()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_API_KEY'] = app_key
    os.environ['OPENAI_API_KEY'] = app_key
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['HF_TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_TOKEN'] = hugg_key
    return


def set_api_env_and_keys_in_parent():
    app_key = get_app_key_in_parent()
    hugg_key = get_hugg_key_in_parent()
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_API_KEY'] = app_key
    os.environ['OPENAI_API_KEY'] = app_key
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['HF_TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_TOKEN'] = hugg_key
    return


def get_vector_db(db_path, parent):
    embeddings = get_embedding('huggingface', parent)
    #embeddings = DeterministicFakeEmbedding(size=4096)
    #embeddings = DeterministicFakeEmbedding(size=1024)
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = SKLearnVectorStore(embedding=embeddings,
                                   persist_path=db_path,
                                   serializer="parquet")
    return vector_db


def load_vector_db(parent):
    if parent:
        get_app_key_in_parent()
        get_hugg_key_in_parent()
    else:
        get_app_key()
        get_hugg_key()
    db_to_use = input("DB to use -> compscience/history/literature/temp : ")
    db_path = ''
    if db_to_use == 'compscience':
        db_path = get_db_inf_path()
    elif db_to_use == 'history':
        db_path = get_db_history_path()
    elif db_to_use == 'literature':
        db_path = get_db_lit_path()
    else:
        db_path = get_db_temp_path()
    return get_vector_db(db_path, parent=parent)


def generate_follow_ups(vector_db, original_query, context, generated_answer):
    print(f"Start query at: {actual_time()}")
    prompt = ChatPromptTemplate.from_template(
        "Original Question: {query}\n"
        "Retrieved Knowledge: {context}\n"
        "Provided Answer: {answer}\n\n"
        "Logic Task:\n"
        "1. Identify key terms in the Knowledge that weren't fully explained in the Answer.\n"
        "2. Write a summary\n"
        "3. Propose 3 follow-up questions that lead to a deeper understanding.\n"
        #"3. Propose 3 follow-up questions\n"
        "4. Give the references \n"

    )

    raw_query = prompt.invoke({
        "query": original_query,
        "context": context,
        "answer": generated_answer
    })
    vector_db.get_all_documents()
    query = raw_query.to_string()
    result = vector_db.similarity_search(query, k=3)
    answer = vector_db.similarity_search_with_score(query, k=3)
    relevance = vector_db.similarity_search_with_relevance_scores(query, k=3)
    retriever = get_as_retriever(vector_db)
    q_result = retriever.invoke(query)
    print(f"Finish query at: {actual_time()}")
    return answer, result, relevance, q_result


def get_as_retriever(vector_db):
    return vector_db.as_retriever(search_kwargs={"k": 3})


def query_execute(vector_db, query):
    print(f"Start query at: {actual_time()}")
    result = vector_db.similarity_search(query, k=3)
    answer = vector_db.similarity_search_with_score(query, k=3)
    relevance = vector_db.similarity_search_with_relevance_scores(query, k=3)

    retriever = get_as_retriever(vector_db)
    q_result = retriever.invoke(query)

    print(f"Finish query at: {actual_time()}")

    return answer, result, relevance, q_result


def get_embedding(key: str, parent: bool):
    if parent:
        set_api_env_and_keys_in_parent()
    else:
        set_api_env_and_keys()
    if key == 'openai':
        print('OpenAI Embeddings selected')
        return OpenAIEmbeddings()
    if key == 'ollama':
        print('Ollama Embeddings selected')
        return OllamaEmbeddings(model="qwen3:latest")
    elif key == 'huggingface':
        print('Hugging Face Embeddings selected')
        return HuggingFaceEmbeddings()
    else:
        print('Fake Embeddings selected')
        return DeterministicFakeEmbedding(size=4096)


def printout_results(answer, result, relevance, q_result):
    print(f"Relevance result: {relevance}")
    print(f"Query result scored: {answer}")
    printout_retrieved_docs(q_result)
    print(f"Document content: {result[0].page_content}")


def printout_retrieved_docs(q_result):
    print(f"Retrieved Result: {q_result}")

if __name__ == "__main__":
    vector_db = load_vector_db(True)
    query = input("give query or exit: ")
    while query != "exit":
        answer, result, relevance, q_result = query_execute(vector_db, query)
        if "link" in query:
            response = analyze_CSV("youtube_index.csv", query, True)
            print(response)
            q_result = q_result.append(response)
            printout_results(answer, result, relevance, q_result)
        query = input("give query or exit: ")


def get_keywords(query,parent):
    if parent:
        set_api_env_and_keys_in_parent()
    else:
        set_api_env_and_keys()
    client = HuggingChat()

# Send to OpenAI
    response = client.execute_query("You are a keyword generator.",
            f"create up to four keywords for the answer  {query}")
    return response.content

def get_query_keywords(query,parent):
    if parent:
        set_api_env_and_keys_in_parent()
    else:
        set_api_env_and_keys()
    client = HuggingChat(dtype='cuda',
                         device_map='auto',
                         model_id= "openai/gpt-oss-20b")

# Send to OpenAI
    response = client.execute_query("You are a keyword generator.",
                                    f"create up to five short keywords with only substantives no numbering for the query  {query}")


    return response.content
