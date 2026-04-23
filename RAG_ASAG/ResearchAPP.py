from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utilities.RAGUtils import query_execute, generate_follow_ups, load_vector_db, printout_results


def process_query(vector_db, context, query):
    answer, result, relevance, q_result = query_execute(vector_db, query)
    add_on_answer, add_on_result, add_on_relevance, add_on_q_result = generate_follow_ups(vector_db, query, context, q_result)
    print(f"Result -> {q_result} {add_on_q_result}")
    print(f"Result Readable -> ")
    for i in range(len(result)):
        print(result[i].page_content)
    print(f"Add on -> ")
    add_on_fin_result = ''
    for i in range(len(add_on_q_result)):
        add_on_fin_result = add_on_fin_result + add_on_q_result[i].page_content
        print(add_on_q_result[i].page_content)
    printout_results(add_on_answer, add_on_result, add_on_relevance, add_on_q_result)
    return add_on_fin_result


if __name__ == '__main__':
    vector_db = load_vector_db(False)
    context = input("Context: >")
    query = input("Ask me: > ")
    while query != 'exit':
        process_query(vector_db, context, query)
        query = input("Ask me more: > ")
        while  query != 'new' and query != 'exit':
            process_query(vector_db, context, query)
            query = input("Ask me more: > ")
        if query != 'exit':
            context = input("Context: >")
            query = input("Ask me: > ")