import os
import sys

import pandas as pd
from openai import OpenAI
from RAG_ASAG.utilities.RAGUtils import *

def analyze_CSV(csv_file_path, query,parent):
    if parent:
        api_key = get_app_key_in_parent()
    else:
        api_key = get_app_key()
    client = OpenAI(api_key=api_key)

# Load CSV data
    df = pd.read_csv(csv_file_path)
    csv_string = df.to_string()

# Send to OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data researcher."},
            {"role": "user", "content": f"where criteria is   {query}  in this data\n{csv_string}"}
        ]
        )
    return response.choices[0].message.content


if __name__ == "__main__":
    query  = input("give query or exit: ")
    while query != "exit":
        response  =  analyze_CSV("youtube_index.csv", query,True)
        print(response)
        query = input("give query or exit: ")
