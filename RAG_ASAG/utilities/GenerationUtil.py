import os
import tempfile
from  pathlib import Path
from openai import  OpenAI
from torch.testing._internal import generated
from torch.utils import cmake_prefix_path
from RAG_ASAG.utilities.RAGUtils import *

from RAG_ASAG.utilities.RAGUtils import set_api_env_and_keys_in_parent

response = {}
tmp_file = tempfile.NamedTemporaryFile()

def generate_text(prompt, model="gpt-4"):
    client = OpenAI()

    #  ResponseTextConfigParam
    response = client.responses.create(model= model,  input= prompt)

    return response

def write_to_temp(content):
    with open(tmp_file.name, "w") as f:
        f.write(content)
        f.close()
    return tmp_file.name

def  read_from_temp(file_path):
    with open(file_path, 'r') as f:
        result = f.read()
        f.close()
    return result

def get_keywords(query,parent):
    if parent:
        api_key = get_app_key_in_parent()
    else:
        api_key = get_app_key()
    client = OpenAI(api_key=api_key)

# Send to OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data generater."},
            {"role": "user", "content": f"create four keywords for the answer  {query}"}
        ]
        )
    return response.choices[0].message.content

if __name__ == '__main__':
    set_api_env_and_keys_in_parent()
    conf_section = input("Configuration / discipline:")
    conf_section = conf_section.strip()
    work_name = input("Work File Name:")
    exam_work = os.path.join(Path.home(), 'examinations', conf_section)
    exam_work_path = os.path.join(exam_work, work_name + '.exam')
    print(f"File  written will be: {exam_work_path}")
    with open(exam_work_path, 'w') as f:
        mc_prefix = ''
        question_gen = input("Please enter a text for generate a question or 'exit' to leave: ")
        if "multiple choice" in question_gen:
            mc_prefix = '::MC::'
        while question_gen != 'exit':
            if question_gen != 'exit':
                generated_question =  generate_text(question_gen).output_text
                f.write(generated_question + "##" )
                print(generated_question)
                answer_gen =input("Please enter a text for generate the answer 'exit' to leave: ")
                if answer_gen != 'exit':
                    augmented_answer = ''
                    generated_answer = generate_text(answer_gen)
                    print(f"Generated Answer: {generated_answer}")
                    manual_answer  = input("Please enter a text for manual (format ([r - replace, a - append]:: text)answer 'exit' to leave: ")
                    manual_answer = manual_answer.strip().split("::")
                    cmd = manual_answer[0].strip()
                    manual_text = manual_answer[1].strip()
                    print(cmd)
                    print(manual_text)
                    if cmd == 'a':
                        augmented_answer = mc_prefix + generated_answer.output_text + ' ' +  manual_text
                    elif cmd == 'r':
                        augmented_answer = manual_text
                    tmp_file_path =write_to_temp(augmented_answer)
                    f.write(augmented_answer + "##")
                    print(augmented_answer)
                    keywords_gen = read_from_temp(tmp_file_path)
                    dummy = input("Please enter")
                    keywords = get_keywords(keywords_gen, True)
                    f.write(keywords + "##")
                    points = input("Please enter points(integer) for this question:")
                    print(keywords)
                    f.write(points)
                    print(points)
                    question_gen = input("Please enter a text for generate a question or 'exit' to leave: ")
                    if question_gen != 'exit':
                        f.write('|')


        f.close()
        set_api_env_and_keys_in_parent()