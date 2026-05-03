import os
import pathlib
import random
from random import shuffle
from typing import Any

import fitz  # PyMuPDF
from pypdf._codecs import pdfdoc
from rapidfuzz import process, fuzz


from utilities.RAGUtils import get_query_keywords


def get_pdf_files(path):
    pdffiles = []
    for dirpath, subdirs, files in os.walk(path):
        for x in files:
            if x.endswith(".pdf"):
                pdffiles.append(os.path.join(dirpath, x))
    return pdffiles


def fuzzy_search_pdf(pdf_path, query,  num_key_words=4 ,threshold=80):
    doc = fitz.open(pdf_path)
    base_name = os.path.basename(pdf_path)
    base_wo_ext = base_name.split(".")[0]
    results = []
    for page_num, page in enumerate(doc):
        # 1. Extract words as a list of tuples: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        words_on_page = page.get_text("words")

        # 2. Extract just the strings for comparison
        word_strings = [w[4] for w in words_on_page]
        # 3. Use RapidFuzz to find similar strings
        matches_w = process.extract(query, word_strings, scorer=fuzz.WRatio, score_cutoff=threshold)
        results_w = walk_result(matches_w, page_num, words_on_page, base_wo_ext, num_key_words=num_key_words)
        if len(results_w) > 0:
            results.append(results_w)
    doc.close()
    return results


def walk_result(matches, page_num,  words_on_page,base_name,num_key_words=4):
    results = []
    for match_str, score, index in matches:
        # Match word back to its original coordinate data
        rect = words_on_page[index][:num_key_words]
        score_scaled = float(score / 100)
        sentence = build_sentence(words_on_page)
        results.append({
            "pdfdoc": base_name,
            "page": page_num + 1,
            "word": match_str,
            "score": score_scaled,
            "rect": rect,
            "content": sentence
        })
    return results


def build_sentence(words_on_page) -> str:
    sentence = ''
    char_blank = ' '
    index = 0
    for act_word in words_on_page:
        if not index:
            sentence += act_word[4]
        else:
            sentence += (char_blank + act_word[4])
        index += 1
    return sentence


def get_collections_path(rubric):
    return os.path.join(pathlib.Path.home(), "collections", rubric)


def search_all_pdfs(dir_path, query, num_key_words=4, threshold=80):
    files = get_pdf_files(dir_path)
    overall_result_w = []
    for file in files:
        results_w = fuzzy_search_pdf(file, query, num_key_words=num_key_words,threshold=threshold)
        if len(results_w) > 0:
            for result in results_w:
                overall_result_w.append(result)
    return overall_result_w

def flatten(overall_result):
    return [item for result in overall_result for item in result]

def sort_by_score(overall_result_w):
    res_array = flatten(overall_result_w)
    return sorted(res_array, key=lambda x: x['score'], reverse=True)

def print_results(overall_result, num_hits=15):
    hit_count = min(num_hits, len(overall_result))
    for index in range(hit_count):
        value  = overall_result[index]
        if len(value) > 0:
            print(value)

def get_results_cooked(raw_result):
    cooked_overall_result_w = []
    for result in raw_result:
       cooked_overall_result_w.append(
           [{"pdfdoc": result["pdfdoc"]},
             {"page": result["page"],
              "word": result["word"],
              "score": result["score"],
              "rect": result["rect"]},
            {'content': result["content"]}])

    return cooked_overall_result_w

def extract_keywords(query, strict=True):
    keywords = get_query_keywords(query, True)
    print(keywords)
    keyword_array = keywords.split(". ")
    cleaned_keywords = split_by_chars(keyword_array, ['\n', '\"', ' '], filter_dec=True)
    index = 0
    keys = []
    print(len(cleaned_keywords))
    print(cleaned_keywords)
    for keyword in cleaned_keywords:
        keyword_split = keyword.split(" ")
        for  i in range(len(keyword_split)):
            k = keyword_split[i].lower()
            if k and (k in query or not strict) and not k in keys:
                keys.append(k)
    my_shuffle(keys)
    key_string = " ".join(keys)
    print(key_string)
    return key_string.strip(),len(keys)


def split_by_chars(keyword_array: list[str], chars_list: list[str], filter_dec=False):
    keyword_recur = keyword_array
    keyword_pre = []
    for character in chars_list:
        for pre in keyword_recur:
            res = pre.split(character)
            for word in res:
                keyword_pre.append(word.strip())
        keyword_recur = keyword_pre
        keyword_pre = []
        print(keyword_recur)

    if filter_dec:
        cleaned_keywords = []
        for keyword in keyword_recur:
            keyword = keyword.lower().strip()
            if not keyword.isdecimal():
                cleaned_keywords.append(keyword)
        return cleaned_keywords
    else:
        return keyword_recur


def my_shuffle(keys):
    random.shuffle(keys)
    return keys


def get_key_strict(query):
    strict = False
    if query  != 'exit':
        strict_keys = input("Enter strict keys(y/n): ")
        if  strict_keys == "y":
            strict = True
        else:
            strict = False
    else:
        strict = False
    print(f"Strict query : {strict}")
    return strict


def get_threshold():
    threshold = input("Enter threshold: ")
    if not threshold or not threshold.isdecimal():
        threshold = 80
    print(f"Threshold : {threshold}")
    return threshold


if __name__ == '__main__':
    rubric =  input("Enter rubric: ")
    query = input("Enter query: ")
    threshold  = get_threshold()
    strict_keys = get_key_strict(query)
    while query != "exit":
        key_string, num_keys = extract_keywords(query, strict=strict_keys)
        dummy = input("Press enter to continue...")
        overall_result_w = search_all_pdfs(get_collections_path(rubric), key_string, num_key_words=num_keys, threshold=int(threshold))
        overall_sorted_w = get_results_cooked(sort_by_score(overall_result_w))
        print_results(overall_sorted_w, num_hits=25)
        query = input("Enter query: ")
        strict_keys = get_key_strict(query)