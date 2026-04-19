from configparser import ConfigParser
import os
from pathlib import Path

from utilities.ConfigReader import ConfigReader
from utilities.RAGUtils import get_rag_config_path,actual_classw_ts
from utilities.ASAGUtils import get_asag_score, semantic_asag, get_jaccard_sim, rule_based_grading
from utilities.ASAGUtils import get_sentence_scoring, get_bert_model_scoring
from utilities.HashUtility import calculate_and_write_hash_to_file

config = None
mc_prefix = '::MC::'
def read_exam_solution(exam_file):
    questions = []
    answers = []
    keywords = []
    points_array = []
    tot_points = 0
    with open(exam_file) as f:
        content = f.read()
        splitted_cont = content.split('|')
        for item in splitted_cont:
            splitted_item = item.split("##")
            question = splitted_item.__getitem__(0)
            answer = splitted_item.__getitem__(1)
            keyword = splitted_item.__getitem__(2)
            points = splitted_item.__getitem__(3)
            questions.append(question)
            answers.append(answer)
            keywords.append(keyword)
            int_points = int(points.strip())
            tot_points = tot_points + int_points
            points_array.append(int_points)
    return questions, answers, keywords, points_array, tot_points

def do_score(question, stud_answer, ref, keywords):
    config = ConfigReader.myinstance(get_rag_config_path(), conf_section)
    asag_semantic_weight = config.read_val_float('asag_semantic_weight')
    asag_jaccard_weight = config.read_val_float('asag_jaccard_weight')
    asag_bert_weight = config.read_val_float('asag_bert_weight')
    asag_sentence_weight = config.read_val_float('asag_sentence_weight')
    rule_grad = rule_based_grading(stud_answer, keywords)
    tdiff_scoring = get_asag_score(stud_answer, ref)
    sentence_scoring = get_sentence_scoring(stud_answer, ref)
    bert_scoring = get_bert_model_scoring(question,stud_answer, ref)
    score, relevance, answer = semantic_asag(stud_answer, ref, False)
    score_jac, relevance_jac, distance_jac, rel_dist_jac = get_jaccard_sim(stud_answer, ref)
    tot_rel = (((relevance * asag_semantic_weight) +
               (sentence_scoring * asag_sentence_weight) +
               (relevance_jac * asag_jaccard_weight)) /
               (asag_jaccard_weight + asag_semantic_weight + asag_sentence_weight))
    tot_score = ((((score * asag_semantic_weight)  + (score_jac * asag_jaccard_weight))) /
                 (asag_jaccard_weight + asag_semantic_weight))
    return score, relevance, tot_score, tot_rel, score_jac, relevance_jac,distance_jac, rel_dist_jac, rule_grad, tdiff_scoring

def calculate_points_per_item(ref, tot_score, tot_rel, rule_grad, tdiff_scoring, points):
    config = ConfigReader.myinstance(get_rag_config_path(), conf_section)
    asag_base_score_limit = config.read_val_float('asag_base_score_limit')
    asag_keywords_limit = config.read_val_float('asag_keywords_limit')
    asag_total_score_limit= config.read_val_float('asag_total_score_limit')
    calc_points = 0.0
    if mc_prefix  in ref:
        if rule_grad == 1:
            calc_points = points
    else:
        if (rule_grad >= asag_keywords_limit and tdiff_scoring >= asag_base_score_limit
                and tot_score <= asag_total_score_limit):
            calc_points = float(tot_rel) * float(points)
        else:
            calc_points = 0.0
    return round(calc_points, 1)

def score_answers(questions, answers, keyword_array, points_array, tot_points,stud_ans, mode):
    tot_calc_points = 0
    for index in range(len(questions)):
        if mode == 'batch':
            stud_answer = stud_ans[index]
        else:
            stud_answer = input(f"Answer the question: {questions[index]}")
        print(f'stud_answer: {stud_answer}')
        question = questions[index]
        answer = answers[index]
        keywords = keyword_array[index]
        points = points_array[index]
        score, relevance, tot_score, tot_rel, score_jac, relevance_jac,dist_jac, rel_dist_jac, rule_grad, tdiff_scoring = do_score(question,stud_answer, answer, keywords)
        print(f"rule grad: {rule_grad}")
        print(f"ASAG score: {tdiff_scoring}")
        print(f"GPT score: {score}, relevance: {relevance}")
        print(f"Jaccard score: {score_jac}, relevance: {relevance_jac} Distance: {dist_jac} Relevance Dist {rel_dist_jac}")
        print(f"Total score: {tot_score}, relevance: {tot_rel}")
        calc_points = calculate_points_per_item(answer, tot_score, tot_rel, rule_grad, tdiff_scoring, points)
        print(f"Calculate Points Task {index} = {calc_points}")
        tot_calc_points = tot_calc_points + calc_points
    print(f"Total reached points: {round(tot_calc_points, 1)} of {tot_points}")
    percent = ((tot_calc_points  * 100) / tot_points)
    return round(percent, 0)

def set_config(conf_section):
    ConfigReader.myinstance(get_rag_config_path(), conf_section)
    return config

def read_exam_work(path, fname):
    full_path = os.path.join(path, fname)
    answer_array = []
    with open(full_path) as f:
        lines = f.read()
        answers = lines.split('#')
        for index in range(len(answers)):
            item = answers.__getitem__(index)
            answer_array.append(item)
    return answer_array


def record_answers(questions):
    for index in range(len(questions)):
        answer = input(f"Answer the question: {questions[index]}")
        work_file.write(answer)
        if answer != 'exit':
            work_file.write('#')


if __name__ == '__main__':
    record = False
    work_file =None
    my_name = input("Please enter your name:")
    discipline = input("Please enter your discipline:")
    record_q = input("Do you want to record the session as classwork (y/n): ")

    conf_section = input("Configuration / discipline:")
    exam_path = os.path.join(Path.home(), 'examinations', conf_section)
    record_hash_file_path = "hash.txt"
    record_file_path = "file.txt"
    if record_q.lower() == 'y':
        record = True
        file_name = my_name + f"_{discipline}_" + actual_classw_ts()
        hash_file_name = file_name + ".hash"
        record_base_path = os.path.join(exam_path,'works')
        record_file_path = os.path.join(record_base_path, file_name)
        record_hash_file_path = os.path.join(record_base_path, hash_file_name)
        work_file =open(record_file_path,'w')
    set_config(conf_section)
    mode = input("Mode interactive/batch:")
    if mode == 'batch':
        fname = input('Class work:')
        exam_work_path = os.path.join(exam_path, 'works')
        answer_array = read_exam_work(exam_work_path, fname)
    else:
        answer_array = []
    filename = input("Test exam for deal with :")
    filename = filename.strip() + '.exam'
    abs_path = os.path.join(exam_path, filename)
    print(f"Examination solution file. {abs_path}")
    questions, answers, keywords, points_array, tot_points = read_exam_solution(abs_path)
    if work_file  and record:
        record_answers(questions)
    else:
        percent =  score_answers(questions, answers, keywords,
                             points_array, tot_points,
                             answer_array, mode)
        print(f"Quit with {percent} %")
    if work_file and record:
        work_file.close()
        calculate_and_write_hash_to_file(record_file_path, record_hash_file_path)
