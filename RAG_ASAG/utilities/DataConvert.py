
def convert_to_chat_tabbed_text_file(filePath):
    lines = []
    txt_file = filePath + ".txt"
    csv_file = filePath + ".csv"
    with open(txt_file, 'r') as data_file:
        qna_list = [f.split('\t') for f in data_file]

        questions = [x[0] for x in qna_list]
        answers = [x[1] for x in qna_list]
        index = 0
        for question in questions:
            answer = answers[index]
            line = "user:" + question + ";bot:" + answer + "\n"
            lines.append(line)
            index += 1
        data_file.close()

    with open(csv_file, 'w') as output_file:
        output_file.writelines(lines)
        output_file.close()

if __name__ == "__main__":
    convert_to_chat_tabbed_text_file("data/qna.txt")