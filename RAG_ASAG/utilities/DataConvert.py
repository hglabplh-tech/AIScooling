from tensorflow.python.ops.control_flow_ops import tuple_v2


def convert_to_chat_tabbed_text_file(filePath, filePath2):
    lines = []
    txt_file1 = filePath + ".txt"
    txt_file2 = filePath2 + ".txt"
    csv_file = filePath + ".out"
    with open(txt_file1, 'r') as data_file:
        qna_list = [f.split('\t') for f in data_file]

        questions = [x[0] for x in qna_list]
        answers = [x[1] for x in qna_list]
        index = 0
        for question in questions:
            answer = answers[index]
            line = "user:" + question + "bot:" + answer
            lines.append(line)
            index += 1
        data_file.close()

    with open(txt_file2, 'rb') as df:
        raw_lines = df.readlines()
        usr_txts = []
        answers = []
        for raw_line in raw_lines:
            str_line = raw_line.__str__()
            print(str_line)
            pair = str_line.split(':')
            if 'Human 1' in str_line:
                usr_txts.append(pair[1])
            if 'Human 2' in str_line:
                answers.append(pair[1])
        index = 0
        for usr_txt in usr_txts:
            line = "user:" + usr_txt + "bot:" + answer + '\n'
            lines.append(line)
            index += 1
        df.close()
    print(lines)
    with open(csv_file, 'w') as output_file:
        output_file.writelines(lines)
        output_file.close()

if __name__ == "__main__":
    convert_to_chat_tabbed_text_file("dialogs", "human_chat")