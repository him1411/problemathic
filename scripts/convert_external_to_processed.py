import json
import pandas as pd

df = pd.read_csv("./dataset/curated/gsm_8k.csv")

# GSM-8k
def extract_input_and_question(sentence):
    input_passage = "NoInputFound"
    question = "NoQuestionFound"

    question_index = sentence.rfind('?')
    if question_index != -1:
        period_index = sentence.rfind('.', 0, question_index)
        if period_index != -1:
            input_passage = sentence[:period_index + 1].strip()
            question = sentence[period_index + 1: question_index + 1].strip()
    
    return input_passage, question

final_json_list = []
for record in df.to_dict(orient="records"):
    new_record_dict = {}
    input_passage, question = extract_input_and_question(record["question"])
    answer = record["answer"].split("#### ")[-1]
    new_record_dict["passage"] = input_passage
    new_record_dict["question"] = question
    new_record_dict["answer"] = answer
    final_json_list.append(new_record_dict)

with open("./dataset/curated/gsm_8k_test_processed_data.json", "w") as json_file:
    json.dump(final_json_list, json_file, indent=4)