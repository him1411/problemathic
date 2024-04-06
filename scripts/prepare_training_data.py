#!/usr/bin/env python
# coding=utf-8

import os
import json
import argparse
import pandas as pd


def convert_problemathic_non_aversarial_data(data_path, output_path, difficulty_type):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = pd.read_csv(data_path, header=0)
    with open(output_path, "w") as fout:
        for idx, row in data.iterrows():
            if row["explanation"] == "CouldNotExtractError":
                continue
            fout.write(json.dumps({
                "dataset": f"problemathic_non_adv_{difficulty_type}",
                "id": f"passage_{idx}",
                "prompt": f'<<Passage>>{row["input_passage"]}<<Question>>{row["question"]}',
                "completion": f'<<Explanation>>{row["explanation"]}<<FinalAnswer>>{row["answer"]}'
                }) + "\n")
            
def convert_problemathic_aversarial_data(data_path, output_path, difficulty_type):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = pd.read_csv(data_path, header=0)
    with open(output_path, "w") as fout:
        for idx, row in data.iterrows():
            if row["explanation_adv"] == "CouldNotExtractError":
                continue
            fout.write(json.dumps({
                "dataset": f"problemathic_adv_{difficulty_type}",
                "id": f"passage_{idx}",
                "prompt": f'<<Passage>>{row["input_passage"]}<<Question>>{row["question"]}',
                "completion": 
                f'<<Reasoning>>{row["explanation_adv"]}<<RelevantVariables>>{row["relevant_variables_extracted"]}<<Explanation>>{row["solution"]}<<FinalAnswer>>{row["answer"]}'
                }) + "\n")
            

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--is_adversarial", 
        type=bool,
    )
    arg_parser.add_argument(
        "--raw_data_path", 
        type=str, 
    )
    arg_parser.add_argument(
        "--output_path", 
        type=str, 
    )
    arg_parser.add_argument(
        "--is_complex", 
        type=bool
    )

    args = arg_parser.parse_args()
    if args.is_complex:
        difficulty_type = "complex"
        print("Stage: Complex")
    else:
        difficulty_type = "simple"
        print("Stage: Simple")
        
    if args.is_adversarial:
        print("Type: Adversarial")
        convert_problemathic_aversarial_data(
            data_path=args.raw_data_path, 
            output_path=args.output_path,
            difficulty_type=difficulty_type
            )
    else:
        print("Type: Non Adversarial")
        convert_problemathic_non_aversarial_data(
            data_path=args.raw_data_path, 
            output_path=args.output_path,
            difficulty_type=difficulty_type
            )
        