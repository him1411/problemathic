import os
import json
import pandas as pd

from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain_core.output_parsers import JsonOutputParser

from problemathic import AdversarialMathDataGen


# Load environment variables
load_dotenv()

# CONSTANTS
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
root_path = "./"
trial = True


# Set Constants
input_tuples = [
    ("./dataset/curated/stage1_train_processed_data.json", "./dataset/processed/json/stage_one.json", 1),
    ("./dataset/curated/stage2_train_processed_data.json", "./dataset/processed/json/stage_two.json", 2),
    ("./dataset/curated/stage3_train_processed_data.json", "./dataset/processed/json/stage_three.json", 3)
    ]

start_idx=0
end_idx=-1
debug=False
overwrite=False
get_explanation=True

for input_tuple in input_tuples:
    
    input_path, output_path, stage_type = input_tuple

    with open(input_path) as f:
        docs = json.load(f)

    madgen = AdversarialMathDataGen(model="openai_chat",
                                    api_key=OPENAI_API_KEY, 
                                    use_langchain=False,
                                    debug=debug)
    madgen.transform(docs=docs,
                     stage_type=stage_type,
                     start_idx=start_idx,
                     end_idx=end_idx,
                     output_path=output_path,
                     overwrite=overwrite,
                     get_explanation=get_explanation,
                     )
    
    

    # final_return_output = {}
    # # for input_file_id in tqdm(input_files[start_idx:end_idx], initial=start_idx, total=len(input_files[start_idx:end_idx])):
    # for idx_, input_file_id in enumerate(input_files):
    #     sample = input_files[input_file_id]["passage"]
    #     final_return_output[input_file_id] = {}
    #     multi_step_prompt_chain = MultiStepAdversarialStageOneGenerationChain(from_model=model).get_prompts()

    #     for idx, chain_tuple in enumerate(multi_step_prompt_chain):
    #         chain_metadata, chain = chain_tuple
    #         if idx == 0:
    #             # First chain in multi step chain
    #             return_output = chain.invoke(input=sample)
    #             final_return_output[input_file_id]["input_passage"] =  return_output["input_passage"]
    #             final_return_output[input_file_id]["adversary"] = return_output["text"]
    #             return_output["text"]["input_passage"] = return_output["input_passage"]
    #         else:
    #             return_output = chain.invoke(input=return_output["text"], return_only_outputs=True)

    #     try:
    #         final_return_output[input_file_id]["augmented_passage"] = return_output["text"]["augmented_passage"]
    #     except:
    #         final_return_output[input_file_id]["augmented_passage"] = "CouldNotExtractError"
    #     return_output = None
        
    #     if idx_ == end_idx:
    #         break

    # with open('./output/temp_try.json', 'w', encoding='utf-8') as f:
    #     json.dump(final_return_output, f, ensure_ascii=False, indent=4)


    # print(pt.format(
    #         input_passage = sample
    #         )
    #     )
    

    # # Instantiate Object
    # madgen_object = AdversarialMathDataGen(input_files=input_files,
    #                                        prompt_config_file="./config.ini",
    #                                        output_path=output_paths[idx],
    #                                        model_type=model_type
    #                                        )
    
    # madgen_object.transform(start_idx=start_idx,
    #                         end_idx=end_idx, 
    #                         overwrite=overwrite, 
    #                         get_exp=get_explanation, 
    #                         debug=debug, 
    #                         only_inference=only_inference,
    #                         run_few_shot=is_few_shot)

    # # Add Other Columns
    # if output_paths[idx].endswith("csv"):
    #     # if not get_explanation:
    #     #     final_df = pd.read_csv(output_paths[idx])
    #     #     question_files = df['qa_pairs'].apply(lambda x: x[0]['question'])
    #     #     answer_files = df['qa_pairs'].apply(lambda x: x[0]['answer']['number'])
    #     #     final_df['question'] = list(itertools.chain(*zip(question_files[start_idx:end_idx],
    #     #                                                     question_files[start_idx:end_idx])))
    #     #     final_df['answer'] = list(itertools.chain(*zip(answer_files[start_idx:end_idx],
    #     #                                                 answer_files[start_idx:end_idx])))
    #     # else:
    #     final_df = pd.read_csv(output_paths[idx])
    #     question_files = df['question'].to_list()[::2]
    #     answer_files = df['answer'].to_list()[::2]
    #     final_df['question'] = question_files[start_idx:end_idx]
    #     final_df['answer'] = answer_files[start_idx:end_idx]
    #     final_df.to_csv(output_paths[idx], index=False)