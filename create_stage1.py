import itertools
import pandas as pd
from madgen import AdversarialMathDataGen

# Set Constants
input_path = "./dataset/stage1_train_processed_data.json"
output_path = "./output/temp.csv"
start_idx=0
end_idx=2

# Load data
df = pd.read_json(input_path).T
input_files = df['passage'].to_list()

# Define Postprocess Function
def postprocess_simple(input_str: str):
    if "Augmented Passages" in input_str:
        input_str = input_str.split('Augmented Passages:')[1]
    else:
        input_str = input_str.split('Augmented Passage:')[1]
        
    if "New Variables" in input_str:
        input_str = input_str.split('New Variables:')[0]
    else:
        input_str = input_str.split('New Variable:')[0]
    
    input_str = input_str.strip("\n").strip(" ").replace('"', '')
    return input_str

# Instantiate Object
madgen_object = AdversarialMathDataGen(input_list=input_files, 
                                       output_path=output_path, 
                                       postprocess_func = postprocess_simple)
madgen_object.transform(start_idx=start_idx, end_idx=end_idx, overwrite=True)

# Add Other Columns
if output_path.endswith("csv"):
    final_df = pd.read_csv(output_path)
    question_files = df['qa_pairs'].apply(lambda x: x[0]['question'])
    answer_files = df['qa_pairs'].apply(lambda x: x[0]['answer']['number'])
    final_df['question'] = list(itertools.chain(*zip(question_files[start_idx:end_idx],
                                                     question_files[start_idx:end_idx])))
    final_df['answer'] = list(itertools.chain(*zip(answer_files[start_idx:end_idx],
                                                   answer_files[start_idx:end_idx])))
    final_df.to_csv(output_path, index=False)