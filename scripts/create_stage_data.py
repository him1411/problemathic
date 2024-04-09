import os
import sys
import json
from dotenv import load_dotenv

sys.path.append("./")
from problemathic import AdversarialMathDataGen

# AZURE ENDPOINT FLAG
azure_api = True # when using gpt-4 chat keys

# Load environment variables
load_dotenv()

# CONSTANTS
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# OPENAI_API_BASE = os.getenv("OPENAI_API_BASE_AI23", os.getenv("OPENAI_API_BASE_AI23"))
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_AI23", os.getenv("OPENAI_API_KEY_AI23"))
root_path = "./"
trial = True
if azure_api:
    model_keys = {
        "openai_api_base": os.getenv("OPENAI_API_BASE_AI23", os.getenv("OPENAI_API_BASE_AI23")),
        "api_key": os.getenv("OPENAI_API_KEY_AI23", os.getenv("OPENAI_API_KEY_AI23"))
    }

# Set Constants
input_tuples = [
    # ("./dataset/curated/stage1_train_processed_data.json", "./dataset/processed/json/stage_one_temp.json", 1),
    # ("./dataset/curated/stage2_train_processed_data.json", "./dataset/processed/json/stage_two.json", 2),
    ("./dataset/curated/stage3_train_processed_data.json", "./dataset/processed/json/stage_three.json", 3)
    ]

start_idx=0
end_idx=-1
debug=False
overwrite=True
use_langchain_core=True
get_explanation=True
max_retries=3

for input_tuple in input_tuples:
    
    input_path, output_path, stage_type = input_tuple

    with open(input_path) as f:
        docs = json.load(f)

    madgen = AdversarialMathDataGen(model="openai_gpt4_chat",
                                    api_key=OPENAI_API_KEY, 
                                    use_langchain_core=use_langchain_core,
                                    max_retries=max_retries,
                                    debug=debug,
                                    keys = model_keys,
                                    azure_api = azure_api)
    madgen.transform(docs=docs,
                     stage_type=stage_type,
                     start_idx=start_idx,
                     end_idx=end_idx,
                     output_path=output_path,
                     overwrite=overwrite,
                     get_explanation=get_explanation,
                     )
                     