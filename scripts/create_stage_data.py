import os
import sys
import json
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI

sys.path.append("./")
from problemathic import AdversarialMathDataGen

# Load environment variables
load_dotenv()

# CONSTANTS
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

root_path = "./"


# Set Constants
input_tuples = [
    # ("./dataset/curated/stage1_train_processed_data.json", "./dataset/generated/json/stage_one_temp.json", 1),
    # ("./dataset/curated/stage2_train_processed_data.json", "./dataset/generated/json/stage_two.json", 2),
    ("./dataset/curated/stage3_train_processed_data.json", "./dataset/generated/json/stage_three.json", 3)
    ]

start_idx=0
end_idx=-1
debug=False
overwrite=True
use_langchain_core=True
get_explanation=True
max_retries=3

# Running with Azure EndPoint
model_keys = {
    "openai_api_base": os.getenv("OPENAI_API_BASE_AI23", os.getenv("OPENAI_API_BASE_AI23")),
    "api_key": os.getenv("OPENAI_API_KEY_AI23", os.getenv("OPENAI_API_KEY_AI23"))
}
model = AzureChatOpenAI(
                        deployment_name="gpt-4",
                        openai_api_version="2023-03-15-preview",
                        max_tokens=2048,
                        temperature=0.3,
                        max_retries=max_retries,
                        request_timeout=5,
                        **model_keys
                    )

for input_tuple in input_tuples:
    
    input_path, output_path, stage_type = input_tuple

    with open(input_path) as f:
        docs = json.load(f)

    madgen = AdversarialMathDataGen(model=model,
                                    api_key=OPENAI_API_KEY, 
                                    use_langchain_core=use_langchain_core,
                                    max_retries=max_retries,
                                    debug=debug)
    madgen.transform(docs=docs,
                     stage_type=stage_type,
                     start_idx=start_idx,
                     end_idx=end_idx,
                     output_path=output_path,
                     overwrite=overwrite,
                     get_explanation=get_explanation,
                     )
                     