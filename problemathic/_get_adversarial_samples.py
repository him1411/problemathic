import os
import json
import time
from tqdm.auto import tqdm
from typing import Union
from langchain_core.language_models.base import BaseLanguageModel

from ._chains import MultiStepAdversarialStageOneGeneration, AdversarialGeneration, GetExplanation
from .models import Model


class AdversarialMathDataGen:
    def __init__(self, 
                 model: Union[str, BaseLanguageModel],
                 use_langchain_core,
                 api_key,
                 max_retries=1,
                 wait_time=10,
                 debug=False,
                 ) -> None:
        self.use_langchain = use_langchain_core
        self.debug = debug

        # Set Model
        if model == 'openai_gpt4_chat':
            model = Model(use_langchain=self.use_langchain).get_openai_chat_model(temperature=0.3,
                                                                            api_key=api_key,
                                                                            model="gpt-4",
                                                                            max_tokens=2048)
            self.llm = model
            
        elif model == 'openai_gpt4_completion':
            model = Model(use_langchain=self.use_langchain).get_openai_model(temperature=0.7,
                                                                             api_key=api_key,
                                                                             model="gpt-4",
                                                                             max_tokens=2048)
            self.llm = model
        elif model == 'openai_gpt3.5_chat':
            model = Model(use_langchain=self.use_langchain).get_openai_chat_model(temperature=0.7,
                                                                                  api_key=api_key,
                                                                                  model="gpt-3.5-turbo",
                                                                                  max_tokens=2048)
            self.llm = model
        elif model == 'openai_gpt3.5_completion':
            model = Model(use_langchain=self.use_langchain).get_openai_model(temperature=0.7,
                                                                             api_key=api_key,
                                                                             model="gpt-3.5-turbo-instruct",
                                                                             max_tokens=2048)
            self.llm = model
        elif model == "gemini":
            if not self.use_langchain:
                print("Current backend only supports with LangChain")

            model = Model(use_langchain=self.use_langchain).get_google_gemini_chat_model(temperature=0.7,
                                                         google_api_key=api_key,
                                                         model="gemini-pro",
                                                         max_tokens=2048)
            self.llm = model
        else:
            try:
                self.llm = model
            except:
                raise Exception('Pass a valid model client or model name')

        self.max_retries = max_retries
        self.wait_time = wait_time
        self.stage_one_multi_step_prompt_chain = MultiStepAdversarialStageOneGeneration(
            from_model=self.llm, use_langchain=self.use_langchain).get_stage_one_multi_step_prompt_chain()
        self.stage_two_and_three_prompt_chain = AdversarialGeneration(
            from_model=self.llm, use_langchain=self.use_langchain).get_prompt_chain()
        self.get_explanation = GetExplanation(from_model=self.llm, use_langchain=self.use_langchain)
        self.simple_explanation_prompt_chain = self.get_explanation.get_explanation_for_simple_passage_prompt_chain()
        self.adversary_explanation_prompt_chain = self.get_explanation.get_explanation_for_adversarial_passage_prompt_chain()
    
    def transform(self,
                  docs,
                  stage_type, 
                  start_idx=0, 
                  end_idx=None,
                  output_path=None, 
                  overwrite=False, 
                  get_explanation=False, 
                  ) -> json:
        """
        TODO: Add detailed doc string
        """

        if end_idx is None:
            end_idx = len(docs)
        
        if output_path:
        
            if not output_path.endswith(".json"):
                raise Exception(f"Output file should be a .json. Provided {os.path.splitext(output_path)[1]}")
            
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            result = []

            if not overwrite and os.path.exists(output_path):
                with open(output_path) as existing_file:
                    file_contents = existing_file.read()
                result = json.loads(file_contents)
                start_idx = len(result)
        else:
            result = []
    
        for idx, sample in tqdm(enumerate(docs[start_idx:end_idx], start=start_idx), 
                                initial=start_idx, total=start_idx+len(docs[start_idx:end_idx])):            
            
            input_file_id = f"passage_{idx}"
            final_return_output = {}
            final_return_output[input_file_id] = {}
            final_return_output[input_file_id]["question"] = sample["question"]
            final_return_output[input_file_id]["answer"] = sample["answer"]

            final_return_output[input_file_id]["simple"] = {}
            final_return_output[input_file_id]["adversary"] =  {}
            final_return_output[input_file_id]["simple"]["passage"] = sample["passage"]

            if stage_type == 1:
                stage_one_step_one_prompt_chain = self.stage_one_multi_step_prompt_chain[0]

                first_step_output, first_step_attempts = self.retry(trace_header="StageOneStepOne", 
                                                                    input_file_id=input_file_id, 
                                                                    prompt_or_chain=stage_one_step_one_prompt_chain,
                                                                    input_dict={"input_passage":sample["passage"]})
                
                first_step_output["text"]["input_passage"] = sample["passage"]
                final_return_output[input_file_id]["adversary"]["new_variable"] = first_step_output["text"].get("new_variable", "CouldNotExtract")
                final_return_output[input_file_id]["adversary"]["start_value"] = first_step_output["text"].get("start_value", "CouldNotExtract")
                final_return_output[input_file_id]["adversary"]["end_value"] = first_step_output["text"].get("end_value", "CouldNotExtract")
                final_return_output[input_file_id]["adversary"]["attempt_for_stage_one_step_one"] = first_step_attempts-1

                stage_one_step_two_prompt_chain = self.stage_one_multi_step_prompt_chain[1]
                second_step_output, second_step_attempts = self.retry(trace_header="StageOneStepTwo",
                                                                      input_file_id=input_file_id,
                                                                      prompt_or_chain=stage_one_step_two_prompt_chain,
                                                                      input_dict=first_step_output["text"])
                        
                final_return_output[input_file_id]["adversary"]["augmented_passage"] = second_step_output["text"].get("augmented_passage", "CouldNotExtractError")
                final_return_output[input_file_id]["adversary"]["attempt_for_stage_one_step_two"] = second_step_attempts-1

            else:
                prompt_chain = self.stage_two_and_three_prompt_chain[0]
                output, attempts = self.retry(trace_header="StageTwoOrThree",
                                              input_file_id=input_file_id,
                                              prompt_or_chain=prompt_chain,
                                              input_dict={"input_passage":sample["passage"]})
                
                final_return_output[input_file_id]["adversary"]["new_variable"] = output["text"].get("new_variable", "CouldNotExtract")
                final_return_output[input_file_id]["adversary"]["start_value"] = output["text"].get("start_value", "CouldNotExtract")
                final_return_output[input_file_id]["adversary"]["end_value"] = output["text"].get("end_value", "CouldNotExtract")
                final_return_output[input_file_id]["adversary"]["augmented_passage"] = output["text"].get("augmented_passage", "CouldNotExtractError")
                final_return_output[input_file_id]["adversary"]["attempt_for_stage_one_step_one"] = attempts-1

            if get_explanation:
                # Get Simple Explanation
                simple_explanation_input_variables = {
                    "input_passage": sample["passage"], 
                    "question":sample["question"],
                    "answer": sample["answer"]
                    }
                simple_explanation_step_output, simple_explanation_step_attempts = self.retry(
                    trace_header="SimpleExplanation", 
                    input_file_id=input_file_id, 
                    prompt_or_chain=self.simple_explanation_prompt_chain,
                    input_dict=simple_explanation_input_variables
                    )
 
                final_return_output[input_file_id]["simple"]["explanation"] = simple_explanation_step_output["text"].get("simple_solution", "CouldNotExtractError")
                final_return_output[input_file_id]["simple"]["attempt_explanation"] = simple_explanation_step_attempts-1
                final_return_output[input_file_id]["simple"]["llm_answer"] = simple_explanation_step_output["text"].get("llm_answer", "CouldNotExtractError")

                # Get Adversary Explanation
                adversary_explanation_input_variables = {
                    "augmented_passage": final_return_output[input_file_id]["adversary"]["augmented_passage"],
                    "question": sample["question"], 
                    "simple_explanation": final_return_output[input_file_id]["simple"]["explanation"],
                    "answer": sample["answer"], 
                    "new_variable": final_return_output[input_file_id]["adversary"]["new_variable"], 
                    "start_value": final_return_output[input_file_id]["adversary"]["start_value"],
                    "end_value": final_return_output[input_file_id]["adversary"]["end_value"]
                    }
                
                adversary_explanation_step_output, adversary_explanation_step_attempts = self.retry(
                    trace_header="AdversarialExplanation", 
                    input_file_id=input_file_id, 
                    prompt_or_chain=self.adversary_explanation_prompt_chain,
                    input_dict=adversary_explanation_input_variables
                    )

                final_return_output[input_file_id]["adversary"]["relevant_variables_extracted"] = adversary_explanation_step_output["text"].get("relevant_variables", "CouldNotExtractError")
                final_return_output[input_file_id]["adversary"]["explanation"] = adversary_explanation_step_output["text"].get("adversary_explanation", "CouldNotExtractError")
                final_return_output[input_file_id]["adversary"]["solution"] = adversary_explanation_step_output["text"].get("adversary_solution", "CouldNotExtractError")
                final_return_output[input_file_id]["adversary"]["llm_answer"] = adversary_explanation_step_output["text"].get("llm_answer", "CouldNotExtractError")
                final_return_output[input_file_id]["adversary"]["attempt_explanation"] = adversary_explanation_step_attempts-1
            
            result.append(final_return_output)
            if output_path:
                with open(output_path, mode='w', encoding='utf-8') as append_json:
                    json.dump(result, append_json, indent=2)

    def retry(self, trace_header, input_file_id, prompt_or_chain, input_dict):
        retry_flag = False
        output_dict = {}
        output_dict["text"] = {}
        for attempt in range(1, self.max_retries+1):
            if attempt == 1 or retry_flag:
                try:
                    if retry_flag and trace_header == "AdversarialExplanation":
                        # Try parsing using manual logic
                        print(f"[INFO] Trying manual parse logic after first attempt failed.")
                        prompt_or_chain_custom = self.get_explanation.get_explanation_for_adversarial_passage_prompt_chain(disable_json_parser=True)
                        output_dict = prompt_or_chain_custom.invoke(input=input_dict)
                        text_to_manually_parse = output_dict["text"]
                        relevant_variables = text_to_manually_parse.split("Explanation:")[0].split("Extracted: ")[-1].strip()
                        adversary_explanation = text_to_manually_parse.split("Explanation:")[-1].split("Solution: ")[0].strip()
                        adversary_solution = text_to_manually_parse.split("Explanation:")[-1].split('Solution: ')[-1].split("LLM Answer: ")[0].strip()
                        llm_answer = text_to_manually_parse.split("Explanation:")[-1].split('Solution: ')[-1].split("LLM Answer: ")[-1].rstrip("'}").strip()
                        output_dict["text"] = {"relevant_variables": relevant_variables,
                                               "adversary_explanation": adversary_explanation,
                                               "adversary_solution": adversary_solution,
                                               "llm_answer": llm_answer
                                               }
                        retry_flag = False
                        # If the parsing happens without fail, break the loop and return the parsed json.
                        break
                        
                    if self.use_langchain:
                        output_dict = prompt_or_chain.invoke(input=input_dict)
                    else:
                        text_output_to_json = json.loads(self.non_langchain_invoke(prompt_tuple=prompt_or_chain, 
                                                                                    **input_dict))
                        output_dict["text"] = text_output_to_json
                
                    retry_flag = False
                    if self.debug:
                        print("ATTEMPT: ", attempt)
                        print(f"{trace_header}:")
                        print(output_dict)
                except Exception as e:
                    retry_flag = True
                    time.sleep(self.wait_time)
                    print(f"[ERROR] {trace_header} -> {input_file_id}, Attempt: {attempt}, retry_flag: {retry_flag}, Error: {e}")                        
            else:
                break
        return output_dict, attempt

    def non_langchain_invoke(self, prompt_tuple, **kwargs):
        inferred_schema = prompt_tuple[-1]
        prompt = prompt_tuple[0]
        instruction_prompt = f"""Ensure you return the results as a JSON object 
        with the following schema {inferred_schema}"""

        input_prompt = prompt.format(format_instructions = instruction_prompt, 
                                     **kwargs)
        
        response = self.llm.chat.completions.create(
                        model="gpt-3.5-turbo",
                        response_format={ "type": "json_object" },
                        messages=[
                            {"role": "system", "content": "Do as directed. Think and reason."},
                            {"role": "user", "content":f"{input_prompt}"}
                        ]
                    )
        return response.choices[0].message.content
