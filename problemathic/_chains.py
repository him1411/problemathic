# Load packages
import warnings
warnings.filterwarnings("ignore")

from typing import List
from langchain_core.language_models.base import BaseLanguageModel

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


# Custom Output Parser Class
class StageOneStepOneOutputParser(BaseModel):
    new_variable: str = Field(description="""The new variables introduced in the new passage based on the rules. 
                               Also print the change in the variables values.""")
    start_value: str = Field(description="The start value with units of the new variable introduced.")
    end_value: str = Field(description="The end value with units of the new variable introduced.")


class StageOneStepTwoOutputParser(BaseModel):
    augmented_passage: str = Field(description="The augmented passage with the new variables ranging from the start and end value.")


class SimpleExplanationParser(BaseModel):
    simple_solution: str = Field(description="The solution to the given question and how to arrive at the given answer.")
    llm_answer: str = Field(description="THe final answer arrived at based on the explanation provided.")


class AdversarialExplanationParser(BaseModel):
    relevant_variables: str = Field(description="The relevant variables identified in the input passage.")
    adversary_explanation: str = Field(description="The explanation for why the new variable is not required to solve the question.")
    adversary_solution: str = Field(description="The explanation for solving the question based on the given input passage to arrive at the given answer.")
    llm_answer: str = Field(description="The final answer arrived at based on the solution.")


# Stage One Generation Chain Class
class MultiStepAdversarialStageOneGeneration:
    def __init__(self, 
                 from_model: BaseLanguageModel,
                 use_langchain, 
                 debug=False
                 ) -> None:
        self.step_one_parser = JsonOutputParser(pydantic_object=StageOneStepOneOutputParser)
        self.step_two_parser = JsonOutputParser(pydantic_object=StageOneStepTwoOutputParser)
        self.llm = from_model
        self.use_langchain = use_langchain
        self.debug = debug

    def get_stage_one_multi_step_prompt_chain(self, ) -> List:
        """
        TODO: Write detailed description.
        But for now, this method returns the formatted preference 2 for a given sample.
        """
        step_one_template = """
        Propose a new variable for this problem.
        Rules:
        1. The new variable must be one of the following types: [Volume, Humidity, Temperature, Weight, Luminosity, Density, Speed, Area].
        2. The new variable must not be related to or derived from the existing variables in the passage.
        3. The new variable must not share the same physical unit as any of the original variables.
        4. The variable must have a start value and end value.
        If you follow all the rules, you will win $200. If even a single rule is broken, the world will end.
        
        Example 1:
        Passage:  My car gets 20 miles per gallon.
        Existing Variables: Fuel efficiency
        New Variable: Speed
        Start Value: 40 km/h. 
        End Value: 80 km/h
        Rule 1: New variable is one of the variables mentioned.
        Rule 2: Speed is not related to fuel efficiency. It also cannot be derived from only fuel efficiency.
        Rule 3: Speed is not measured in the same unit as fuel efficiency.
        Rule 4: Start value is 40 km/h, and end value is 80 km/h

        Passage: {input_passage}

        Additional Output Formatting Instructions:
        \n{format_instructions}\n
        """

        step_two_template = """
        You are given a passage followed by a new variable and its values. 
        Augment the passage such that the new variable is part of the passage. 
        Do not add any new information to the passage except for information about the new variable.
        Example:
        Passage:  My car gets 20 miles per gallon and was made in 1950.
        New Variable: Speed
        Start Value: 40 km/h. End Value: 80 km/h
        Augmented Passage: My car, which was made in 1950, gets 20 miles per gallon and can accelerate in speed from 40 km/h to 80 km/h.

        Passage: I have 2 pencils. I went out and bought 3 more.
        New Variable: Weight
        Start value: 220 gms. End Value: 300 gms.
        Augmented: I have 2 pencils weighing 220 gms. I went out and bought 3 more. Now my pencils weigh 300 gms.
        Passage: {input_passage}
        New Variable: {new_variable}
        Start Value: {start_value}. End Value: {end_value}

        Additional Output Formatting Instructions:
        \n{format_instructions}\n
        """

        if self.use_langchain:
            stage_one_step_one_prompt = PromptTemplate(
                input_variables=["input_passage"],
                template=step_one_template,
                partial_variables={"format_instructions": self.step_one_parser.get_format_instructions()}
                )

            stage_one_step_two_prompt = PromptTemplate(
                input_variables=["input_passage", "new_variable", "start_value", "end_value"],
                template=step_two_template,
                partial_variables={"format_instructions": self.step_two_parser.get_format_instructions()}
            )

            stage_one_step_one_chain = LLMChain(llm=self.llm, 
                                                prompt=stage_one_step_one_prompt, 
                                                output_parser=JsonOutputParser()
                                                )
            stage_one_step_two_chain = LLMChain(llm=self.llm, 
                                                prompt=stage_one_step_two_prompt, 
                                                output_parser=JsonOutputParser()
                                                )

            multi_step_prompt_chain = [stage_one_step_one_chain, stage_one_step_two_chain]
            return multi_step_prompt_chain
        else:
            stage_one_step_one_prompt = PromptTemplate(
                input_variables=["input_passage", "format_instructions"],
                template=step_one_template,
                )

            stage_one_step_two_prompt = PromptTemplate(
                input_variables=["input_passage", "new_variable", "start_value", "end_value", "format_instructions"],
                template=step_two_template,
            )

            stage_one_step_one_schema = {f"{key}":"Appropriate Response" for \
                                              key in self.step_one_parser.pydantic_object.schema()["required"]}
            stage_one_step_two_schema = {f"{key}":"Appropriate Response" for \
                                              key in self.step_two_parser.pydantic_object.schema()["required"]}
            multi_step_prompt_chain =  [(stage_one_step_one_prompt, stage_one_step_one_schema), 
                                       (stage_one_step_two_prompt, stage_one_step_two_schema)
                                       ]
            return multi_step_prompt_chain


# Stage Two and Three Generation Chain Class
class AdversarialStageTwoAndThreeGeneration:
    def __init__(self, 
                 from_model: BaseLanguageModel,
                 use_langchain, 
                 debug=False
                 ) -> None:
        self.output_parser = JsonOutputParser(pydantic_object=StageOneStepOneOutputParser)
        self.llm = from_model
        self.use_langchain = use_langchain
        self.debug = debug

    def get_stage_one_multi_step_prompt_chain(self, ) -> List:
        """
        TODO: Write detailed description.
        But for now, this method returns the formatted preference 2 for a given sample.
        """
        step_one_template = """
        Propose a new variable for this problem.
        Rules:
        1. The new variable must be one of the following types: [Volume, Humidity, Temperature, Weight, Luminosity, Density, Speed, Area].
        2. The new variable must not be related to or derived from the existing variables in the passage.
        3. The new variable must not share the same physical unit as any of the original variables.
        4. The variable must have a start value and end value.
        If you follow all the rules, you will win $200. If even a single rule is broken, the world will end.
        
        Example 1:
        Passage:  My car gets 20 miles per gallon.
        Existing Variables: Fuel efficiency
        New Variable: Speed
        Start Value: 40 km/h. 
        End Value: 80 km/h
        Rule 1: New variable is one of the variables mentioned.
        Rule 2: Speed is not related to fuel efficiency. It also cannot be derived from only fuel efficiency.
        Rule 3: Speed is not measured in the same unit as fuel efficiency.
        Rule 4: Start value is 40 km/h, and end value is 80 km/h

        Passage: {input_passage}

        Additional Output Formatting Instructions:
        \n{format_instructions}\n
        """

        step_two_template = """
        You are given a passage followed by a new variable and its values. 
        Augment the passage such that the new variable is part of the passage. 
        Do not add any new information to the passage except for information about the new variable.
        Example:
        Passage:  My car gets 20 miles per gallon and was made in 1950.
        New Variable: Speed
        Start Value: 40 km/h. End Value: 80 km/h
        Augmented Passage: My car, which was made in 1950, gets 20 miles per gallon and can accelerate in speed from 40 km/h to 80 km/h.

        Passage: I have 2 pencils. I went out and bought 3 more.
        New Variable: Weight
        Start value: 220 gms. End Value: 300 gms.
        Augmented: I have 2 pencils weighing 220 gms. I went out and bought 3 more. Now my pencils weigh 300 gms.
        Passage: {input_passage}
        New Variable: {new_variable}
        Start Value: {start_value}. End Value: {end_value}

        Additional Output Formatting Instructions:
        \n{format_instructions}\n
        """

        if self.use_langchain:
            stage_one_step_one_prompt = PromptTemplate(
                input_variables=["input_passage"],
                template=step_one_template,
                partial_variables={"format_instructions": self.step_one_parser.get_format_instructions()}
                )

            stage_one_step_two_prompt = PromptTemplate(
                input_variables=["input_passage", "new_variable", "start_value", "end_value"],
                template=step_two_template,
                partial_variables={"format_instructions": self.step_two_parser.get_format_instructions()}
            )

            stage_one_step_one_chain = LLMChain(llm=self.llm, 
                                                prompt=stage_one_step_one_prompt, 
                                                output_parser=JsonOutputParser()
                                                )
            stage_one_step_two_chain = LLMChain(llm=self.llm, 
                                                prompt=stage_one_step_two_prompt, 
                                                output_parser=JsonOutputParser()
                                                )

            multi_step_prompt_chain = [stage_one_step_one_chain, stage_one_step_two_chain]
            return multi_step_prompt_chain
        else:
            stage_one_step_one_prompt = PromptTemplate(
                input_variables=["input_passage", "format_instructions"],
                template=step_one_template,
                )

            stage_one_step_two_prompt = PromptTemplate(
                input_variables=["input_passage", "new_variable", "start_value", "end_value", "format_instructions"],
                template=step_two_template,
            )

            stage_one_step_one_schema = {f"{key}":"Appropriate Response" for \
                                              key in self.step_one_parser.pydantic_object.schema()["required"]}
            stage_one_step_two_schema = {f"{key}":"Appropriate Response" for \
                                              key in self.step_two_parser.pydantic_object.schema()["required"]}
            multi_step_prompt_chain =  [(stage_one_step_one_prompt, stage_one_step_one_schema), 
                                       (stage_one_step_two_prompt, stage_one_step_two_schema)
                                       ]
            return multi_step_prompt_chain


# Explanation Chain Class
class GetExplanation:
    def __init__(self, 
                 from_model,
                 use_langchain=False,
                 ) -> None:
        self.llm = from_model
        self.use_langchain = use_langchain
        self.simple_explanation_parser = JsonOutputParser(pydantic_object=SimpleExplanationParser)
        self.adversarial_explanation_parser = JsonOutputParser(pydantic_object=AdversarialExplanationParser)

    def get_explanation_for_simple_passage_prompt_chain(self, ) -> LLMChain:
        prompt_template = """
        You are given a mathematical problem and its answer. Show the step by step working to start from the problem and arrive at the solution.
        Example: 
        Problem: I ate 65 oranges, followed by 5 apples. 
        Question: How many fruits did I eat?
        Answer: 70
        Explanation: In order to find the total number of fruits eaten, we add the number of oranges (65) and apples (5). 65 and 5 make 70. 
        LLM Answer: 70

        Example:
        Problem: 

        Now do the same for the following problem and answer.
        Problem: {input_passage}
        Question: {question}
        Answer: {answer}

        Additional Output Formatting Instructions:
        \n{format_instructions}\n
        """
        
        if self.use_langchain:
            simple_explanation_prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["simple_passage", "question", "answer"],
                partial_variables={"format_instructions": self.simple_explanation_parser.get_format_instructions()}
            )

            simple_explanation_chain = LLMChain(
                llm=self.llm,
                prompt=simple_explanation_prompt,
                output_parser=JsonOutputParser(),
            )
            return simple_explanation_chain
        else:
            simple_explanation_prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["simple_passage", "question", "answer", "format_instructions"],
            )
            simple_explanation_schema = {f"{key}":"Appropriate Detailed Response" for \
                                              key in self.simple_explanation_parser.pydantic_object.schema()["required"]}
            return (simple_explanation_prompt, simple_explanation_schema)
        

    def get_explanation_for_adversarial_passage_prompt_chain(self, ) -> LLMChain:
        prompt_template = """
        You are given a mathematical problem and some additional metadata. Return the explanation and show the step by step 
        working to start from the problem and arrive at the solution.
        Example:
        Input: There are 2 pencils in the drawer with a weight of 10 grams each. Tim placed 3 pencils, each weighing 12 grams, in the drawer. 
        Now the pencils in the drawer have a total weight of 56 grams.', 
        Metadata:
        Question: What is the total number of pencils?
        Original Explanation: Initially there were 2 pencils. 3 more were added. Hence the total number of pencils is the sum of 2 and 3, which is 5. 
        Final Answer: 5
        Irrelevant Variables: Weight (changes from 10 to 12)
        Output:
        Relevant Variables Extracted: Number of pencils (Changes from 2 to 5)
        Explanation: As the problem deals with number of pencils, which we already have, there is no need to consider the weight of the pencils 
        when counting the total quantity.
        Solution: Initially there were 2 pencils. 3 more were added. Hence the total number of pencils is the sum of 2 and 3, which is 5.
        LLM Answer: 5

        Now do the same for the following.
        Input: {augmented_passage}
        Metadata:
        Question: {question}
        Original Explanation: {simple_explanation}
        Final Answer: {answer}
        Irrelevant Variables: {new_variable} (Changes from {start_value} to {end_value})
        Output:

        Additional Output Formatting Instructions:
        \n{format_instructions}\n
        """

        if self.use_langchain:
            adversary_explanation_prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["augmented_passage", "question", "simple_explanation", 
                                 "answer", "new_variable", "start_value", "end_value"],
            )

            adversary_explanation_chain = LLMChain(
                llm=self.llm,
                prompt=adversary_explanation_prompt,
                output_parser=JsonOutputParser()
            )
            return adversary_explanation_chain
        else:
            adversary_explanation_prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["augmented_passage", "question", "simple_explanation", 
                                 "answer", "new_variable", "start_value", "end_value",
                                 "format_instructions"],
            )
            adversarial_explanation_schema = {f"{key}":"Detailed Explanations and Fields Extracted" for \
                                              key in self.adversarial_explanation_parser.pydantic_object.schema()["required"]}
            return (adversary_explanation_prompt, adversarial_explanation_schema)
        