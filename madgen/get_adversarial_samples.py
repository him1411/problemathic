import os
import json
import pandas as pd
from tqdm.auto import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class AdversarialMathDataGen:
    def __init__(self, 
                 input_list,
                 output_path=None,
                 postprocess_func=None
                 ) -> None:
        self.model = genai.GenerativeModel("gemini-pro") 
        self.input_list = input_list
        self.output_path = output_path
        self.return_data = []
        self.postprocess_func = postprocess_func

    def get_response(self, input_text):
        response = self.model.generate_content(input_text)
        return response.text
    
    def prompt_builder(self, prompt, prefix=None):

        if prefix is not None:
            ret_prompt = prefix
        else:
            ret_prompt = '''Augment the passage as shown by adding one new variable. 
                    Rules:
                    1. The new variable must change values over the course of the augmented passage.
                    2. Old and new value of the new variable should be explicitly mentioned.
                    3. The augmented passage should add numerical information regarding variables not 4. measured or derived from variables in the passage. 
                    4. The new variable must be one of the following types: [Volume, Humidity, Temperature, Weight, Luminosity, Density, Speed, Area]. 
                    5. The new variable type must not be present in the passage before augmentation.
                    6. The new variable must not share the same physical unit as any of the original variables. 
                    7. The new variable must not be related to or derived from the existing variables in the passage.
                    8. The augmented text should not add any numerical information about existing variables that is not already mentioned explicitly in the original passage.
                    9. Add no more than one sentence.

                    Example 1:
                    Passage:  My car gets 20 miles per gallon.
                    Existing Variables: Fuel efficiency
                    Augmented Passage: My car, which gets 20 miles for each gallon, when I drive at 120 miles per hour. I start going at 10 feet a minute. 
                    New Variables: Speed (changes from 120 to 10)

                    Explanation: the new variable speed is added. It is independent of the Fuel efficiency variable in the original passage. Speed cannot be derived from the original passage. The units for speed are not related to the units for fuel efficiency. The augmented passage does not add any information about existing variables (fuel efficiency) at all, fulfilling Rule 8. 

                    Example 2:
                    Passage:  There are 64 pigs in the barn. Some more came in, now there are 86 pigs.
                    Existing Variables: Number of pigs (changes from 64 to 86)
                    Augmented Passage: In the barn, where the temperature is a cozy 72 degrees Fahrenheit, there are 64 pigs. Some more came in. The temperature goes up to 83 degrees fahrenheit for 86 pigs. 
                    New Variables: Temperature (changes from 72 to 83)

                    Explanation: the new variable temperature is added that is independent of the number of pigs variable. The units for temperature are not related to the units for the number of pigs.  The augmented passage does not add any information about existing variables (number of pigs) at all, fulfilling Rule 8. 

                    Now output the existing variables and augment the passage below following all rules. Also print the new variables in the augmented passage.
                    Passage: {}
                    Existing Variables:
                    '''.format(prompt)

        return ret_prompt
    
    def transform(self, prefix=None, start_idx=0, end_idx=-1, overwrite=False):
        
        if overwrite and os.path.exists(self.output_path):
            os.remove(self.output_path)

        for idx, file in tqdm(enumerate(self.input_list[start_idx:end_idx])):

            madgen_packet_1 = {
                    'id':idx+1,
                    'passage':file,
                    'isAdversarial':False
                }
            self.return_data.append(madgen_packet_1)

            input_text = self.prompt_builder(prompt=file, prefix=prefix)

            if self.postprocess_func is not None:
                try:
                    response = self.postprocess_func(self.get_response(input_text=input_text))
                except:
                    response = self.get_response(input_text=input_text)
            else:
                response = self.get_response(input_text=input_text)

            madgen_packet_2 = {
                    'id':idx+1,
                    'passage':response,
                    'isAdversarial':False
                }
            self.return_data.append(madgen_packet_2)

            if self.output_path is None:
                return response
            elif self.output_path.endswith("csv"):
                if os.path.exists(self.output_path):
                    load_output_df = pd.read_csv(self.output_path)
                    load_output_df = pd.concat([load_output_df, pd.DataFrame([madgen_packet_1])], 
                                               ignore_index=True)
                    load_output_df = pd.concat([load_output_df, pd.DataFrame([madgen_packet_2])], 
                                               ignore_index=True)
                    load_output_df.to_csv(self.output_path, index=False)
                else:
                    output_df = pd.DataFrame(madgen_packet_1, index=[0])
                    output_df = pd.concat([output_df, pd.DataFrame([madgen_packet_2])], 
                                          ignore_index=True)
                    output_df.to_csv(self.output_path, index=False)
            elif self.output_path.endswith("json"):
                pass
            else:
                raise Exception("Filetype to be saved not supported. CSV/JSON currently supported.")