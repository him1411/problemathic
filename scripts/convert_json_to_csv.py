import json
import pandas as pd

is_simple = False

if is_simple:
    # Stage One
    with open("./dataset/generated/json/stage_one.json") as existing_file:
        file_contents = existing_file.read()
    result = json.loads(file_contents)

    # Stage Two
    with open("./dataset/generated/json/stage_two.json") as existing_file:
        file_contents = existing_file.read()
    result = result + json.loads(file_contents)
else:
    # Stage Three
    with open("./dataset/generated/json/stage_three.json") as existing_file:
        file_contents = existing_file.read()
    result = json.loads(file_contents)

dff = pd.DataFrame([key[list(key.keys())[0]] for key in result])

dff['simple'].apply(lambda x: x.keys())[0]
dff['adversary'].apply(lambda x: x.keys())[0]

dff['input_passage'] = dff['simple'].apply(lambda x: x["passage"])
dff['explanation'] = dff['simple'].apply(lambda x: x["explanation"])
dff['attempt_explanation'] = dff['simple'].apply(lambda x: x["attempt_explanation"])

dff['new_variable'] = dff['adversary'].apply(lambda x: x["new_variable"])
dff['start_value'] = dff['adversary'].apply(lambda x: x["start_value"])
dff['end_value'] = dff['adversary'].apply(lambda x: x["end_value"])
dff['attempt_for_stage_one_step_one'] = dff['adversary'].apply(lambda x: x["attempt_for_stage_one_step_one"])
dff['augmented_passage'] = dff['adversary'].apply(lambda x: x["augmented_passage"])
dff['attempt_for_stage_one_step_two'] = dff['adversary'].apply(lambda x: x.get("attempt_for_stage_one_step_two", 0))
dff['relevant_variables_extracted'] = dff['adversary'].apply(lambda x: x["relevant_variables_extracted"])
dff['explanation_adv'] = dff['adversary'].apply(lambda x: x["explanation"])
dff['solution'] = dff['adversary'].apply(lambda x: x["solution"])
dff['attempt_explanation'] = dff['adversary'].apply(lambda x: x["attempt_explanation"])

dff = dff.drop(columns=["adversary", "simple"], axis=1)

if is_simple:
    dff.to_csv("./dataset/generated/csv/simple_train.csv", index=False)
else:
    dff.to_csv("./dataset/generated/csv/complex_train.csv", index=False)