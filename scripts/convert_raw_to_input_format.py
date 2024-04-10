import json


# Set Constants
input_paths = [
    "./dataset/raw/stage1_train_processed_data.json",
    "./dataset/raw/stage2_train_processed_data.json",
    "./dataset/raw/stage3_train_processed_data.json",
    "./dataset/raw/stage4_train_processed_data.json",
    ]

for idx, input_path in enumerate(input_paths[:]):

    with open(input_path) as f:
        input_file = json.load(f)
    docs = [input_file[key] for key in input_file.keys()]

    normalized_docs = []
    for doc in docs[:]:
        unpacked_dict = {}
        for key, val in doc.items():
            if key == "passage":
                unpacked_dict[key]=val
            elif key == "qa_pairs":
                for key_key, key_val in doc[key][0].items():
                    if key_key == "question":
                        unpacked_dict[key_key]=key_val
                    if key_key == "answer":
                        unpacked_dict[key_key]=key_val["number"]
        normalized_docs.append(unpacked_dict)
    
    file_name = input_path.split("/")[-1].split('.')[0]
    print(file_name)
    with open(f"./dataset/raw_normalized/{file_name}.json", 'w', encoding='utf-8') as f:
        json.dump(normalized_docs, f, ensure_ascii=False, indent=4)
        