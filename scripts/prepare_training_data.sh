echo "Preparing Simple Non-Adversarial Dataset..."
python ./scripts/prepare_training_data.py \
--raw_data_path ./dataset/generated/csv/simple_train.csv \
--output_path ./dataset/training/simple_non_adversarial_train.json \

echo "Preparing Simple Adversarial Dataset..."
python ./scripts/prepare_training_data.py \
--raw_data_path ./dataset/generated/csv/simple_train.csv \
--output_path ./dataset/training/simple_adversarial_train.json \
--is_adversarial true

echo "Preparing Complex Non-Adversarial Dataset..."
python ./scripts/prepare_training_data.py \
--raw_data_path ./dataset/generated/csv/complex_train.csv \
--output_path ./dataset/training/complex_non_adversarial_train.json \
--is_complex true

echo "Preparing Complex Adversarial Dataset..."
python ./scripts/prepare_training_data.py \
--raw_data_path "./dataset/generated/csv/complex_train.csv" \
--output_path "./dataset/training/complex_adversarial_train.json" \
--is_adversarial true \
--is_complex true