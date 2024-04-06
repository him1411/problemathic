echo "Preparing Simple Non-Adversarial Dataset..."
python ./scripts/prepare_training_data.py \
--raw_data_path ./dataset/processed/csv/simple.csv \
--output_path ./dataset/training/simple_non_adversarial.json \

echo "Preparing Simple Adversarial Dataset..."
python ./scripts/prepare_training_data.py \
--raw_data_path ./dataset/processed/csv/simple.csv \
--output_path ./dataset/training/simple_adversarial.json \
--is_adversarial true

echo "Preparing Complex Non-Adversarial Dataset..."
python ./scripts/prepare_training_data.py \
--raw_data_path ./dataset/processed/csv/complex.csv \
--output_path ./dataset/training/complex_non_adversarial.json \
--is_complex true

echo "Preparing Complex Adversarial Dataset..."
python ./scripts/prepare_training_data.py \
--raw_data_path "./dataset/processed/csv/complex.csv" \
--output_path "./dataset/training/complex_adversarial.json" \
--is_adversarial true\
--is_complex true