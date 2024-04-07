Adversarial Math Problem Generation

To create adversarial data, follow the steps:

### Step 1: APIP_KEYS
In the root path, create a .env file and the following keys based on the models to be used:
```
GOOGLE_API_KEY=YOUR_KEY
OPENAI_API_KEY=YOUR_KEY
```

### Step 2: Run the generation script:
```console
python scripts/create_stage_data.py
```
