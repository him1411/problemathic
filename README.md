# Adversarial Math Problem Generation

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


## BibTeX Entry and Citation Info

If you find our work useful, please cite our work: 

```bibtex
@misc{anantheswaran2024investigatingrobustnessllmsmath,
      title={Investigating the Robustness of LLMs on Math Word Problems}, 
      author={Ujjwala Anantheswaran and Himanshu Gupta and Kevin Scaria and Shreyas Verma and Chitta Baral and Swaroop Mishra},
      year={2024},
      eprint={2406.15444},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.15444}, 
}
```
