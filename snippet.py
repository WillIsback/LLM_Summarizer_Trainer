from pathlib import Path
from unsloth import FastLanguageModel
from modelSaver import ModelSaver
from ChatTemplate import ChatTemplate
import random
import torch
from tests import check_token_threshold_and_truncate, test_dataset, test_text_generation
from evaluate import load as load_metric
from rouge_score import rouge_scorer
"""
INFO:root:Base Model name: mistral-Summarizer-7b-instruct-v0.2-bnb-4bit
INFO:root:Output Model name: mistral-Summarizer-Summarizer-7b-instruct-v0.2-bnb-4bit
INFO:root:Max sequence length: 1024
INFO:root:Load in 4-bit: True
INFO:root:Fine-tuned model directory: /home/will/model
INFO:root:Weights & Biases run URL: https://wandb.ai/william-derue/LLM-summarizer_trainer/runs/s9xqw6o8
INFO:root:Weights & Biases run path: william-derue/LLM-summarizer_trainer/s9xqw6o8
"""

model_name = "mistral-Summarizer-Summarizer-7b-instruct-v0.2-bnb-4bit"
output_model_name = "mistral-Summarizer-7b-instruct-v0.2-bnb-4bit"
max_seq_length = 1024
load_in_4bit = True
Fine_tuned_model_directory = Path("/home/will/model")
wandb_run_url = "https://wandb.ai/william-derue/LLM-summarizer_trainer/runs/s9xqw6o8"
wandb_run_path = "william-derue/LLM-summarizer_trainer/s9xqw6o8"
# unsloth/gemma-2b-it-bnb-4bit
# unsloth/llama-2-7b-chat-bnb-4bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2b-it-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=load_in_4bit,
)

chatml = ChatTemplate(tokenizer)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
# Load your data
dataset_train, dataset_val = chatml.load_data()

# Validate the training and validation datasets
test_dataset(dataset_train)
test_dataset(dataset_val)

# Get the length of the list
length = len(dataset_val["messages"])
# Generate a random index
index = random.randrange(0, length)
if True : index = 6 # Force index to have to truncate

print(f"\n\nIndex: {index}\n\n")

rouge = load_metric("rouge", trust_remote_code=True)

# Access the element at the random even index
messages_chat = dataset_val[0]["messages"]
test_text_generation(tokenizer, model, messages_chat, max_seq_length)

