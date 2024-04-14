
import random
from ChatTemplate import ChatTemplate
from evaluator import Evaluator
from modelSaver import ModelSaver
from unsloth import FastLanguageModel
from pathlib import Path


def GetOutputModelName(model_name):
    # Get the base name of the model and use it to name the fine-tuned model
    base_name_parts = model_name.split("/")
    base_name = (
        base_name_parts[-1] if len(base_name_parts) > 1 else base_name_parts[0]
    )
    base_name_parts = base_name.split("-")
    if "Summarizer" in base_name_parts:
        base_name_parts.remove("Summarizer")
    base_name_parts.insert(1, "Summarizer")
    out_model_name = "-".join(base_name_parts)
    return out_model_name

def GetRandomValidationMessage(dataset_val):
    # Get the length of the list
    length = len(dataset_val["messages"])
    # Generate a random index
    index = random.randrange(length)        # trunk-ignore(bandit/B311)
    # Access the element at the random even index
    messages_chat = dataset_val[index]["messages"]
    # Remove reference from dictionaries with role 'assistant'
    for message in messages_chat:
        if message["role"] == "assistant":
            message["content"] = ""
    messages_text = dataset_val[index]["text"]
    messages_str = "".join(messages_text)
    Reference_summary = messages_str.split("assistant", 1)[1]
    return messages_chat, Reference_summary
# Get the absolute path of the root directory of the project
root_dir = Path(__file__).resolve().parent
model_name = "unsloth/gemma-2b-it-bnb-4bit"
out_model_name = GetOutputModelName(model_name)
fine_tuned_model_dir = root_dir /"model"
local_model_path = f"{fine_tuned_model_dir}/{out_model_name}"
max_seq_length = 1024
load_in_4bit = True
wandb_run_url = 'https://wandb.ai/william-derue/LLM-summarizer_trainer/runs/nlrru9au'
wandb_run_path = 'william-derue/LLM-summarizer_trainer/nlrru9au'
wandb_run_name = 'run-unsloth/gemma-2b-it-bnb-4bit-7449'
fine_tuned_model_dir = '/home/will/LM_summarizer_trainer/model'

print(f"Loading model from {local_model_path}")


model, tokenizer = FastLanguageModel.from_pretrained(
model_name=local_model_path,
max_seq_length=max_seq_length,
dtype=None,
load_in_4bit=load_in_4bit,
)

chatml = ChatTemplate(tokenizer)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

_, dataset_val = chatml.load_data()

# Get a random validation message
message, _ = GetRandomValidationMessage(dataset_val)

# Ask the user for input
user_input = input("Do you want to test the model? (y/n): ")
if user_input.lower() == "y":
    eval = Evaluator(model,
                     tokenizer,
                     model_name,
                     local_model_path,
                     max_seq_length,
                     load_in_4bit,
                     wandb_run_name)
    eval_file_path, model, tokenizer, evaluation = eval.evaluate(message)
    eval.display(eval_file_path)

model_saver = ModelSaver(
    model,
    tokenizer,
    fine_tuned_model_dir,
    out_model_name,
    wandb_run_url,
    wandb_run_path,
    eval_file_path,
    evaluation,
)
model_saver.save_model()

