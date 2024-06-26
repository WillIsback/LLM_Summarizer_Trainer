
from huggingface_hub import ModelCard, HfApi
from dotenv import load_dotenv
import os
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
load_dotenv()
HUGGING_FACE = os.getenv('HUGGING_FACE')
WANDB_API_KEY = os.getenv('WANDB_API_KEY')
wandb.login(key=WANDB_API_KEY)
api = wandb.Api()

class ModelSaver:
    def __init__(self, model, tokenizer, fine_tuned_model_dir, out_model_name, wandb_run_url, wandb_run_path, eval_file_path=None, evaluation=''):
        self.model = model
        self.tokenizer = tokenizer
        self.fine_tuned_model_dir = fine_tuned_model_dir
        self.out_model_name = out_model_name
        self.method = ""
        self.wandb_run_url = wandb_run_url
        self.wandb_run_path = wandb_run_path
        self.eval_file_path = eval_file_path
        self.evaluation = evaluation
    def save_model(self):
        print("\nEnter the types of models you want to save. Options are: '16bit', '4bit', 'lora', 'gguf_q8_0', 'gguf_f16', 'gguf_q4_k_m'. Enter 'all' to save all types. Separate multiple options with commas.\n")
        user_input = [x.strip() for x in input().split(',')]

        if '16bit' in user_input or 'all' in user_input:
            self.method = "16bit"
            temp_model_name = self.out_model_name
            if(temp_model_name.endswith("-bnb-4bit")):
                temp_model_name = temp_model_name.replace("-bnb-4bit", "")
            print(f"\033[32m\nSaving 16bit model as \033[34m{temp_model_name}\033[32m\n\033[0m")
            self.model.save_pretrained_merged(f"{self.fine_tuned_model_dir}/{temp_model_name}", self.tokenizer, save_method = "merged_16bit",)
            self.model.push_to_hub_merged(f"Labagaite/{temp_model_name}", self.tokenizer, save_method = "merged_16bit", token =  HUGGING_FACE)
            self.UpdateModelCard(f"Labagaite/{temp_model_name}", HUGGING_FACE)

        if '4bit' in user_input or 'all' in user_input:
            self.method = "4bit"
            temp_model_name = self.out_model_name
            if(temp_model_name.endswith("-bnb-4bit")):
                temp_model_name = temp_model_name.replace("-bnb-4bit", "-bnb-4bit")
            print(f"\033[32m\nSaving 16bit model as \033[34m{temp_model_name}\033[32m\n\033[0m")
            self.model.save_pretrained_merged(f"{self.fine_tuned_model_dir}/{temp_model_name}", self.tokenizer, save_method = "merged_4bit_forced",)
            self.model.push_to_hub_merged(f"Labagaite/{temp_model_name}", self.tokenizer, save_method = "merged_4bit_forced", token =  HUGGING_FACE)
            self.UpdateModelCard(f"Labagaite/{temp_model_name}", HUGGING_FACE)

        if 'lora' in user_input or 'all' in user_input:
            self.method = "lora"
            temp_model_name = self.out_model_name
            if(temp_model_name.endswith("-bnb-4bit")):
                temp_model_name = temp_model_name.replace("-bnb-4bit", "-LORA-bnb-4bit")
            else:
                temp_model_name = temp_model_name + "-LORA"
            print(f"\033[32m\nSaving 16bit model as \033[34m{temp_model_name}\033[32m\n\033[0m")
            self.model.save_pretrained_merged(f"{self.fine_tuned_model_dir}/{temp_model_name}", self.tokenizer, save_method = "lora",)
            self.model.push_to_hub_merged(f"Labagaite/{temp_model_name}", self.tokenizer, save_method = "lora", token =  HUGGING_FACE)
            self.UpdateModelCard(f"Labagaite/{temp_model_name}", HUGGING_FACE)

        if 'gguf_q8_0' in user_input or 'all' in user_input:
            self.method = "q8_0"
            temp_model_name = self.out_model_name
            if(temp_model_name.endswith("-bnb-4bit")):
                temp_model_name = temp_model_name.replace("-bnb-4bit", "-GGUF-Q8-0")
            else:
                temp_model_name = temp_model_name + "-GGUF-Q8-0"
            print(f"\033[32m\nSaving 16bit model as \033[34m{temp_model_name}\033[32m\n\033[0m")
            self.model.save_pretrained_gguf(f"{self.fine_tuned_model_dir}/{temp_model_name}", self.tokenizer,)
            self.model.push_to_hub_gguf(f"Labagaite/{temp_model_name}", self.tokenizer, token =  HUGGING_FACE)
            self.UpdateModelCard(f"Labagaite/{temp_model_name}", HUGGING_FACE)

        if 'gguf_f16' in user_input or 'all' in user_input:
            self.method = "f16"
            temp_model_name = self.out_model_name
            if(temp_model_name.endswith("-bnb-4bit")):
                temp_model_name = temp_model_name.replace("-bnb-4bit", "-GGUF")
            else:
                temp_model_name = temp_model_name + "-GGUF"
            print(f"\033[32m\nSaving 16bit model as \033[34m{temp_model_name}\033[32m\n\033[0m")
            self.model.save_pretrained_gguf(f"{self.fine_tuned_model_dir}/{temp_model_name}", self.tokenizer, quantization_method = "f16")
            self.model.push_to_hub_gguf(f"Labagaite/{temp_model_name}", self.tokenizer, quantization_method = "f16", token =  HUGGING_FACE)
            self.UpdateModelCard(f"Labagaite/{temp_model_name}", HUGGING_FACE)

        if 'gguf_q4_k_m' in user_input or 'all' in user_input:
            self.method = "q4_k_m"
            temp_model_name = self.out_model_name
            if(temp_model_name.endswith("-bnb-4bit")):
                temp_model_name = temp_model_name.replace("-bnb-4bit", "-GGUF-q4-k-m")
            else:
                temp_model_name = temp_model_name + "-GGUF-q4-k-m"
            print(f"\033[32m\nSaving 16bit model as \033[34m{temp_model_name}\033[32m\n\033[0m")
            self.model.save_pretrained_gguf(f"{self.fine_tuned_model_dir}/{temp_model_name}", self.tokenizer, quantization_method = "q4_k_m")
            self.model.push_to_hub_gguf(f"Labagaite/{temp_model_name}", self.tokenizer, quantization_method = "q4_k_m", token =  HUGGING_FACE)
            self.UpdateModelCard(f"Labagaite/{temp_model_name}", HUGGING_FACE)

    def UpdateModelCard(self, save_directory, token):
        self.get_metrics(save_directory,token)
        content = CUSTOM_MODEL_CARD.format(
            username="Labagaite",
            base_model=self.model.config._name_or_path,
            model_type=self.model.config.model_type,
            method=self.method,
            extra="",
            wandb_run_url=self.wandb_run_url,
            eval_file_path = self.eval_file_path,
            evaluation = self.evaluation
        )
        if(self.eval_file_path is not None):
            self.push_files(self.eval_file_path,save_directory,token)

        card = ModelCard(content)
        card.push_to_hub(save_directory, token = token)

    def get_metrics(self,save_directory,token):
        run = api.run(self.wandb_run_path)
        hist = run.history(keys=['train/global_step', 'eval/loss'])
        # Plot the data
        plt.figure(figsize=(10, 10))
        sns.regplot(x='train/global_step', y='eval/loss', data=hist, color='red', lowess=True)
        plt.title('Evaluation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.savefig('eval_loss_plot.png')
        self.push_files('eval_loss_plot.png', save_directory, token)


    def push_files(self,file_path,save_directory,token):
        hfapi = HfApi()
        hfapi.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=save_directory,
        repo_type="model",
        token=token,
        )
        # Remove the file
        os.remove(file_path)

# Define new custom Model Card
CUSTOM_MODEL_CARD = """
---
base_model: {base_model}
tags:
- text-generation-inference
- transformers
- unsloth
- {model_type}
- {extra}
- summarizer
- {method}
license: apache-2.0
language:
- fr
---

# Uploaded as {method} model

- **Developed by:** {username}
- **License:** apache-2.0
- **Finetuned from model :** {base_model}

# Training Logs

## Traning metrics
![Evaluation Loss Plot](eval_loss_plot.png)

## Evaluation score
{evaluation}
[Evaluation report and scoring]({eval_file_path})

## Wandb logs
You can view the training logs [<img src="https://raw.githubusercontent.com/wandb/wandb/main/docs/README_images/logo-light.svg" width="200"/>]({wandb_run_url}).

## Training details

### training data
- Dataset : [fr-summarizer-dataset](https://huggingface.co/datasets/Labagaite/fr-summarizer-dataset)
- Data-size : 7.65 MB
- train : 1.97k rows
- validation : 440 rows
- roles : user , assistant
- Format chatml "role": "role", "content": "content", "user": "user", "assistant": "assistant"
<br>
*French audio podcast transcription*

# Project details
[<img src="https://avatars.githubusercontent.com/u/116890814?v=4" width="100"/>](https://github.com/WillIsback/Report_Maker)
Fine-tuned on French audio podcast transcription data for summarization task. As a result, the model is able to summarize French audio podcast transcription data.
The model will be used for an AI application: [Report Maker](https://github.com/WillIsback/Report_Maker) wich is a powerful tool designed to automate the process of transcribing and summarizing meetings.
It leverages state-of-the-art machine learning models to provide detailed and accurate reports.

This {model_type} model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.
This {model_type} was trained with [LLM summarizer trainer](images/Llm_Summarizer_trainer_icon-removebg.png)
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
**LLM summarizer trainer**
[<img src="https://github.com/WillIsback/LLM_Summarizer_Trainer/blob/main/images/Llm_Summarizer_trainer_icon-removebg.png?raw=true" width="150"/>](https://github.com/WillIsback/LLM_Summarizer_Trainer)
"""
