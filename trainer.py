# trainer.py is part of the project "LLM summarizer trainer" and is used to train Large Langage Model to the summarizing task using QLora as fine tuning method.
import gc
import glob
import locale
import logging
import os
import random
from pathlib import Path
import torch
from dotenv import load_dotenv
from evaluate import load as load_metric
from huggingface_hub import login
from transformers import EarlyStoppingCallback, TrainingArguments
from transformers.integrations import WandbCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel
import wandb
from ChatTemplate import ChatTemplate
from model_selector import get_model_list, select_model
from modelSaver import ModelSaver
from SampleGenerationCallback import SampleGenerationCallback
from evaluator import Evaluator
from tests import check_token_threshold_and_truncate, test_dataset, test_text_generation


class Unsloth_LLM_Trainer:
    def __init__(
        self, model_name, load_in_4bit=True, max_seq_length=512, dry_run=False
    ):
        gc.collect()
        torch.cuda.empty_cache()
        locale.getpreferredencoding = lambda: "UTF-8"
        load_dotenv()
        # Get the Hugging Face API key from the environment variables
        self.HUGGING_FACE = os.getenv("HUGGING_FACE")
        # Get the Weights & Biases API key from the environment variables
        self.WANDB_API_KEY = os.getenv("WANDB_API_KEY")
        # Log in to Weights & Biases
        wandb.login(key=self.WANDB_API_KEY)
        # Log in to Hugging Face
        login(self.HUGGING_FACE)
        # Get the absolute path of the root directory of the project
        self.root_dir = Path(__file__).resolve().parent
        # metrics
        self.rouge = load_metric("rouge", trust_remote_code=True)
        self.max_seq_length = max_seq_length
        # select the model to fine-tune
        self.model_name = model_name
        self.out_model_name = self.GetOutputModelName()
        self.fine_tuned_model_dir = self.root_dir /"model"
        self.load_in_4bit = load_in_4bit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the model and tokenizer
        print(f"\n\nLoading model and tokenizer of : {model_name}\n\n")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        self.tokenizer.paddding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        chatml = ChatTemplate(self.tokenizer, self.GetChatTemplate_Type())
        # Load your data
        self.dataset_train, self.dataset_val = chatml.load_data()
        test_dataset(self.dataset_train)
        test_dataset(self.dataset_val)
        self.dry_run = dry_run
        self.run_name = None
        self.wandb_run_url = None
        self.wandb_run_path = None
        test_text_generation(
            self.tokenizer,
            self.model,
            self.dataset_val[0]["messages"],
            self.max_seq_length,
        )
        logging.basicConfig(filename="logs/training.log", level=logging.INFO)

    def GetChatTemplate_Type(self):
        if 'gemma' in self.model_name:
            return 'gemma_chatml'
        if 'unsloth' in self.model_name:
            return 'unsloth'
        if 'zephyr' in self.model_name:
            return 'zephyr'
        if 'mistral' in self.model_name:
            return 'mistral'
        if 'alpaca' in self.model_name:
            return 'alpaca'
        if 'llama' in self.model_name:
            return 'llama'
        else:
            return 'chatml'

    def log_end_of_training(self):
        # Log the values to a local log file
        logging.info(f"Base Model name: {self.model_name}")
        logging.info(f"Output Model name: {self.out_model_name}")
        logging.info(f"Max sequence length: {self.max_seq_length}")
        logging.info(f"Load in 4-bit: {self.load_in_4bit}")
        logging.info(f"Fine-tuned model directory: {str(self.fine_tuned_model_dir)}")
        logging.info(f"Weights & Biases run URL: {self.wandb_run_url}")
        logging.info(f"Weights & Biases run path: {self.wandb_run_path}")

    def generate_summary(self, messages, temperature=0.7, top_k=20, top_p=0.95, repetition_penalty=1.2):
        try:
            # Check if the model is in training mode
            if self.model.training:
                # If it's in training mode, switch it to inference mode
                FastLanguageModel.for_inference(self.model)
            # check if the input token length is less than the max_seq_length, if it is set truncation to True
            truncation = check_token_threshold_and_truncate(
                self.tokenizer, self.model, messages, self.max_seq_length
            )
            # Tokenize the input messages
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # Must add for generation
                return_tensors="pt",
                max_length=self.max_seq_length,
                truncation=truncation,
            ).to(device=self.device)
            # Generate the summary
            summary_ids = self.model.generate(
                input_ids=inputs,
                max_new_tokens=self.max_seq_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            # Decode the summary
            summary_text = self.tokenizer.decode(
                summary_ids[0][inputs.shape[1] :], skip_special_tokens=True
            )
            return summary_text
        except RuntimeError as e:
            print(f"An error occurred during summary generation: {e}")
            if "probability tensor contains either `inf`, `nan` or element < 0" in str(e):
                print("Adjusting parameters and retrying...")
                return self.generate_summary(messages, temperature=temperature+0.1, top_k=top_k+5, top_p=min(1, top_p+0.05), repetition_penalty=max(1, repetition_penalty-0.1))

    def GetOutputModelName(self):
        # Get the base name of the model and use it to name the fine-tuned model
        base_name_parts = self.model_name.split("/")
        base_name = (
            base_name_parts[-1] if len(base_name_parts) > 1 else base_name_parts[0]
        )
        base_name_parts = base_name.split("-")
        if "Summarizer" in base_name_parts:
            base_name_parts.remove("Summarizer")
        base_name_parts.insert(1, "Summarizer")
        out_model_name = "-".join(base_name_parts)
        return out_model_name

    def GetRandomValidationMessage(self):
        # Get the length of the list
        length = len(self.dataset_val["messages"])
        # Generate a random index
        index = random.randrange(length)        # trunk-ignore(bandit/B311)
        # Access the element at the random even index
        messages_chat = self.dataset_val[index]["messages"]
        # Remove reference from dictionaries with role 'assistant'
        for message in messages_chat:
            if message["role"] == "assistant":
                message["content"] = ""
        messages_text = self.dataset_val[index]["text"]
        messages_str = "".join(messages_text)
        Reference_summary = messages_str.split("assistant", 1)[1]
        return messages_chat, Reference_summary

    def save_model(self, model, tokenizer, eval_file_path=None, evaluation=''):
        model_saver = ModelSaver(
            model,
            tokenizer,
            self.fine_tuned_model_dir,
            self.out_model_name,
            self.wandb_run_url,
            self.wandb_run_path,
            eval_file_path,
            evaluation,
        )
        model_saver.save_model()

    def Test_Model(self):
        while True:
            # Get a random validation message
            message, _ = self.GetRandomValidationMessage()

            # Ask the user for input
            user_input = input("Do you want to test the model? (y/n): ")
            if user_input.lower() == "y":
                eval = Evaluator(self.model,
                                 self.tokenizer,
                                 self.model_name,
                                 self.local_model_path,
                                 self.max_seq_length,
                                 self.load_in_4bit,
                                 self.run_name)
                eval_file_path, model, tokenizer, evaluation = eval.evaluate(message)
                eval.display(self.eval_file_path)

            elif user_input.lower() == "n":
                continue

            # Ask the user for input
            user_input = input("Enter 'r' to retry, 's' to save, or 'q' to quit: ")

            if user_input.lower() == "r":
                continue
            elif user_input.lower() == "s":
                self.save_model(model, tokenizer, eval_file_path, evaluation)
                print("Model saved.")
                break
            elif user_input.lower() == "q":
                break
            else:
                print("Invalid input. Please enter 'r', 's', or 'q'.")

    def trainer(self):
        model_name = self.model_name
        model = self.model
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            max_seq_length=self.max_seq_length,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
        self.run_name = f"run-{model_name}-{random.randint(0, 100000)}"       # trunk-ignore(bandit/B311)
        run = wandb.init(project="LLM-summarizer_trainer", name=self.run_name)
        self.wandb_run_url = run.get_url()
        self.wandb_run_path = run.path
        trainer = SFTTrainer(
            model=model,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_val,
            max_seq_length=self.max_seq_length,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            packing=False,
            args=TrainingArguments(
                fp16_full_eval=True,
                per_device_eval_batch_size=2,
                eval_accumulation_steps=4,
                evaluation_strategy="steps",
                eval_steps=5,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=20,
                max_steps=380 if not self.dry_run else 20,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                output_dir="outputs",
                save_strategy="steps",
                save_steps=20,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                load_best_model_at_end=True,
            ),
            callbacks=[
                SampleGenerationCallback(
                    every_x_steps=20,
                    dataset_val=self.dataset_val,
                    generate_summary=self.generate_summary,
                    generate_random_text=self.GetRandomValidationMessage,
                    score_threshold = 0.001,
                    patience=1000,
                    min_delta=0.001,
                    warmup_steps=60,
                ),
                EarlyStoppingCallback(
                    early_stopping_patience=20, early_stopping_threshold=0.001
                ),
                WandbCallback(),
            ],
        )
        return trainer

    def Start_Training(self, trainer):
        local_model_path = f"{self.fine_tuned_model_dir}/{self.out_model_name}"
        if glob.glob("outputs/checkpoint-*"):
            trainer.train(resume_from_checkpoint=True)
            if wandb.run is not None:
                wandb.finish()
            print(f"\nTraining completed for {self.out_model_name}\n\n")
            trainer.save_model(local_model_path)
            self.log_end_of_training()
        else:
            trainer.train(resume_from_checkpoint=False)
            if wandb.run is not None:
                wandb.finish()
            print(f"\nTraining completed for {self.out_model_name}\n\n")
            trainer.save_model(local_model_path)
            self.log_end_of_training()


def main():
    # list of models available for fine-tuning on unsloth
    standard_models, faster_models = get_model_list()
    # Select the model to fine-tune
    selected_model, is_4bit = select_model(standard_models, faster_models)
    # Check if the selected model is a 4-bit model
    print("\nSelected model:", selected_model)
    print("\nIs the selected model a 4-bit model?", is_4bit)
    # Instantiate the trainer with the desired parameters
    trainer_instance = Unsloth_LLM_Trainer(
        model_name=selected_model,  # replace with your model name
        load_in_4bit=is_4bit,
        max_seq_length=1024,
        dry_run=False,
    )
    print("\n\nInitialization done\n\n")
    # Get the trainer
    trainer = trainer_instance.trainer()
    print("\n\nTrainer created\n\n")
    # Start the training
    trainer_instance.Start_Training(trainer)
    print("\n\nTraining ended\n\n")
    # test and save the model
    trainer_instance.Test_Model()
    print("\n\Training done\n\n")


if __name__ == "__main__":
    main()
