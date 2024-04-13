# Description: This file contains the SampleGenerationCallback class which is used to generate and score summaries during training.
from transformers import TrainerCallback
import wandb
import random
from rouge_score import rouge_scorer

class SampleGenerationCallback(TrainerCallback):
    def __init__(self, every_x_steps=5, dataset_val=None, generate_summary=None, score_threshold = 0.2, patience=5, min_delta=0.01, warmup_steps=10):
        self.every_x_steps = every_x_steps
        self.dataset_val = dataset_val
        self.generate_summary = generate_summary
        self.score_threshold = score_threshold
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_steps = warmup_steps
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.summary_table_data = []
        self.best_score = None
        self.patience_counter = 0

    def generate_and_score_summary(self):
        # Get the length of the list
        length = len(self.dataset_val["messages"])
        # Generate a random index
        index = random.randrange(0, length)
        messages_chat = self.dataset_val[index]["messages"]
        # Remove content from dictionaries with role 'assistant'
        for message in messages_chat:
            if message['role'] == 'assistant':
                message['content'] = ''
        messages_text = self.dataset_val[index]["text"]
        messages_str = "".join(messages_text)
        Reference_summary = messages_str.split('assistant', 1)[1]
        summary_text = self.generate_summary(messages_chat)
        scores = self.rouge.score(Reference_summary, summary_text)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure
        return summary_text, Reference_summary, rouge1, rouge2, rougeL

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % self.every_x_steps == 0:
            summary_text, Reference_summary, rouge1, rouge2, rougeL = self.generate_and_score_summary()
            self.summary_table_data.append([Reference_summary, summary_text, f"Rouge-1: {rouge1},\n Rouge-2: {rouge2},\n Rouge-L: {rougeL}"])
            my_table = wandb.Table(columns=["Reference_summary", "Generated_summary", "Rouge-Score"], data=self.summary_table_data)
            wandb.log({"summary_table": my_table})

            if state.global_step % args.eval_steps == 0 and state.global_step > self.warmup_steps:
                _, _, rouge1, rouge2, rougeL = self.generate_and_score_summary()
                wandb.log({"ROUGE-1": rouge1, "ROUGE-2": rouge2, "ROUGE-L": rougeL})

                # Check if the performance has improved
                if self.best_score is None or rouge2 > self.best_score + self.min_delta:
                    self.best_score = rouge2
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Check if the patience has been exceeded
                if self.patience_counter >= self.patience:
                    control.should_training_stop = True
                    print(f"\033[91m\nEarly stopping at step {state.global_step}, rouge2 score did not improve: {rouge2}\n\033[0m")