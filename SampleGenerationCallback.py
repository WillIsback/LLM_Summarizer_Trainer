# Description: This file contains the SampleGenerationCallback class which is used to generate and score summaries during training.
from transformers import TrainerCallback
import wandb
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
import csv
import os

class SampleGenerationCallback(TrainerCallback):
    def __init__(self, every_x_steps=5, dataset_val=None, generate_summary=None, generate_random_text=None, score_threshold = 0.2, patience=5, min_delta=0.01, warmup_steps=10):
        self.every_x_steps = every_x_steps
        self.dataset_val = dataset_val
        self.generate_summary = generate_summary
        self.generate_random_text = generate_random_text
        self.score_threshold = score_threshold
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_steps = warmup_steps
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.summary_table_data = []
        # Initialize the best scores dictionary
        self.best_scores = {'rouge1': None, 'rouge2': None, 'rougeL': None}
        # Initialize the scores history list
        self.scores_history = []
        self.patience_counter = 0

    def generate_and_score_summary(self):
        try:
            messages_chat, Reference_summary = self.generate_random_text()
            summary_text = self.generate_summary(messages_chat)
            if summary_text is None:
                print("Summary generation failed.")
                return None, None, None, None, None
            scores = self.rouge.score(Reference_summary, summary_text)
            rouge1 = scores['rouge1'].fmeasure
            rouge2 = scores['rouge2'].fmeasure
            rougeL = scores['rougeL'].fmeasure
            # Add the scores to the history
            self.scores_history.append((rouge1, rouge2, rougeL))
            return summary_text, Reference_summary, rouge1, rouge2, rougeL
        except RuntimeError as e:
            print(f"An error occurred during summary generation: {e}")
            return None, None, None, None, None

    def log_rouge_scores(self, rouge1, rouge2, rougeL):
        # Log the rouge scores one by one
        wandb.log({"ROUGE-1": rouge1, "ROUGE-2": rouge2, "ROUGE-L": rougeL})

    def log_summary_table(self, Reference_summary, summary_text, rouge1, rouge2, rougeL, state):
        # Log in my_table the score and the summary every x step
        if state.global_step % self.every_x_steps == 0:
            self.summary_table_data.append([Reference_summary, summary_text, f"Rouge-1: {rouge1},\n Rouge-2: {rouge2},\n Rouge-L: {rougeL}"])
            my_table = wandb.Table(columns=["Reference_summary", "Generated_summary", "Rouge-Score"], data=self.summary_table_data)
            wandb.log({"summary_table": my_table})

    def calculate_and_log_moving_average(self, global_steps):
        # Calculate the moving average of the scores
        # Moving average smooths out short-term fluctuations and highlights longer-term trends in the Rouge scores.
        scores_df = pd.DataFrame(self.scores_history, columns=['rouge1', 'rouge2', 'rougeL'])
        moving_avg = scores_df.rolling(window=3).mean().values[-1]
        data = [[i, avg] for i, avg in zip(global_steps, moving_avg)]
        # Log the moving average as a plot
        table = wandb.Table(data=data, columns=["Step", "Moving Average"])
        wandb.log({"moving_average_plot": wandb.plot.line(table, "Step", "Moving Average", title="Moving Average Plot")})
        return moving_avg

    def calculate_and_log_exp_smoothing(self, global_steps):
        # Calculate the exponential smoothing of the scores
        # Exponential smoothing also smooths the scores but gives more weight to recent scores, useful if the model's performance is changing over time.
        scores_df = pd.DataFrame(self.scores_history, columns=['rouge1', 'rouge2', 'rougeL'])
        exp_smoothing = scores_df.ewm(span=3).mean().values[-1]
        data = [[i, exp] for i, exp in zip(global_steps, exp_smoothing)]
        # Log the exponential smoothing as a plot
        table = wandb.Table(data=data, columns=["Step", "Exponential Smoothing"])
        wandb.log({"exp_smoothing_plot": wandb.plot.line(table, "Step", "Exponential Smoothing", title="Exponential Smoothing Plot")})
        return exp_smoothing

    def calculate_and_log_derivative(self, global_steps):
        # Calculate the derivative (rate of change) of the scores
        # The derivative gives the rate of change of the Rouge scores, useful to understand how quickly the scores are improving or declining.
        # Initialize avg_derivative to a default value
        avg_derivative = [0, 0, 0]
        if len(self.scores_history) > 1:
            derivatives = np.diff(self.scores_history, axis=0)
            avg_derivative = np.mean(derivatives, axis=0)
        data = [[i, der] for i, der in zip(global_steps, avg_derivative)]
        # Log the derivative as a plot
        table = wandb.Table(data=data, columns=["Step", "Derivative"])
        wandb.log({"derivative_plot": wandb.plot.line(table, "Step", "Derivative", title="Derivative Plot")})
        return avg_derivative

    def calculate_and_log_integral(self, global_steps):
        # Calculate the integral (total change) of the scores
        # The integral gives the total accumulated value of the Rouge scores, useful to understand the overall performance of the model.
        integral = [0, 0, 0]
        if len(self.scores_history) > 2:
            integral = np.trapz(self.scores_history, axis=0)
        # Use global_steps for the x-axis data
        data = [[step, intg] for step, intg in zip(global_steps, integral)]
        table = wandb.Table(data=data, columns=["Step", "Integral"])
        wandb.log({"integral_plot": wandb.plot.line(table, "Step", "Integral", title="Integral Plot")})
        return integral

    def check_warmup_steps(self, state, avg_derivative, rouge1, rouge2, rougeL, moving_avg, exp_smoothing):
        # Check if the warmup steps have been exceeded
        # If the average derivative is positive, the scores are improving, else they are declining
        if state.global_step > self.warmup_steps:
            # Check if all derivatives are positive (scores are improving)
            # and if the latest score is greater than the moving average and exponential smoothing (model is not underperforming)
            if all(derivative > 0 for derivative in avg_derivative) and all(score > avg for score, avg in zip([rouge1, rouge2, rougeL], moving_avg)) and all(score > exp for score, exp in zip([rouge1, rouge2, rougeL], exp_smoothing)):
                self.best_scores = {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}
                self.patience_counter = 0
            else:
                self.patience_counter += 1

    def check_patience(self, state, rouge1, rouge2, rougeL, control):
        # Check if the patience has been exceeded
        # If the patience counter exceeds a certain limit, it means the model is not improving and training should stop.
        if self.patience_counter >= self.patience:
            control.should_training_stop = True
            print(f"\033[91m\nEarly stopping at step {state.global_step}, rouge scores performance decrease: {rouge1}, {rouge2}, {rougeL}\n\033[0m")

    def on_step_end(self, args, state, control, model, **kwargs):
        # Score the summary metrics at every eval_steps
        if state.global_step % args.eval_steps == 0:
            summary_text, Reference_summary, rouge1, rouge2, rougeL = self.generate_and_score_summary()
            if summary_text is None:
                print("Summary generation failed. Skipping this step.")
                return
            self.log_rouge_scores(rouge1, rouge2, rougeL)
            self.log_summary_table(Reference_summary, summary_text, rouge1, rouge2, rougeL, state)
            moving_avg = self.calculate_and_log_moving_average(state.global_step)
            exp_smoothing = self.calculate_and_log_exp_smoothing(state.global_step)
            avg_derivative = self.calculate_and_log_derivative(state.global_step)
            integral = self.calculate_and_log_integral(state.global_step)
            self.check_warmup_steps(state, avg_derivative, rouge1, rouge2, rougeL, moving_avg, exp_smoothing)
            self.check_patience(state, rouge1, rouge2, rougeL, control)

            # Log metrics to a local CSV file
            log_dir = os.path.join('logs', wandb.run.name)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'metrics.csv')
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([state.global_step, rouge1, rouge2, rougeL, moving_avg, exp_smoothing, avg_derivative, integral])
