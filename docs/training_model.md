# Training the Model

## trainer Method

The `trainer` method initializes the model and the trainer. It first patches the model and adds fast LoRA weights using the `get_peft_model` method of the `FastLanguageModel` class. It then initializes a Weights & Biases run and sets up the trainer with the model, the training and evaluation datasets, the tokenizer, and the training arguments.

The training arguments include settings for the batch size, gradient accumulation steps, warmup steps, maximum steps, learning rate scheduler, and more. The trainer also includes several callbacks for generating samples, early stopping, and logging to Weights & Biases.

## Start_Training Method

The `Start_Training` method starts the training process. It checks if there are any saved checkpoints in the `outputs` folder. If there are, it resumes training from the latest checkpoint. If there are no saved checkpoints, it starts training from scratch.

After training is completed, the method saves the model to the specified directory and logs the end of training.

The trainer saves checkpoints after every few steps as specified in the training arguments. This allows you to resume training from the latest checkpoint if the training process is interrupted for any reason.

## Usage

To use the `trainer` and `Start_Training` methods, you first need to initialize the `TrainerClass` with your model, tokenizer, training and evaluation datasets, and other necessary parameters. Then, you can call the `trainer` method to initialize the trainer and the `Start_Training` method to start the training process. Here's an example:

```python
trainer_class = TrainerClass(model, tokenizer, dataset_train, dataset_val, max_seq_length, out_model_name, fine_tuned_model_dir)
trainer = trainer_class.trainer()
trainer_class.Start_Training(trainer)
```

This document provides a detailed description of the `trainer.py` script. For more information on how to use this script in the context of the LLM Summarizer Trainer project, please refer to the main README file. In the next section, we will look in more detail at the metric and sample generation callback, a custom callback function.0