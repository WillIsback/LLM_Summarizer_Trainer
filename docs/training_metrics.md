# Training Metrics

The `SampleGenerationCallback.py` script is part of the LLM Summarizer Trainer project. It defines a custom callback class `SampleGenerationCallback` that is used during the training process to generate and score summaries. This class is a subclass of the `TrainerCallback` class from the Hugging Face Transformers library.

## How it Works

The `SampleGenerationCallback` class is initialized with several parameters:

- `every_x_steps`: The number of steps after which a summary is generated and scored.
- `dataset_val`: The validation dataset used to generate the summaries.
- `generate_summary`: A function that generates a summary given a list of messages.
- `score_threshold`: The threshold below which training should stop.

The class has two main methods:

- `generate_and_score_summary`: This method generates a random summary and scores it using the ROUGE metric. It first selects a random message from the validation dataset, generates a summary for it, and then scores the summary against the reference summary using the ROUGE-1, ROUGE-2, and ROUGE-L metrics.

- `on_step_end`: This method is called at the end of each training step. If the current step is a multiple of `every_x_steps`, it generates and scores a summary and logs the results to Weights & Biases. If the current step is a multiple of `args.eval_steps`, it generates and scores a summary and logs the ROUGE scores to Weights & Biases. If the ROUGE-2 score is below the `score_threshold`, it stops the training process.

## Why it's Used

The `SampleGenerationCallback` class is used to monitor the performance of the model during training. By generating and scoring summaries at regular intervals, it provides a way to track how well the model is learning to generate summaries. The ROUGE scores give a quantitative measure of the quality of the summaries, and logging these scores to Weights & Biases allows for easy tracking and visualization of the training progress.

The `SampleGenerationCallback` class also provides a form of early stopping. If the ROUGE-2 score falls below a certain threshold, it stops the training process. This can prevent overfitting and save computational resources by stopping the training process when the model is no longer improving.

## How to Use

To use the `SampleGenerationCallback` class, you need to initialize it with the appropriate parameters and pass it to the `Trainer` class when initializing the trainer. Here's an example:

```python
sample_generation_callback = SampleGenerationCallback(every_x_steps=5, dataset_val=dataset_val, generate_summary=generate_summary, score_threshold=0.2)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[sample_generation_callback]
)
```

In this example, `generate_summary` is a function that generates a summary given a list of messages. This function is defined in [Trainer script](../trainer.py)