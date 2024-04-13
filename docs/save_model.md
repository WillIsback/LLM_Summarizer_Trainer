How to use the `ModelSaver` class to save the model.


## Saving the Model

The `ModelSaver` class in `modelSaver.py` is used to save the model. Here's how you can use it:

1. Initialize the `ModelSaver` class with your model, tokenizer, and other necessary parameters:

```python
model_saver = ModelSaver(model, tokenizer, fine_tuned_model_dir, out_model_name, wandb_run_url, wandb_run_path)
```

2. Call the `save_model` method of the `ModelSaver` class:

```python
model_saver.save_model()
```

When you run `save_model`, you will be prompted to enter the types of models you want to save. Options are: '16bit', '4bit', 'lora', 'gguf_q8_0', 'gguf_f16', 'gguf_q4_k_m'. You can enter 'all' to save all types. If you want to save multiple types, separate them with commas.

The `ModelSaver` class will then save your model in the specified formats and update the model card with the training details and performance metrics.

Please replace `model`, `tokenizer`, `fine_tuned_model_dir`, `out_model_name`, `wandb_run_url`, and `wandb_run_path` with your actual parameters. They are automatically retrieve by `trainer.py` during training.


## Updating the Model Card

The `ModelSaver` class in `modelSaver.py` also includes functionality to update the model card on Hugging Face Model Hub. The model card provides important information about the model, including its base model, the method used for training, the ROUGE scores achieved, and a link to the training logs on Weights & Biases.

The `UpdateModelCard` method is used to update the model card. It first retrieves the ROUGE scores from the Weights & Biases run using the `get_wandb_run` method. It then formats the model card content using these scores and other information about the model. Finally, it pushes the updated model card to the Hugging Face Model Hub.

Here's how you can use the `UpdateModelCard` method:

```python
model_saver.UpdateModelCard(save_directory, token)
```

Please replace `save_directory` and `token` with your actual parameters. The `save_directory` is the directory where the model is saved, and `token` is your Hugging Face API token.

The model card is formatted using the `CUSTOM_MODEL_CARD` string, which is a template for the model card content. You can modify this template to include any additional information you want to display on the model card.

