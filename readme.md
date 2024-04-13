<p align="center">
  <img src="images/Llm_Summarizer_trainer_icon-removebg.png" width="200"/>
</p>

<h1 align="center">LLM Summarizer Trainer</h1>

<p align="center">
  <strong>Fine-tuning Large Language Models for text summarization</strong>
</p>
<div style="display: flex; justify-content: center;">
  <div style="display: inline-block; text-align: center;">
    <a href="https://github.com/WillIsback/Report_Maker">
      <img src="https://avatars.githubusercontent.com/u/116890814?v=4" width="100"/>
      <br>
      <span>Report Maker</span>
    </a>
  </div>
  <div style="display: inline-block; text-align: center; margin-left: 10px; margin-right: 10px;">
    <a href="https://huggingface.co/Labagaite">
      <img src="https://avatars.githubusercontent.com/u/25720743?s=200&v=4" width="100"/>
      <br>
      <span>Model & Data</span>
    </a>
  </div>
  <div style="display: inline-block; text-align: center;">
    <a href="https://github.com/unslothai/unsloth">
      <img src="https://unsloth.ai/cgi/image/unsloth_green_sticker_cME6ryC59BlZg-VtqGN4p.png?width=640&quality=80&format=auto" width="100"/>
      <br>
      <span>unsloth</span>
    </a>
  </div>
</div>

## Table of Contents
- [Summary](#summary)
- [Motivation](#motivation)
- [Setup](#setup)
  - [Automated Setup (Linux)](#automated-setup-linux)
  - [Manual Setup](#manual-setup)
- [Setting Up Environment Variables](#setting-up-environment-variables)
- [How to Use](#how-to-use)
- [Model Selection](#model-selection)
- [Data Preprocessing](#data-preprocessing)
- [Training the Model](#training-the-model)
  - [Training Metrics](#training-metrics)
- [Saving and Updating the Model](#saving-and-updating-the-model)
- [Conclusion](#conclusion)
- [License](#license)

## Summary
The `trainer.py` script is part of the LLM Summarizer Trainer project. It is used to fine-tune a Large Language Model (LLM) for the task of summarizing text using QLora as the fine-tuning method. The script leverages the power of the Hugging Face Transformers library, the Weights & Biases tool for experiment tracking, and the TRL library for training. All in the fast tool [unsloth](https://github.com/unslothai/unsloth).
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png" width="200"/>](https://github.com/unslothai/unsloth)

## Motivation
[<img src="https://avatars.githubusercontent.com/u/116890814?v=4" width="100"/>](https://github.com/WillIsback/Report_Maker)
To fine-tuned on French audio podcast transcription data for summarization task. The model will be used for an AI application: [Report Maker](https://github.com/WillIsback/Report_Maker) wich is a powerful tool designed to automate the process of transcribing and summarizing meetings. It leverages state-of-the-art machine learning models to provide detailed and accurate reports.

## Setup

### Automated Setup (Linux)

If you are using a Linux system, you can simply run the `setup.sh` script to set up your environment. This script creates a virtual environment, installs the necessary requirements, and configures the environment based on your GPU architecture and PyTorch version.

To run the script, open a terminal and navigate to the directory containing the `setup.sh` file. Then, run the following command:

```bash
./setup.sh
```

### Manual Setup

If you prefer to set up your environment manually or are using a different operating system, follow these steps:

1. **Create a virtual environment**: You can use `venv` to create a virtual environment. Open a terminal and run the following command:

    ```bash
    python3 -m venv env
    ```

2. **Activate the virtual environment**: The command to activate the environment depends on your operating system:

    - On Linux or MacOS, run:

        ```bash
        source env/bin/activate
        ```

3. **Install the requirements**: The `requirements.txt` file lists the Python packages that your project depends on. You can install these using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    This will install the following packages:

    - `wandb`: Weights & Biases, a tool for tracking and visualizing machine learning experiments.
    - `rouge_score`: A Python package for calculating the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score, a common metric for evaluating text summaries.
    - `evaluate`: A package for evaluating machine learning models.

4. **Install additional packages based on your GPU architecture and PyTorch version**: Refer to the `setup.sh` script for the specific packages to install based on your GPU architecture (Ampere or older) and PyTorch version (2.1.0, 2.1.1, 2.2.0, or 2.2.1). You can check your GPU architecture and PyTorch version using the following Python commands:

    ```python
    import torch
    print(torch.version.cuda)  # prints the CUDA version
    print(torch.version.__version__)  # prints the PyTorch version
    print(torch.cuda.get_device_capability()[0])  # prints the GPU architecture
    ```

    Then, install the appropriate packages using `pip`. For example, if your GPU architecture is Ampere or newer and your PyTorch version is 2.2.1, you would run:

    ```bash
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
    ```

Remember to replace the commands with the appropriate ones for your GPU architecture and PyTorch version. Find more at [unsloth Github](https://github.com/unslothai/unsloth)

## Setting Up Environment Variables

For the application to function correctly, you need to set up several environment variables. These variables hold the API keys for various services that the application uses.

Create a `.env` file in the root directory of the project and add the following lines to it:

```properties
HUGGING_FACE=your_hugging_face_token
WANDB_API_KEY=your_wandb_api_key
# Add more tokens as needed
```

Replace `your_hugging_face_token`, `your_openai_key`, and `your_wandb_api_key` with your actual API keys.

- `HUGGING_FACE`: Your Hugging Face API token. You can find this on your Hugging Face account page.
- `WANDB_API_KEY`: Your Weights & Biases API key. You can find this on your Weights & Biases account page.

Remember not to share these tokens with anyone or publish them online. They provide access to your accounts on these services, and anyone with these tokens can use these services as if they were you.


## How to Use

To use the script, you need to have the necessary libraries installed, including `torch`, `transformers`, `trl`, `wandb`, and others. You also need to have a Hugging Face and Weights & Biases account, and the API keys should be stored in environment variables.

To run the script, navigate to the directory containing the script and run the command:

```bash
python trainer.py
```

During the testing phase, you will be prompted to enter 'r' to retry with a different message, 's' to save the model, or 'q' to quit.

## Model Selection

The LLM Summarizer Trainer includes functionality to select a model for fine-tuning. The `model_selector.py` script is used for this purpose. It allows you to search for a model, select a model from a list, or select a model from a local folder.

For more details on how to use the `model_selector.py` script to select a model, please refer to the [Model Selection Documentation](docs/model_selection.md).


The `trainer.py` script provides functionality to train a model using the Hugging Face's `Trainer` class. It contains two main methods: `trainer` and `Start_Training`.

## Data Preprocessing

The `ChatTemplate.py` script is a crucial part of the LLM Summarizer Trainer project. It defines a `ChatTemplate` class that is used to preprocess datasets in chat conversational format to fit the trainer.

The `ChatTemplate` class is initialized with a tokenizer and has several methods for formatting and loading the data. It allows you to configure the chat template and the mapping between the roles and contents of the chat messages and the keys in the dataset. It also allows you to load any dataset in chat format.

For more details on how to use the `ChatTemplate.py` script to preprocess your data, please refer to the [Data Preprocessing Documentation](docs/ChatTemplate.md).

## Training the Model

The `trainer.py` script provides functionality to train a model using the Hugging Face's `Trainer` class. It contains two main methods: `trainer` and `Start_Training`.

The `trainer` method initializes the model and the trainer. It first patches the model and adds fast LoRA weights using the `get_peft_model` method of the `FastLanguageModel` class. It then initializes a Weights & Biases run and sets up the trainer with the model, the training and evaluation datasets, the tokenizer, and the training arguments.

The `Start_Training` method starts the training process. It checks if there are any saved checkpoints in the `outputs` folder. If there are, it resumes training from the latest checkpoint. If there are no saved checkpoints, it starts training from scratch.

After training is completed, the method saves the model to the specified directory and logs the end of training.

The trainer saves checkpoints after every few steps as specified in the training arguments. This allows you to resume training from the latest checkpoint if the training process is interrupted for any reason.

### Training Metrics

During the training process, the `SampleGenerationCallback.py` script is used to generate and score summaries at regular intervals. This custom callback class monitors the performance of the model and provides a form of early stopping. It generates a random summary, scores it using the ROUGE metric, and logs the results to Weights & Biases. If the ROUGE-2 score falls below a certain threshold, it stops the training process.

For more details on how the `SampleGenerationCallback` works and why it's used, please refer to the [Training Metrics Documentation](docs/training_metrics.md).

For more details on how the `Trainer` works , please refer to the [Training Documentation](docs/training_model.md).

## Saving and Updating the Model

The LLM Summarizer Trainer includes functionality to save the fine-tuned model and update the model card on Hugging Face Model Hub. The `ModelSaver` class in `modelSaver.py` is used for this purpose. It allows you to save the model in various formats and update the model card with important information about the model, including its base model, the method used for training, the ROUGE scores achieved, and a link to the training logs on Weights & Biases.

For more details on how to use the `ModelSaver` class to save the model and update the model card, please refer to the [Save Model Documentation](docs/save_model.md).



## Conclusion

The `trainer.py` script provides a comprehensive pipeline for fine-tuning a Large Language Model for the task of summarizing text. It includes features for logging, model saving, and testing, making it a versatile tool for model training and evaluation.

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.
