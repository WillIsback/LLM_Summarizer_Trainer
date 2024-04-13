# Model Selection

The `model_selector.py` script provides functionality to select a model from a list of models compatible with Unsloth, search for a model, or select a model from a local folder. It contains two main functions: `select_model` and `get_model_list`.

## select_model Function

The `select_model` function allows the user to select a model in one of three ways:

1. **Search for a model:** The user can enter a search term, and the function will print out all models that contain this term in their name. The user can then enter the name of the model they want to select.

2. **Select a model from a list:** The function prints out a list of standard models and 4x faster models. The user can then enter the name of the model they want to select.

3. **Select a model from a local folder:** The user can enter the path of a local folder containing the model.

The function returns the selected model and a boolean value indicating whether the selected model is a 4x faster model.

## get_model_list Function

The `get_model_list` function retrieves a list of models from the Hugging Face Model Hub. It sends a GET request to the Hugging Face API and parses the response to separate the models into standard models and 4x faster models.

The function returns these two lists of models.

## Usage

To use the `select_model` function, you first need to get a list of models using the `get_model_list` function. Then, you can pass these lists to the `select_model` function. Here's an example:

```python
standard_models, faster_models = get_model_list()
selected_model, is_4bit = select_model(standard_models, faster_models)
```

This will prompt the user to select a model as described above.

Please note that you need to have the `requests` library installed to use the `get_model_list` function. You can install it with the following command:

```bash
pip install requests
```

This document provides a detailed description of the `model_selector.py` script. For more information on how to use this script in the context of the LLM Summarizer Trainer project, please refer to the main README file.