# ChatTemplate

The `ChatTemplate.py` script is a key part of the LLM Summarizer Trainer project. It defines a `ChatTemplate` class that is used to preprocess datasets in chat conversational format to fit the trainer.

## Key Points

The `ChatTemplate` class is initialized with a tokenizer. It has several methods:

- `formating_messages`: This method takes an example from the dataset and formats it into a dictionary with a "messages" key. The value of the "messages" key is a list of dictionaries, each representing a chat message with "role" and "content" keys.

- `formatting_prompts_func`: This method takes a batch of examples and applies the chat template to the "messages" of each example. It returns a dictionary with a "text" key and a list of formatted texts as the value.

- `load_data`: This method loads the training and validation datasets, groups the data, converts the grouped data into a DataFrame, creates a new Dataset object from the DataFrame, and applies the `formating_messages` and `formatting_prompts_func` methods to the datasets.

## Configuration

In the `load_data` method, you can configure the chat template and the mapping by modifying the arguments passed to the `get_chat_template` function. The `chat_template` argument specifies the chat template to use. You can choose from several chat templates as described in this [link](https://github.com/unslothai/unsloth/blob/4606443b77f98a624896d4ca50710255d8436d86/unsloth/chat_templates.py#L258). For example, you can change `chat_template = "chatml"` to `chat_template = "zephyr"` to use the zephyr chat template.

The `mapping` argument specifies the mapping between the roles and contents of the chat messages and the keys in the dataset. You can configure this by modifying the following code:

```python
user_chat = {"role": example["user"]["role"], "content": example["user"]["content"]}
assistant_chat = {"role": example["assistant"]["role"], "content": example["assistant"]["content"]}
```

Here, you need to replace `"user"` and `"assistant"` with the keys present in your dataset. For example, if your dataset uses `"human"` and `"gpt"` as the keys, you can modify the code as follows:

```python
user_chat = {"role": example["user"]["role"], "content": example["human"]["content"]}
assistant_chat = {"role": example["assistant"]["role"], "content": example["gpt"]["content"]}
```

To use a different dataset, you need to modify the arguments passed to the `load_dataset` function. The first argument is the name of the dataset to load. For example, you can change `"Labagaite/fr-summarizer-dataset"` to `"your_dataset_name"` to load your dataset. The `split` argument specifies the split of the dataset to load. Note that this script is designed to work with datasets in chat format where each entry is an instruction and the next entry is the response. The script first groups the entries two by two to form a single entry as a conversation.

## Usage

To use the `ChatTemplate` class to preprocess a dataset, you need to call the `load_data` method. This method returns the preprocessed training and validation datasets. Here's an example:

```python
dataset_train, dataset_val = chat_template.load_data()
```

In this example, `dataset_train` and `dataset_val` are the preprocessed training and validation datasets, respectively. You can then pass these datasets to the trainer for training.

For more details on how the `ChatTemplate` class works and how to use it, please refer to the [ChatTemplate Documentation](docs/ChatTemplate.md).