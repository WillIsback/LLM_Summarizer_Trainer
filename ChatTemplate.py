from datasets import load_dataset, Dataset
import pandas as pd
from unsloth.chat_templates import get_chat_template

class ChatTemplate():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def formating_messages(self,example):
        user_chat = {"role": example["user"]["role"], "content": example["user"]["content"]}
        assistant_chat = {"role": example["assistant"]["role"], "content": example["assistant"]["content"]}
        return {"messages": [user_chat, assistant_chat]}

    def formatting_prompts_func(self,examples):
        convos = examples["messages"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    def load_data(self):
        self.tokenizer = get_chat_template(
        self.tokenizer,
        chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}, # ShareGPT style
        map_eos_token = True, # Maps <|im_end|> to </s> instead
        )
        dataset_train = load_dataset("Labagaite/fr-summarizer-dataset", split = "train")
        dataset_val = load_dataset("Labagaite/fr-summarizer-dataset", split = "validation")
        # Group the data to pair the user and assistant messages in a single example
        grouped_data_train = [{"user": dataset_train[i], "assistant": dataset_train[i+1]} for i in range(0, len(dataset_train), 2)]
        grouped_data_val = [{"user": dataset_val[i], "assistant": dataset_val[i+1]} for i in range(0, len(dataset_val), 2)]
        # Convert the list of dictionaries to a DataFrame
        df_train = pd.DataFrame(grouped_data_train)
        df_val = pd.DataFrame(grouped_data_val)
        # Create a new Dataset object
        dataset_train = Dataset.from_pandas(df_train)
        dataset_val = Dataset.from_pandas(df_val)
        # Apply the formating functions to the datasets
        dataset_train = dataset_train.map(self.formating_messages, batched = False)
        dataset_train = dataset_train.map(self.formatting_prompts_func, batched = True)
        dataset_val = dataset_val.map(self.formating_messages, batched = False)
        dataset_val = dataset_val.map(self.formatting_prompts_func, batched = True)

        return dataset_train, dataset_val