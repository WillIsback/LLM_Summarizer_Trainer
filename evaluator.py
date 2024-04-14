# evaluator.py is a python script that evaluates the performance between fine tuned model and the base model. First it will write down summary from both in an output file for a manual evaluation. future features will include automatic evaluation of the model.
import torch
from unsloth import FastLanguageModel
from tests import test_text_generation
import gc
from openai import OpenAI
from tiktoken import get_encoding
from dotenv import load_dotenv
import os
from IPython.display import display, Markdown

class GPT:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        load_dotenv()
        # Get the OpenAI API key from the environment variables
        self.client = OpenAI(api_key=os.getenv('OPENAI_KEY'))
        self.tokenizer = get_encoding("cl100k_base")
        self.max_tokens = 16400
        self.max_output_tokens = 1024
        self.isNameEntitiesCleaned = False

    def tokenlen(self, text):
        return len(self.tokenizer.encode(text))


    def Evaluate(self, text, verbose=False):
        # Check if 'text' is a list of dictionaries with a 'sentence' field that contains a 'text' field
        if isinstance(text, list) and all(isinstance(sentence, dict) and 'sentence' in sentence and 'text' in sentence['sentence'] for sentence in text):
            # Extract the 'text' field from each sentence and join them into a single string
            text = ' '.join(sentence['sentence']['text'] for sentence in text)
        # If 'text' is not a list of such dictionaries, use it as it is
        elif isinstance(text, str):
            pass
        else:
            raise ValueError("Invalid input: 'text' should be a string or a list of dictionaries with a 'sentence' field that contains a 'text' field")
        evaluate_prompt =[
                    {
                        "role": "user",
                        "content": f"""tu vas recevoir 1 référence de l'insctruction reçus par 2 modèles d'IA et 2 exemplaires du rapport généré par ces modèles, l'un est généré par un model de base et l'autre par la version fine tuned du model de base.
                                        Tu es un évaluateur et tu dois évaluer les qualités et les défauts de chacun et leur attribuer 3 score sur 10. Ces scores seront basé sur la performances de la structuration du rapport, la qualité du language, la cohérence. finalement un score global avec une conclusion: \n\n{text}\n"""
                    },
                    {
                        "role": "assistant",
                                "content": ""
                    }]
        messages_length = self.tokenlen(f"{evaluate_prompt[0]['content']} {evaluate_prompt[1]['content']}")


        if verbose:
            print("\033[1;34m\n-------------------------------------------------------------------------------------\n\n\033[0m")
            print(f"\033[1;34m\nGenerating summary with GPT with input token size: {messages_length} and output: {self.max_output_tokens}\n\033[0m")
        # Use the language model to generate the summary
        response = self.client.chat.completions.create(model=self.model,
                                                messages=evaluate_prompt,
                                                max_tokens=self.max_output_tokens,
                                                temperature=0)
        # Extract the generated text from the response
        generated_response = response.choices[0].message.content
        reponse_token_size = self.tokenlen(generated_response)
        if verbose:
            print(f"\nResponse generated \033[1;34m{reponse_token_size}\033[1;32m token : \n\n\033[0m")
            print(f"{generated_response}\n")
        return generated_response


class Evaluator:
    def __init__(self, model, tokenizer, base_model_name,fine_tuned_model_name,max_seq_length,load_in_4bit, wandb_run_name):
        self.model = model
        self.tokenizer = tokenizer
        self.base_model_name = base_model_name
        self.fine_tuned_model_name = fine_tuned_model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.wandb_run_name = wandb_run_name
        self.gpt = GPT()
        self.eval_dir = os.path.join('evaluation', self.wandb_run_name)
        os.makedirs(self.eval_dir, exist_ok=True)

    def GetBaseModel_Summary(self, instructions):
        base_model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        base_summary = test_text_generation(base_tokenizer, base_model, instructions, self.max_seq_length)
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        return base_summary

    def GetFineTunedModel_Summary(self, instructions):
        fine_tuned_model = self.model
        fine_tuned_tokenizer = self.tokenizer
        fine_tuned_summary = test_text_generation(fine_tuned_tokenizer, fine_tuned_model, instructions, self.max_seq_length)
        del fine_tuned_model
        gc.collect()
        torch.cuda.empty_cache()
        return fine_tuned_summary

    def ReloadModel(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=self.fine_tuned_model_name,
        max_seq_length=self.max_seq_length,
        dtype=None,
        load_in_4bit=self.load_in_4bit,
        )
        return model, tokenizer
    def GetScore(self, instructions, base_summary, fine_tuned_summary):
        Comparativ_summary = f"""
        ## Reference instruction \n\n{instructions}\n\n
        ## Generated summary with base model {self.base_model_name} \n\n{base_summary}\n\n
        ## Generated summary with fine tuned model model {self.fine_tuned_model_name} \n\n{fine_tuned_summary}\n\n
        """
        gpt = GPT()
        evaluation = gpt.Evaluate(Comparativ_summary)
        return evaluation


    def WriteEvaluation(self, instructions, base_summary, fine_tuned_summary, evaluation):
        # save the model summary for human evaluation:
        fine_tuned_name_parts = self.fine_tuned_model_name.split("/")
        fine_tuned_ext = (
            fine_tuned_name_parts[-1] if len(fine_tuned_name_parts) > 1 else fine_tuned_name_parts[0]
        )
        Model_evaluator = f"""
# Report-Evaluator\n\n
## Table of Contents
- [Reference instruction](#Reference-instruction)
- [Generated summary with base model {self.base_model_name}](#Generated-summary-with-base-model-{self.base_model_name})
- [Generated summary with fine tuned model {fine_tuned_ext}](#Generated-summary-with-fine-tuned-model-{fine_tuned_ext})
- [Evaluation](#Evaluation)
## Reference instruction
<details>
<summary>View Full Instruction</summary>

```markdown
\n\n{instructions}\n\n
```

</details>

## Generated summary with base model
### {self.base_model_name}

<details>
<summary>View Full base summary</summary>

```markdown
\n\n{base_summary}\n\n
```

</details>

## Generated summary with fine tuned model
### {fine_tuned_ext}

<details>
<summary>View Full fine tuned summary</summary>

```markdown
\n\n{fine_tuned_summary}\n\n
```

</details>

## Evaluation: \n\n{evaluation}\n\n

"""




        eval_file_path = os.path.join(self.eval_dir, f'Model_evaluator-{fine_tuned_ext}.md')
        with open(eval_file_path, "a") as f:
            f.write(Model_evaluator)

        return eval_file_path

    def evaluate(self, instructions):
        print("\nEvaluating the fine tuned model and the base model ...\n")
        fine_tuned_summary = self.GetFineTunedModel_Summary(instructions)
        base_summary = self.GetBaseModel_Summary(instructions)
        evaluation = self.GetScore(instructions, base_summary, fine_tuned_summary)
        eval_file_path = self.WriteEvaluation(instructions, base_summary, fine_tuned_summary, evaluation)
        model, tokenizer = self.ReloadModel()
        print(f"\033[1;32m\n\nEvaluation completed check the result: \033[1;34m{eval_file_path}\033[1;32m \n\n\033[0m")
        return eval_file_path, model, tokenizer, evaluation

    def display(self, eval_file_path):
        # Read markdown file
        with open(eval_file_path, 'r') as f:
            md_content = f.read()
        # Display markdown in Jupyter notebook
        display(Markdown(md_content))


