# Description: This script contains the function select_model, which allows the user to select a model from a list of models, search for a model, or select a model from a local folder. The function also returns a boolean value indicating whether the selected model is a 4x faster model.
import requests

def select_model(standard_models, faster_models):
    print("Enter '1' to search for a model, '2' to select a model from a list, or '3' to select a model from a local folder.")
    choice = input()

    if choice == '1':
        print("Enter the name of the model you want to search for:")
        search_term = input()
        for model in standard_models + faster_models:
            if search_term in model:
                print(model)
        print("Enter the name of the model you want to select:")
        selected_model = input()
        is_4bit = selected_model.endswith('4bit')
        return selected_model, is_4bit

    elif choice == '2':
        print("Standard models:")
        for model in standard_models:
            print(model)
        print("\n4x faster models:")
        for model in faster_models:
            print(model)
        print("Enter the name of the model you want to select:")
        selected_model = input()
        is_4bit = selected_model.endswith('4bit')
        return selected_model, is_4bit

    elif choice == '3':
        print("Enter the path of the local folder containing the model:")
        folder_path = input()
        is_4bit = '4bit' in folder_path
        return folder_path, is_4bit

    else:
        print("Invalid choice. Please enter '1', '2', or '3'.")
        return select_model(standard_models, faster_models)

def get_model_list():
    response = requests.get('https://huggingface.co/api/models?search=unsloth')
    data = response.json()

    standard_models = []
    faster_models = []

    for model in data:
        if model['modelId'].endswith('4bit'):
            faster_models.append(model['modelId'])
        else:
            standard_models.append(model['modelId'])

    return standard_models, faster_models
