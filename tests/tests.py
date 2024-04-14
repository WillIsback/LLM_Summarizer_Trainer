from unsloth import FastLanguageModel

def test_dataset(dataset):
    # Check the structure of the dataset
    assert all(key in dataset.features for key in ['messages']), "Dataset structure is incorrect"

    # Check the content of the dataset
    for example in dataset:
        assert 'messages' in example, "Missing 'messages' in example"

        # Check that 'messages' is a list of dictionaries
        assert isinstance(example['messages'], list), "'messages' should be a list"
        for message in example['messages']:
            assert isinstance(message, dict), "Each message should be a dictionary"
            assert 'role' in message, "Missing 'role' in message"
            assert 'content' in message, "Missing 'content' in message"

    # Check the length of the dataset
    assert len(dataset) > 0, "Dataset is empty"

def check_token_threshold_and_truncate(tokenizer, model, messages_chat, max_seq_length):
    # Check if the input token length is less than the max_seq_length
    input_token_length = len(tokenizer.apply_chat_template(messages_chat, tokenize=True))

    if model.config.max_position_embeddings is not None:
        max_model_token_config = model.config.max_position_embeddings
    else:
        max_model_token_config = tokenizer.model_max_length

    MaxTokenCapacityThreshold = (max_model_token_config - (input_token_length + max_seq_length)) < 0

    if MaxTokenCapacityThreshold:
        print("Warning: Maximum token threshold has been reached. Activating truncation to prevent crash. Rouge score will be affected.")
        truncation = True
    else:
        truncation = False
    return truncation

def test_text_generation(tokenizer, model, message_chat, max_seq_length):
    for message in message_chat:
        if message['role'] == 'assistant':
            message['content'] = ''
    # Check if the model is in training mode
    if model.training:
        # If it's in training mode, switch it to inference mode
        FastLanguageModel.for_inference(model)
    #check if the input token length is less than the max_seq_length, if it is set truncation to True
    truncation = check_token_threshold_and_truncate(tokenizer, model, message_chat, max_seq_length)
    # Tokenize the input messages

    inputs = tokenizer.apply_chat_template(
        message_chat,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
        max_length=max_seq_length,
        truncation = truncation,
    ).to(device='cuda')

    # Generate the summary
    summary_ids = model.generate(
        input_ids=inputs,
        max_new_tokens=max_seq_length,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.3,
        top_k=20,
        top_p=0.95,
        repetition_penalty=1.2,
    )
    # Decode the summary
    summary_text = tokenizer.decode(summary_ids[0][inputs.shape[1]:], skip_special_tokens=True)
    # Check if the summary text is not None
    assert summary_text is not None, "Final summary text should not be None"
    # Split the summary text into lines
    summary_lines = summary_text.split('\n')
    # Return the first 3 lines
    sample_text = '\n'.join(summary_lines[:10])
    sample = f"\n\nGeneration test, small sample result: \n\n{sample_text}\n\n"
    print(sample)
    return summary_text

