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