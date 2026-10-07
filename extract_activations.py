import json
import os
import argparse
import torch
import einops
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd

# Load configuration and model parameters
def load_config(config_file="config_extract_activation_fg_8B.yaml"):
    """
    Load the configuration file.

    Args:
        config_file (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration data.
    """
    with open(config_file, 'r') as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)

def load_tokenizer(model_name, hf_token):
    """
    Load the tokenizer for the specified model.

    Args:
        model_name (str): Name of the pre-trained model.
        hf_token (str): Hugging Face token.

    Returns:
        AutoTokenizer: Loaded tokenizer with pad_token set to eos_token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # Set padding side to right for causal language models
    return tokenizer

def load_model(model_name, hf_token, quantization_config):
    """
    Load the pre-trained model with quantization configuration.

    Args:
        model_name (str): Name of the pre-trained model.
        hf_token (str): Hugging Face token.
        quantization_config (dict): Quantization configuration parameters.

    Returns:
        AutoModelForCausalLM: Loaded model.
    """
    # Convert string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    compute_dtype = dtype_map.get(quantization_config.get("bnb_4bit_compute_dtype", "float16"), torch.float16)
    torch.cuda.empty_cache()

    if quantization_config.get("load_in_4bit", True):
        # Quantized path (used for 70B) — build BitsAndBytesConfig as before.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", False),
            bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},
            quantization_config=bnb_config,
            use_auth_token=hf_token,
        )
    else:
        # Full-precision path (used for 8B) — no BitsAndBytesConfig at all,
        # matches how 8B was run in the periodic-table project (fp16, no quantization).
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},
            torch_dtype=compute_dtype,
            use_auth_token=hf_token,
        )
    return model

# Prepare input data
def get_batch_mask(prompts, tokenizer):
    """
    Tokenize the input prompts.

    Args:
        prompts (list of str): Input prompts.
        tokenizer (AutoTokenizer): Tokenizer.

    Returns:
        tuple: input_ids and attention_mask tensors.
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    
    # Print number of valid tokens per input
    # print("Number of valid tokens per input:")
    # print(inputs["attention_mask"].sum(dim=1))
    return inputs["input_ids"], inputs["attention_mask"]

# Register activation hooks
def get_activation_hook(name, activations):
    """
    Create a forward hook to save activations.

    Args:
        name (str): Name of the layer.
        activations (dict): Dictionary to store activations.

    Returns:
        function: Hook function.
    """
    def hook(model, input, output):
        activations[name] = detach_tensor(output)
    return hook

def detach_tensor(tensor):
    """
    Detach tensor from computation graph.

    Args:
        tensor (torch.Tensor or tuple/list): Tensor to detach.

    Returns:
        torch.Tensor or tuple/list: Detached tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach()
    elif isinstance(tensor, (tuple, list)):
        return type(tensor)(detach_tensor(x) for x in tensor)
    else:
        return tensor

def register_hooks(model, activations):
    """
    Register forward hooks for each layer in the model.

    Args:
        model (AutoModelForCausalLM): The model.
        activations (dict): Dictionary to store activations.

    Returns:
        list: List of hook handles.
    """
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hook = layer.register_forward_hook(get_activation_hook(f'layer_{i}', activations))
        hooks.append(hook)
    return hooks

# Retrieve activations
def get_activations(model, input_ids, batch_mask):
    """
    Get activations from the model for given inputs.

    Args:
        model (AutoModelForCausalLM): The model.
        input_ids (torch.Tensor): Input IDs tensor.
        batch_mask (torch.Tensor): Attention mask tensor.

    Returns:
        dict: Activations per layer.
    """
    activations = {}
    hooks = register_hooks(model, activations)
    
    # The tokenizer creates CPU tensors, but the model is on the GPU.
    # Move both input tensors to the model's embedding device.
    input_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(input_device)
    batch_mask = batch_mask.to(input_device)
        
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=batch_mask, output_hidden_states=True)
    
    for hook in hooks:
        hook.remove()
    return activations

# Process activations
def process_activation_batch(activations, batch_mask, aggregation='last'):
    """
    Process activations based on the aggregation method.

    Args:
        activations (torch.Tensor): Activations tensor.
        batch_mask (torch.Tensor): Attention mask tensor.
        aggregation (str): Aggregation method ('last', 'mean', 'max', 'none').

    Returns:
        torch.Tensor: Processed activations.
    """
    if isinstance(activations, tuple):
        activations = activations[0]
    
    batch_mask = batch_mask.to(activations.device)

    if aggregation == 'last':
        # Get the activation of the last valid token
        last_ix = batch_mask.flip(dims=[1]).argmax(dim=1)
        processed_activations = activations[torch.arange(activations.size(0)), activations.size(1) - 1 - last_ix]
    
    elif aggregation == 'mean':
        # Mean activation of all valid tokens
        masked_activations = activations * batch_mask.unsqueeze(-1)
        valid_token_count = batch_mask.sum(dim=1, keepdim=True)
        processed_activations = masked_activations.sum(dim=1) / valid_token_count
    
    elif aggregation == 'max':
        # Max activation among all tokens
        masked_activations = activations * batch_mask.unsqueeze(-1)
        masked_activations[batch_mask == 0] = float('-inf')
        processed_activations = masked_activations.max(dim=1)[0]
    
    elif aggregation == 'none':
        # No aggregation, return activations for all valid tokens
        processed_activations = einops.rearrange(activations, 'b s d -> (b s) d')
        processed_activations = processed_activations[batch_mask.view(-1) == 1]
    
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")
    
    return processed_activations

# Get and process activations
def get_and_process_activations(model, tokenizer, prompts, aggregation='last'):
    """
    Get and process activations for a set of prompts.

    Args:
        model (AutoModelForCausalLM): The model.
        tokenizer (AutoTokenizer): The tokenizer.
        prompts (list of str): Input prompts.
        aggregation (str): Aggregation method.

    Returns:
        dict: Processed activations per layer.
    """
    input_ids, batch_mask = get_batch_mask(prompts, tokenizer)
    activations = get_activations(model, input_ids, batch_mask)

    processed_activations = {}
    for layer_name, layer_activations in activations.items():
        processed_activations[layer_name] = process_activation_batch(layer_activations, batch_mask, aggregation)
    
    return processed_activations

# Save activations

def save_activations(model_name, activations, entity_type, prompt_name, layer_ix, aggregation='last', save_dir='activation_datasets'):
    """
    Save activations to a specified directory as a .pt file.

    Args:
        model_name (str): Name of the model.
        activations (torch.Tensor): The activations to save.
        entity_type (str): Type of the entity (e.g., 'person', 'element').
        prompt_name (str): Name of the prompt.
        layer_ix (int): Index of the layer.
        aggregation (str): Type of aggregation used for activations.
        save_dir (str): Base directory where activations will be saved.
    """
    # Define the model-specific directory
    model_dir = os.path.join(save_dir, model_name.replace('/', '-'))
    
    # Define the save path
    activation_save_path = os.path.join(model_dir, entity_type)
    os.makedirs(activation_save_path, exist_ok=True)
    
    # Define the save file name
    save_name = f'{entity_type}.{aggregation}.{prompt_name}.layer_{layer_ix}.pt'
    save_path = os.path.join(activation_save_path, save_name)
    
    # Save the activations as a .pt file
    torch.save(activations, save_path)
    print(f"Activations saved at: {save_path}")


# Process and save activations
# Process and save activations
def process_and_save_activations_all_layers(model, tokenizer, prompts, num_layers, entity_type, prompt_name, model_name, aggregation='last', save_dir='activation_datasets', batch_size=550):
    all_activations = {layer_ix: [] for layer_ix in range(num_layers)}

    for start_ix in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start_ix:start_ix + batch_size]
        processed_activations = get_and_process_activations(model, tokenizer, batch_prompts, aggregation)

        for layer_ix in range(num_layers):
            layer_key = f'layer_{layer_ix}'
            if layer_key in processed_activations:
                all_activations[layer_ix].append(processed_activations[layer_key])
            else:
                print(f"Layer {layer_ix} not found in activations for batch starting at index {start_ix}.")

    for layer_ix in range(num_layers):
        if all_activations[layer_ix]:
            concatenated_activations = torch.cat(all_activations[layer_ix], dim=0)
            save_activations(
                model_name=model_name,
                activations=concatenated_activations,
                entity_type=entity_type,
                prompt_name=prompt_name,
                layer_ix=layer_ix,
                aggregation=aggregation,
                save_dir=save_dir
            )
        else:
            print(f"No activations found for Layer {layer_ix}.")


# Generate prompts
def generate_prompts(df, templates):
    """
    Generate prompts based on templates and DataFrame rows.

    Args:
        df (pandas.DataFrame): DataFrame containing entity data.
        templates (list of str): List of prompt templates.

    Returns:
        list of str: Generated prompts.
    """
    prompts = []
    for _, row in df.iterrows():
        for template in templates:
            try:
                prompt = template.format(**row.to_dict())
                prompts.append(prompt)
            except KeyError as e:
                print(f"Missing key in data for template: {e}")
    return prompts

# Main processing function
def main():
    """
    Main function to extract activations from language models.
    
    Configuration is loaded from config_extract_activation_fg_8B.yaml file. Key settings include:
    - extraction.model_name: Which model to use
    - extraction.batch_size: Batch size for processing
    - extraction.aggregation: How to aggregate token activations
    - extraction.save_dir: Directory to save activation files
    - extraction.quantization: Model quantization parameters
    - extraction.entities: List of entity types and their templates
    
    To change any of these settings, edit the config_extract_activation_fg_8B.yaml file.
    """
    # Load configuration
    parser = argparse.ArgumentParser(description="Extract activations from a language model.")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config_extract_activation_fg_8B.yaml",
        help="Path to the extraction config YAML file (default: config_extract_activation_fg_8B.yaml)"
    )
    args, _ = parser.parse_known_args()

    config_data = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    HF_TOKEN = config_data.get("HF_TOKEN")
    
    # Get extraction configuration
    extraction_config = config_data.get("extraction", {})
    model_name = extraction_config.get("model_name", "meta-llama/Llama-2-7b-hf")
    batch_size = extraction_config.get("batch_size", 550)
    aggregation = extraction_config.get("aggregation", "last")
    base_save_dir = extraction_config.get("save_dir", "activation_datasets")
    quantization_config = extraction_config.get("quantization", {})
    entities = extraction_config.get("entities", [])
    
    print(f"Using model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Aggregation method: {aggregation}")
    print(f"Save directory: {base_save_dir}")
    
    # Load tokenizer and model
    tokenizer = load_tokenizer(model_name, HF_TOKEN)
    model = load_model(model_name, HF_TOKEN, quantization_config)
    model.eval()
    
    # Validate entities configuration
    if not entities:
        print("No entities found in configuration. Please check your config_extract_activation_fg_8B.yaml file.")
        return
    
    print(f"Found {len(entities)} entity types in configuration.")
    
    # Use entities from configuration
    
    # Remove hardcoded entities list - now using configuration
    # The following entities are now defined in 	config_extract_activation_fg_8B.yaml under extraction.entities:
    # - atomic number, atomic mass, group, period, electronegativity (with various templates)
    # - question variants for each property type
    # - relationship templates and single templates
    # - element templates
    
    # Process each entity type from configuration
    for entity in entities:
        entity_type = entity["entity_type"]
        data_file = entity["data_file"]
        templates = entity["templates"]
        prompt_name = entity["prompt_name"]
        
        print(f"Processing entity type: {entity_type}")
        
        # Load data
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Skipping entity {entity_type}.")
            continue
        df = pd.read_csv(data_file)
        
        # Generate prompts
        prompts = generate_prompts(df, templates)
        print(f"Generated {len(prompts)} prompts for entity type '{entity_type}'.")
        
        # Use configured values
        # batch_size, aggregation, and base_save_dir are now from configuration
        
        # Define number of layers (assuming model has 'n_layers' layers)
        # Alternatively, determine from the model
        num_layers = len(model.model.layers)
        print(f"Model has {num_layers} layers.")

        try:
            process_and_save_activations_all_layers(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                num_layers=num_layers,
                entity_type=entity_type,
                prompt_name=prompt_name,
                model_name=model_name,
                aggregation=aggregation,
                save_dir=base_save_dir,
                batch_size=batch_size
            )
            print(f"Processed and saved all {num_layers} layers for entity type '{entity_type}'.")

        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory: {e}. Reducing batch size or freeing up memory.")
            torch.cuda.empty_cache()
            continue
    
    print("All activations processed and saved.")

if __name__ == "__main__":
    main()