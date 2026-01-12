import os
import argparse
import gc
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def create_pruned_model(source_model_id, new_repo_id, layers_to_keep, username, token):
    print(f"Downloading and loading {source_model_id}...")
    print("NOTE: This uses disk offloading to save RAM. It may take a few minutes.")

    # 1. Load the model with memory optimizations
    # device_map="auto" + offload_folder allows loading models larger than RAM
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source_model_id,
            torch_dtype="auto",           # Use fp16 if available (50% less RAM)
            trust_remote_code=True,
            low_cpu_mem_usage=True,       # Don't load full model into RAM at once
            device_map="auto",            # Offload to disk if RAM is full
            offload_folder="offload_tmp"  # Temporary storage for heavy weights
        )
        tokenizer = AutoTokenizer.from_pretrained(source_model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Verify and Reduce Layers
    try:
        # Locate the layers based on architecture (Qwen/Llama vs GPT/Bloom)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            config_key = "num_hidden_layers"
            model_type = "llama_like"
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
            config_key = "n_layer"
            model_type = "gpt_like"
        else:
            print("Error: Could not locate layer list in model structure.")
            return

        current_layer_count = len(layers)
        print(f"Original layer count: {current_layer_count}")
        
        if layers_to_keep >= current_layer_count:
            print(f"Error: Requested {layers_to_keep} layers, but model has {current_layer_count}.")
            return

        print(f"✂️  Pruning model to first {layers_to_keep} layers...")
        
        # SLICE THE LAYERS
        # This effectively drops the references to the heavy later layers
        if model_type == "llama_like":
            model.model.layers = model.model.layers[:layers_to_keep]
        elif model_type == "gpt_like":
            model.transformer.h = model.transformer.h[:layers_to_keep]

        # UPDATE CONFIG
        setattr(model.config, config_key, layers_to_keep)
        
        # AGGRESSIVE CLEANUP
        # Force Python to release the memory of the dropped layers immediately
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        print(f"New layer count: {layers_to_keep}")

    except AttributeError as e:
        print(f"Error accessing model layers: {e}")
        return

    # 3. Publish to Hugging Face Hub
    full_repo_id = f"{username}/{new_repo_id}"
    print(f"Pushing to {full_repo_id}...")

    try:
        # We push only the remaining (small) layers
        model.push_to_hub(full_repo_id, token=token, private=False)
        tokenizer.push_to_hub(full_repo_id, token=token, private=False)
        
        print("\nSuccess! Your pruned model is live at:")
        print(f"https://huggingface.co/{full_repo_id}")
        
    except Exception as e:
        print(f"Error pushing to hub: {e}")
    finally:
        # Cleanup temporary offload folder
        if os.path.exists("offload_tmp"):
            shutil.rmtree("offload_tmp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune a Hugging Face model with low memory usage.")
    
    parser.add_argument("--source", type=str, required=True, help="Source model ID")
    parser.add_argument("--target", type=str, required=True, help="New model name")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers to keep")
    parser.add_argument("--username", type=str, required=True, help="Your HF username")
    
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set.")
    else:
        create_pruned_model(
            source_model_id=args.source,
            new_repo_id=args.target,
            layers_to_keep=args.layers,
            username=args.username,
            token=hf_token
        )