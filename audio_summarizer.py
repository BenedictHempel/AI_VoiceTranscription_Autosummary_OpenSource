#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import whisper
from llama_cpp import Llama
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files

# Define available LLM models with correct filenames
MODEL_REPOSITORIES = {
    "Llama-3.2-3B-Instruct": {
        "repo_id": "QuantFactory/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct.Q4_K_M.gguf"
    },
    "Llama-3.2-3B-Instruct-Alt": {
        "repo_id": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct.Q4_K_M.gguf"
    },
    "Llama-3.2-1B-Instruct": {
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct.Q4_K_M.gguf"
    },
    "Llama-2-7B": {
        "repo_id": "TheBloke/Llama-2-7B-GGUF",
        "filename": "llama-2-7b.Q4_K_M.gguf"
    },
    "Mistral-7B": {
        "repo_id": "TheBloke/Mistral-7B-v0.1-GGUF",
        "filename": "mistral-7b-v0.1.Q4_K_M.gguf"
    },
    "Starling-7B": {
        "repo_id": "TheBloke/Starling-LM-7B-alpha-GGUF",
        "filename": "starling-lm-7b-alpha.Q4_K_M.gguf"
    },
    "Rogue-Creative-Uncensored": {
        "repo_id": "DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF",
        "filename": "L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q8_0.gguf"
    },
    "NexoNimbus": {
        "repo_id": "TheBloke/NexoNimbus-7B-GGUF",
        "filename": "nexonimbus-7b.Q4_K_M.gguf"
    },
    # "Llama-3-70B": {
    #     "repo_id": "meta-llama/Meta-Llama-3-70B",
    #     "filename": "meta-llama-3-70b.Q4_K_M.gguf"
    # },
    "Baichuan-2-13B": {
        "repo_id": "ridwanlekan/Baichuan2-13B-Base-Q4_K_M-GGUF",
        "filename": "baichuan2-13b-base.Q4_K_M.gguf"
    },
    "Mixtral-8x7B": {
        "repo_id": "TheBloke/Mixtral-8x7B-v0.1-GGUF",
        "filename": "mixtral-8x7b-v0.1.Q4_K_M.gguf"
    },
    "Falcon-Mamba-7B": {
        "repo_id": "tiiuae/falcon-mamba-7b-Q4_K_M-GGUF",
        "filename": "falcon-mamba-7B-Q4_K_M.gguf"
    },
    "Dr_Samantha-7B": {
        "repo_id": "TheBloke/Dr_Samantha-7B-GGUF",
        "filename": "dr_samantha-7b.Q4_K_M.gguf"
    },
    "Beyonder-4x7b": {
        "repo_id": "TheBloke/Beyonder-4x7B-v2-GGUF",
        "filename": "beyonder-4x7b-v2.Q4_K_M.gguf"
    },
    "laser-dolphin-mixtral-2x7B": {
        "repo_id": "laser-dolphin-mixtral-2x7b-dpo.Q4_K_M.gguf",
        "filename": "laser-dolphin-mixtral-2x7b-dpo.Q4_K_M.gguf"
    }
    #TheBloke/Dr_Samantha-7B-GGUF 
    #beyonder-4x7b-v2.Q4_K_M.gguf
    #laser-dolphin-mixtral-2x7b-dpo.Q4_K_M.gguf
}

# Set the default model
DEFAULT_MODEL = "Mistral-7B"

# Define default values for various parameters
DEFAULT_TEMPERATURE = 0.2
DEFAULT_REPEAT_PENALTY = 1.3
DEFAULT_BASE_CTX = 8192  # Base context size (x)


def parse_context_size(value: str) -> int:
    """Parse context size expressions like '8192', 'x', 'x*2', 'x/4', etc."""
    value = str(value).lower().strip()
    base_ctx = DEFAULT_BASE_CTX
    
    # If it's a plain number
    if value.isdigit():
        return int(value)
    
    # Handle just "x" by itself (base context size)
    if value == "x":
        return base_ctx
    
    # Handle multiplication (x*2, x*4, x*8)
    if value.startswith('x*'):
        try:
            multiplier = int(value[2:])
            if multiplier in [2, 4, 8]:
                return base_ctx * multiplier
            else:
                print(f"Multiplier must be 2, 4, or 8, got {multiplier}. Using base context size.")
                return base_ctx
        except ValueError:
            print(f"Invalid multiplier format. Using base context size.")
            return base_ctx
    
    # Handle division (x/2, x/4, x/8)
    if value.startswith('x/'):
        try:
            divisor = int(value[2:])
            if divisor in [2, 4, 8]:
                return base_ctx // divisor
            else:
                print(f"Divisor must be 2, 4, or 8, got {divisor}. Using base context size.")
                return base_ctx
        except ValueError:
            print(f"Invalid divisor format. Using base context size.")
            return base_ctx
    
    # Invalid format - return base context size
    print(f"Invalid context size format: {value}. Using base context size.")
    return base_ctx


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe and summarize audio files")
    parser.add_argument(
        "audio_path", type=str, help="Path to the audio file to process"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to use for transcription (default: small)",
    )
    # Add model selection option
    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODEL_REPOSITORIES.keys()),
        help=f"LLM model to use for summarization (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--llama-model",
        type=str,
        default=None,
        help="Path to local LLaMA model file (overrides --llm-model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the transcription and summary (default: print to console)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for text generation (0.0-1.0, lower is more deterministic, default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=DEFAULT_REPEAT_PENALTY,
        help=f"Repeat penalty (1.0+ higher values penalize repetition more, default: {DEFAULT_REPEAT_PENALTY})",
    )
    # Add context size parameter
    parser.add_argument(
        "--context-size",
        type=str,
        default="x*2",  # Default to 2x the base size
        help="Context window size: 'x' for base size (8192), 'x*2'/'x*4'/'x*8' for larger sizes, "
             "'x/2'/'x/4'/'x/8' for smaller sizes, or a direct number"
    )
    return parser.parse_args()


def check_file_exists(file_path: str) -> None:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def load_whisper_model(model_name: str) -> whisper.Whisper:
    print(f"Loading Whisper {model_name} model...")
    start_time = time.time()
    
    # Create a placeholder progress bar
    pbar = tqdm(total=100, desc="Loading Whisper model", unit="%")
    
    # Load the model (we'll update the progress bar manually since whisper doesn't provide loading progress)
    model = whisper.load_model(model_name)
    
    # Update progress bar to completion
    pbar.update(100)
    pbar.close()
    
    elapsed = time.time() - start_time
    print(f"Whisper model loaded in {elapsed:.2f} seconds")
    return model


def load_llama_model(model_path: Optional[str] = None, model_name: str = DEFAULT_MODEL, context_size: int = DEFAULT_BASE_CTX * 2) -> Llama:
    """
    Load a LLaMA model either from a specified path or download from HuggingFace.
    
    Args:
        model_path: Path to a local model file, or None to download the default model
        model_name: Name of the model to load from MODEL_REPOSITORIES if model_path is None
        context_size: Size of context window in tokens
        
    Returns:
        Llama: The loaded LLaMA model
    """
    print("Loading LLaMA model...")
    print(f"Using context window size: {context_size} tokens")
    start_time = time.time()
    
    # Common model parameters with optimized settings
    model_params = {
        "n_batch": 512,
        "n_gpu_layers": -1,
        "n_threads": 4,
        "verbose": False,
        "n_ctx": context_size  # Use the provided context size
    }
    
    # Rest of the function remains the same, just remove the hardcoded n_ctx assignments
    try:
        if model_path is None:
            # Get the selected model configuration
            if model_name not in MODEL_REPOSITORIES:
                print(f"Warning: Unknown model '{model_name}', falling back to {DEFAULT_MODEL}")
                model_name = DEFAULT_MODEL
                
            model_config = MODEL_REPOSITORIES[model_name]
            repo_id = model_config["repo_id"]
            preferred_filename = model_config["filename"]
            
            print(f"Selected model: {model_name} from {repo_id}")
            print(f"Looking for file matching: {preferred_filename}")
            
            # Create local model directory if it doesn't exist
            os.makedirs('.models', exist_ok=True)
            
            # Get available files in the repo
            try:
                available_files = list_repo_files(repo_id)
                gguf_files = [f for f in available_files if f.endswith('.gguf')]
                
                if not gguf_files:
                    raise ValueError(f"No GGUF files found in repository {repo_id}")
                
                # Try to find exact match first
                filename = None
                if preferred_filename in available_files:
                    filename = preferred_filename
                else:
                    # Look for close match based on Q4_K_M or similar quantization
                    print(f"Exact file match not found, looking for alternatives...")
                    for file in gguf_files:
                        if "Q4_K_M" in file:
                            filename = file
                            print(f"Selected alternative file: {filename}")
                            break
                    
                    # If still no match, just use the first GGUF file
                    if not filename:
                        filename = gguf_files[0]
                        print(f"Using first available GGUF file: {filename}")
                
                # Download the model file
                print(f"Downloading model file: {filename}")
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                
                # Load from the downloaded file (no need to set n_ctx again)
                model = Llama(model_path=model_path, **model_params)
                
            except Exception as e:
                print(f"Error accessing repository: {e}")
                # Fall back to Rogue model which we know works
                print(f"Falling back to {DEFAULT_MODEL}")
                repo_id = MODEL_REPOSITORIES[DEFAULT_MODEL]["repo_id"]
                filename = MODEL_REPOSITORIES[DEFAULT_MODEL]["filename"]
                
                model = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=filename,
                    **model_params
                )
        else:
            # Load from local file path (no need to set n_ctx again)
            print(f"Loading model from local path: {model_path}")
            model = Llama(
                model_path=model_path,
                **model_params
            )
        
        elapsed = time.time() - start_time
        print(f"LLaMA model loaded in {elapsed:.2f} seconds")
        return model
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Failed to load LLaMA model after {elapsed:.2f} seconds: {e}")
        raise


def transcribe_audio(model: whisper.Whisper, audio_path: str) -> str:
    print(f"Transcribing audio: {audio_path}")
    start_time = time.time()
    
    # Perform transcription
    print("Transcribing... This may take a few minutes...")
    result = model.transcribe(
        audio_path,
        fp16=False  # Use fp32 for better accuracy and CPU compatibility
    )
    
    elapsed = time.time() - start_time
    print(f"Transcription completed in {elapsed:.2f} seconds")
    return result["text"]


def summarize_text(model: Llama, text: str, temperature: float = 0.7, repeat_penalty: float = 1.3) -> str:
    print("Generating summary...")
    print(f"Using temperature: {temperature}, repeat penalty: {repeat_penalty}")  # Note "repeat penalty" not "repetition penalty"
    start_time = time.time()
    
    # Prompt engineering for summarization
    prompt = f"""You are an assistant that summarizes transcribed audio content effectively. Below is a transcription of audio content. Please create a concise summary that captures the key points, main ideas, and important details. don´t add any sugestions or own ideas at the end. be consices and clear.

Transcription:
{text}

Summary:"""
    
    # Generate summary with simple completion - using correct parameter name
    response = model.create_completion(
        prompt,
        max_tokens=512,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=repeat_penalty  # Using the correct parameter name
    )
    
    elapsed = time.time() - start_time
    print(f"Summary generated in {elapsed:.2f} seconds")
    return response["choices"][0]["text"].strip()


def save_output(transcription: str, summary: str, output_path: str) -> None:
    """Save the transcription and summary to a file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("TRANSCRIPTION:\n")
        f.write("-" * 80 + "\n")
        f.write(transcription)
        f.write("\n\n")
        f.write("SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(summary)
    print(f"Output saved to: {output_path}")


def main() -> None:
    """Main function to run the audio summarization pipeline."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Parse context size expression
        context_size = parse_context_size(args.context_size)
        
        # Check if the audio file exists
        check_file_exists(args.audio_path)
        
        # Load models
        whisper_model = load_whisper_model(args.whisper_model)
        
        # If a local model path is provided, use that, otherwise use the selected model
        llama_model = load_llama_model(
            args.llama_model,
            args.llm_model,
            context_size=context_size
        )
        
        # Transcribe audio
        transcription = transcribe_audio(whisper_model, args.audio_path)
        
        # Summarize transcription
        summary = summarize_text(
            llama_model, 
            transcription,
            temperature=args.temperature,
            repeat_penalty=args.repeat_penalty  # Pass the repeat_penalty parameter
        )
        
        # Output results
        if args.output:
            save_output(transcription, summary, args.output)
        else:
            print("\nTRANSCRIPTION:")
            print("-" * 80)
            print(transcription)
            print("\nSUMMARY:")
            print("-" * 80)
            print(summary)
            
        print("\nAudio summarization completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
