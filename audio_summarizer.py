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
        help="Whisper model size to use for transcription (default: base)",
    )
    parser.add_argument(
        "--llama-model",
        type=str,
        default=None,
        help="Path to LLaMA model file (default: auto-download LLaMA 3.2 3B)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the transcription and summary (default: print to console)",
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


def load_llama_model(model_path: Optional[str] = None) -> Llama:
    """
    Load a LLaMA model either from a specified path or download from HuggingFace.
    
    Args:
        model_path: Path to a local model file, or None to download the default model
        
    Returns:
        Llama: The loaded LLaMA model
    """
    print("Loading LLaMA model...")
    start_time = time.time()
    
    # Common model parameters with optimized settings for balance between performance and memory
    model_params = {
        "n_batch": 512,  # Keep original batch size
        "n_gpu_layers": -1,  # Use all layers that fit on GPU with Metal
        "n_threads": 4,   # Control CPU thread usage for better performance
        "verbose": False
    }
    
    try:
        if model_path is None:
            # Default model configuration for download
            repo_id = "DavidAU/L3.2-Rogue-Creative-Instruct-7B-GGUF"
            filename = "L3.2-Rogue-Creative-Instruct-7B-D_AU-Q4_k_m.gguf"
            
            # Create local model directory if it doesn't exist
            os.makedirs('.models', exist_ok=True)
            
            # Load from Hugging Face repo with moderate context size to balance memory usage
            model_params["n_ctx"] = 8172*2
            model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                **model_params
            )
        else:
            # Load from local file path with moderate context window to balance memory usage
            model_params["n_ctx"] = 8192*2
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


def summarize_text(model: Llama, text: str) -> str:
    print("Generating summary...")
    start_time = time.time()
    
    # Prompt engineering for summarization
    prompt = f"""You are an assistant that summarizes transcribed audio content effectively. Below is a transcription of audio content. Please create a concise summary that captures the key points, main ideas, and important details. don´t add any sugestions or own ideas at the end. be consices and clear.

Transcription:
{text}

Summary:"""
    
    # Generate summary with simple completion
    response = model.create_completion(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9
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
        
        # Check if the audio file exists
        check_file_exists(args.audio_path)
        
        # Load models
        whisper_model = load_whisper_model(args.whisper_model)
        llama_model = load_llama_model(args.llama_model)
        
        # Transcribe audio
        transcription = transcribe_audio(whisper_model, args.audio_path)
        
        # Summarize transcription
        summary = summarize_text(llama_model, transcription)
        
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
