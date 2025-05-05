#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import json
from datetime import datetime

import numpy as np
import whisper
from llama_cpp import Llama
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files

# ANSI color codes for a subtle, gruvbox-inspired palette
RESET = "\033[0m"
BOLD = "\033[1m"

# Base colors (not bold by default) - Gruvbox-inspired
RED = "\033[38;5;167m"        # Soft red/rust
GREEN = "\033[38;5;142m"      # Olive green
YELLOW = "\033[38;5;214m"     # Muted amber
BLUE = "\033[38;5;109m"       # Dusty blue
CYAN = "\033[38;5;108m"       # Teal/cyan (replacing AQUA)
GRAY = "\033[38;5;246m"       # Medium gray

# Bold combinations (used sparingly)
BOLD_GREEN = f"{BOLD}{GREEN}"
BOLD_CYAN = f"{BOLD}{CYAN}"

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
    "Qwen2.5-0.5B-Instruct": {
        "repo_id": "TheBloke/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct.Q4_K_M.gguf"
    },
    "Qwen2.5-7B-Instruct": {
        "repo_id": "TheBloke/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct.Q4_K_M.gguf"
    },
    "Qwen2.5-14B-Instruct": {
        "repo_id": "TheBloke/Qwen2.5-14B-Instruct-GGUF",
        "filename": "qwen2.5-14b-instruct.Q4_K_M.gguf"
    },
    "DeepSeek-Coder-6.7B": {
        "repo_id": "TheBloke/deepseek-coder-6.7B-instruct-GGUF",
        "filename": "deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
    },
    "Beyonder-4x7b": {
        "repo_id": "TheBloke/Beyonder-4x7B-v2-GGUF",
        "filename": "beyonder-4x7b-v2.Q4_K_M.gguf"
    },
    "laser-dolphin-mixtral-2x7B": {
        "repo_id": "TheBloke/laser-dolphin-mixtral-2x7b-dpo-GGUF",
        "filename": "laser-dolphin-mixtral-2x7b-dpo.Q4_K_M.gguf"
    }
}

# Set the default model
DEFAULT_MODEL = "Mistral-7B"

# Define default values for various parameters
DEFAULT_TEMPERATURE = 0.2
DEFAULT_REPEAT_PENALTY = 1.3
DEFAULT_BASE_CTX = 8192  # Base context size (x)

def show_welcome_message():
    """Display a welcome message when the script starts."""
    print("\n┌──────────────────────────────────────────┐")
    print(f"│  {BOLD_CYAN}Audio Transcription & Summarization{RESET}  │")
    print("└──────────────────────────────────────────┘")
    print("Use --help-full for comprehensive documentation\n")

def show_full_help():
    """Display comprehensive documentation for the script."""
    help_text = f"""
{BOLD}AUDIO SUMMARIZER - COMPREHENSIVE DOCUMENTATION{RESET}

{BOLD}DESCRIPTION{RESET}
    This tool transcribes audio files using Whisper and generates summaries using 
    local LLaMA models. It supports interactive chat capabilities and a range of 
    configuration options.

{BOLD}BASIC USAGE{RESET}
    {GREEN}audio_summarizer.py{RESET} {YELLOW}<audio_file>{RESET} [OPTIONS]
    {GREEN}audio_summarizer.py{RESET} {YELLOW}--transcript-input <file>{RESET} [OPTIONS]

{BOLD}AUDIO INPUT OPTIONS{RESET}
    {YELLOW}audio_path{RESET}                  Path to audio file to process
    {CYAN}--whisper-model{RESET} MODEL       Whisper model size [tiny, base, small, medium, large]
    {CYAN}--use-existing{RESET}              Use existing transcription without prompting
    {CYAN}--transcript-file{RESET} FILE      Custom path for saving transcription
    {CYAN}--transcribe-only{RESET}           Only perform transcription (skip summarization)
    {CYAN}--transcript-input{RESET} FILE     Use existing transcript file (skip transcription)

{BOLD}LLM MODEL OPTIONS{RESET}
    {CYAN}--llm-model{RESET} MODEL           Model for summarization: 
                              {GRAY}Llama-3.2-3B-Instruct, Mistral-7B, etc.{RESET}
    {CYAN}--llama-model{RESET} FILE          Path to local GGUF model file
    {CYAN}--temperature{RESET} VALUE         Temperature for generation (0.0-1.0)
    {CYAN}--repeat-penalty{RESET} VALUE      Repeat penalty for generation (1.0+)
    {CYAN}--context-size{RESET} SIZE         Context window size (see details below)
    {CYAN}--n-gpu-layers{RESET} VALUE        GPU layers (-1=all, 0=CPU only)

{BOLD}OUTPUT OPTIONS{RESET}
    {CYAN}--output{RESET} FILE               Save output to file instead of console
    {CYAN}--verbose{RESET}                   Show detailed initialization messages
    {CYAN}--help-full{RESET}                 Show this comprehensive documentation

{BOLD}CONTEXT SIZE OPTIONS{RESET}
    The context size can be specified in several ways:
    - {YELLOW}Direct number{RESET}:  --context-size 8192
    - {YELLOW}Base size{RESET}:      --context-size x      (8192 tokens)
    - {YELLOW}Multipliers{RESET}:    --context-size x*2    (16384 tokens)
                      --context-size x*4    (32768 tokens)
                      --context-size x*8    (65536 tokens)
    - {YELLOW}Divisions{RESET}:      --context-size x/2    (4096 tokens)
                      --context-size x/4    (2048 tokens)
                      --context-size x/8    (1024 tokens)

{BOLD}INTERACTIVE CHAT COMMANDS{RESET}
    Once in chat mode, you can use:
    {CYAN}/exit{RESET}, {CYAN}/quit{RESET}, {CYAN}/x{RESET}             Exit the chat session
    {CYAN}/--rm_context{RESET}                Clear conversation history
    {CYAN}/--llm-model=MODEL_NAME{RESET}      Switch to a different model

{BOLD}AVAILABLE MODELS{RESET}
    {', '.join(MODEL_REPOSITORIES.keys())}
"""
    print(help_text)

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
        "audio_path", type=str, nargs="?", default=None,
        help="Path to the audio file to process (optional if using --transcript-input)"
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
    # Add flag to use existing transcription
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing transcription file instead of transcribing again"
    )
    # NEW: Add option to specify transcript file path
    parser.add_argument(
        "--transcript-file",
        type=str,
        default=None,
        help="Path to save/read transcription file (overrides default transient_transcription_*.txt)"
    )
    # NEW: Add option to only do transcription
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only perform transcription without summarization"
    )
    # NEW: Add option to use an existing transcript file as input
    parser.add_argument(
        "--transcript-input",
        type=str,
        default=None,
        help="Path to an existing transcript file to use instead of transcribing audio"
    )
    # Add GPU layers control parameter
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all, 0 = CPU only)"
    )
    # Add help-full option
    parser.add_argument(
        "--help-full",
        action="store_true",
        help="Show comprehensive documentation"
    )
    # Add verbose flag (quiet is default)
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed initialization and diagnostic messages (default: quiet)"
    )
    # Replace the --free-chat-only argument with --free-chat
    parser.add_argument(
        "--free-chat",
        action="store_true",
        help="Skip transcription and summarization; start directly in chat mode"
    )
    # Add the list-models flag
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models without starting transcription or chat"
    )
    
    args = parser.parse_args()
    
    # Show full help and exit if requested
    if args.help_full:
        show_full_help()
        sys.exit(0)
        
    return args


def check_file_exists(file_path: str) -> None:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

def check_for_existing_transcription(audio_path: str, force_use_existing: bool = False, transcript_file: Optional[str] = None, 
                                    print_confirmation: bool = True) -> Tuple[bool, str]:
    """Check if a transcription file exists for the given audio file."""
    # Get audio file name without extension
    audio_filename = os.path.basename(audio_path)
    audio_name = os.path.splitext(audio_filename)[0]
    
    # Construct transcription file path or use provided one
    if transcript_file:
        transcription_file = transcript_file
    else:
        transcription_file = f"transient_transcription_{audio_name}.txt"
    
    # Check if the file exists
    if os.path.exists(transcription_file):
        if force_use_existing:
            # Only print if requested
            if print_confirmation:
                print(f"{GREEN}Using existing transcription: {BOLD}{transcription_file}{RESET}")
            return True, transcription_file
        
        # Prompt user
        print(f"{YELLOW}Found existing transcription for {audio_filename}.{RESET}")
        choice = input("Use existing transcription? (y/n/x to cancel): ").strip().lower()
        
        if choice == 'y':
            # Don't print confirmation here - let the caller handle it
            return True, transcription_file
        elif choice == 'x':
            print("Operation cancelled.")
            sys.exit(0)
        else:  # 'n' or anything else
            print(f"Will create a new transcription (existing file will be overwritten).")
            return False, transcription_file
    else:
        # No existing transcription
        return False, transcription_file

def read_transcription(transcription_file: str) -> str:
    """Read transcription from file."""
    with open(transcription_file, "r", encoding="utf-8") as f:
        return f.read()

def load_whisper_model(model_name: str, force_reload: bool = False) -> whisper.Whisper:
    """Load a Whisper model with caching support."""
    # Check cache first if not forcing reload
    cache_info = None
    if not force_reload:
        cache_info = get_model_cache("whisper")
        
    if cache_info and not force_reload:
        cached_name = cache_info.get("name")
        
        # Only use cache if the requested model matches cached model
        if cached_name == model_name:
            print(f"{GREEN}Using cached Whisper model: {BOLD}{cached_name}{RESET}")
            print(f"{GRAY}(Run with /--rm-whisper to force reload next time){RESET}")
            # Whisper handles caching internally, we just need to skip the loading message
            model = whisper.load_model(model_name)
            return model
            
    # Status message based on cache
    if force_reload:
        print(f"Reloading Whisper {model_name} model (forced reload)...")
    elif cache_info:
        print(f"Loading new Whisper {model_name} model (different from cached {cache_info.get('name')})...")
    else:
        print(f"Loading Whisper {model_name} model (first-time load)...")
    
    start_time = time.time()
    
    # Create a visible progress bar
    with tqdm(total=100, desc="Loading Whisper model", unit="%", ncols=80, colour="green") as pbar:
        # Show incremental progress before loading to make bar visible
        for i in range(5):
            time.sleep(0.1)
            pbar.update(10)
            
        # Load the model (this takes time but doesn't update the bar)
        model = whisper.load_model(model_name)
        
        # Complete the progress bar
        pbar.update(50)  # Update to 100%
    
    elapsed = time.time() - start_time
    print(f"{GREEN}Whisper model loaded in {BOLD}{elapsed:.2f} seconds{RESET}")
    
    # Save to cache
    save_model_cache("whisper", "whisper_internal_cache", model_name)
    
    return model

def load_llama_model(model_path: Optional[str] = None, model_name: str = DEFAULT_MODEL, 
                    context_size: int = DEFAULT_BASE_CTX * 2, n_gpu_layers: int = -1,
                    verbose: bool = False, force_reload: bool = False) -> Llama:
    """Load a LLaMA model with caching support."""
    # Check cache first if not forcing reload
    cache_info = None
    cached_path = None
    if not force_reload and not model_path:
        cache_info = get_model_cache("llm")
        
    if cache_info and not force_reload and not model_path:
        cached_path = cache_info.get("path")
        cached_name = cache_info.get("name")
        
        # Only use cache if the requested model matches cached model
        if cached_path and os.path.exists(cached_path) and cached_name == model_name:
            print(f"{GREEN}Using cached LLM model: {BOLD_CYAN}{cached_name}{RESET}")
            print(f"{GRAY}(Run with /--rm-llm to force reload next time){RESET}")
            model_path = cached_path

    # Status message based on cache state
    if model_path and cached_path == model_path:
        print(f"Loading cached LLaMA model...")
    elif force_reload:
        print(f"Reloading LLaMA model (forced reload)...")
    elif model_path:
        print(f"Loading custom LLaMA model from: {model_path}")
    elif cache_info:
        print(f"Loading new LLaMA model (different from cached {cache_info.get('name')})...")
    else:
        print(f"Loading LLaMA model (first-time load)...")
        
    print(f"Context window: {context_size} tokens | GPU acceleration: {'Enabled' if n_gpu_layers != 0 else 'Disabled'}")
    start_time = time.time()
    
    # Collapsed model parameters into one statement
    model_params = {"n_batch": 512, "n_gpu_layers": n_gpu_layers, "n_threads": 4, "verbose": verbose, "n_ctx": context_size}
    
    # Redirect stderr to suppress initialization messages if not verbose
    original_stderr = None
    if not verbose:
        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    
    try:
        # Create a progress indicator for downloading/loading
        pbar = tqdm(total=100, desc="Loading model", unit="%")
        
        # Load model from local file or download from HuggingFace
        if model_path:
            # Use local model file
            print(f"{GRAY}Loading from: {model_path}{RESET}")
            pbar.update(20)  # Show some progress
            model = Llama(model_path=model_path, **model_params)
            pbar.update(80)  # Complete the progress
            
        else:
            # Download from HuggingFace if needed
            if model_name not in MODEL_REPOSITORIES:
                pbar.close()
                available_models = ", ".join(MODEL_REPOSITORIES.keys())
                raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")

            repo_info = MODEL_REPOSITORIES[model_name]
            repo_id = repo_info["repo_id"]
            preferred_filename = repo_info["filename"]
            
            print(f"{GRAY}Selected: {BOLD_CYAN}{model_name}{RESET} from {repo_id}{RESET}")
            
            # Create model directory if it doesn't exist
            model_dir = os.path.join(os.path.expanduser("~"), ".cache", "llama_models")
            os.makedirs(model_dir, exist_ok=True)
            
            # See if we already have this model cached locally
            local_file = os.path.join(model_dir, preferred_filename)
            pbar.update(10)  # Initial progress
            
            if not os.path.exists(local_file):
                pbar.set_description(f"Downloading {preferred_filename}")
                
                # Download model file from HuggingFace
                local_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=preferred_filename,
                    cache_dir=model_dir
                )
                pbar.update(40)  # Download progress
            else:
                print(f"{GRAY}Using cached model{RESET}")
                pbar.update(40)  # Skip download progress
                
            pbar.set_description("Initializing model")
            # Load the model
            model = Llama(model_path=local_file, **model_params)
            pbar.update(50)  # Loading complete
        
        pbar.close()
        elapsed = time.time() - start_time
        print(f"{GREEN}Model loaded in {elapsed:.2f} seconds{RESET}")
        
        # Restore stderr before returning
        if not verbose and original_stderr:
            sys.stderr.close()
            sys.stderr = original_stderr
            
        # After successful load, save to cache if not using custom path
        if not model_path and 'local_file' in locals():
            save_model_cache("llm", local_file, model_name)
        elif model_path:
            save_model_cache("llm", model_path, model_name)
            
        return model
        
    except Exception as e:
        # Ensure stderr is restored even on error
        if not verbose and original_stderr:
            sys.stderr.close()
            sys.stderr = original_stderr
        
        if 'pbar' in locals():
            pbar.close()
            
        elapsed = time.time() - start_time
        print(f"{RED}Failed to load model: {e}{RESET}")
        raise

def transcribe_audio(model: whisper.Whisper, audio_path: str, transcription_file: Optional[str] = None) -> Tuple[str, str]:
    print(f"{BOLD}Transcribing audio: {audio_path}{RESET}")
    start_time = time.time()
    
    # Get audio file name and create transcription filename if not provided
    if not transcription_file:
        audio_filename = os.path.basename(audio_path)
        audio_name = os.path.splitext(audio_filename)[0]
        transcription_file = f"transient_transcription_{audio_name}.txt"
    
    # Perform transcription
    print("Transcribing... This may take a few minutes...")
    result = model.transcribe(
        audio_path,
        fp16=False  # Use fp32 for better accuracy and CPU compatibility
    )
    
    # Save transcription to file
    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    print(f"Transcription saved to: {transcription_file}")
    
    elapsed = time.time() - start_time
    print(f"{GREEN}Transcription completed in {BOLD}{elapsed:.2f} seconds{RESET}")
    return result["text"], transcription_file

def summarize_text(model: Llama, text: str, temperature: float = 0.7, repeat_penalty: float = 1.3) -> str:
    print(f"Generating summary...")
    print(f"Using temperature: {BOLD}{temperature}{RESET}, repeat penalty: {BOLD}{repeat_penalty}{RESET}")
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
    print(f"{GREEN}Summary generated in {BOLD}{elapsed:.2f} seconds{RESET}")
    return response["choices"][0]["text"].strip()

def save_output(transcription: str, summary: str, output_path: str) -> None:
    """Save the transcription and summary to a file."""
    with open(output_path, "w", encoding="utf-8") as f:
        # Don't use color codes in file output
        f.write("TRANSCRIPTION:\n")
        f.write("-" * 80 + "\n")
        f.write(transcription)
        f.write("\n\n")
        f.write("SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(summary)
    print(f"{GREEN}Output saved to: {BOLD}{output_path}{RESET}")

# Add this at the top level of your script, near your other global variables
args = None  # Will be set in main()

def interactive_chat(model: Llama, transcription: str, summary: str, temperature: float = 0.7, repeat_penalty: float = 1.3, 
                     context_size: int = DEFAULT_BASE_CTX * 2, args_param=None, force_free_chat: bool = False) -> None:
    """Start an interactive chat session with the LLM about the transcription and summary."""
    # Use passed args parameter instead of global
    global args
    if args_param:
        args = args_param
        
    print(f"\nStarting interactive chat session.")
    print(f"{GRAY}Commands:{RESET}")
    print(f"  /exit, /quit, /x - End the chat session")
    print(f"  /--rm_context - Clear conversation history")
    print(f"  /--rm-llm - Remove cached LLM model (will force download next time)")
    print(f"  /--rm-whisper - Remove cached Whisper model (will force download next time)")
    
    # Only show these commands if we have transcription data
    if transcription:
        print(f"  /--free_chat - Switch to unrestricted conversation mode")
        print(f"  /--focus_transcript - Return to transcript discussion mode")
    
    print(f"  /--llm-model=MODEL_NAME - Switch to a different model")
    
    # Modified system prompts with chat formatting
    transcript_prompt = f"""System: You are a concise assistant discussing a transcription and its summary. Your responses should be direct and to the point, without repeating the user's questions. Always focus on providing new information.

Context:
Transcription: {transcription}
Summary: {summary}

User: Hi, I'd like to discuss this content.
Assistant: I'm ready to discuss the content. What would you like to know?"""

    general_prompt = """System: You are a helpful but concise assistant. Your responses should be direct and informative, without repeating the user's questions. Focus on providing valuable information efficiently.

User: Hi, let's chat.
Assistant: Hello! What would you like to discuss?"""

    # Initialize chat history with appropriate context
    if force_free_chat or not transcription:
        chat_history = general_prompt
        is_transcript_mode = False
        print(f"{CYAN}In unrestricted conversation mode.{RESET}")
    else:
        chat_history = transcript_prompt
        is_transcript_mode = True
        print("\nYou can ask questions about the content, request alternative summary styles, or discuss details.")

    current_model = model
    while True:
        user_input = input(f"\n{BOLD_GREEN}> {RESET}").strip()
        
        # Check if user wants to exit
        if user_input.lower() in ['/exit', '/quit', '/x']:
            print(f"{CYAN}Ending chat session.{RESET}")
            break
        
        # Check if user wants to clear context
        if user_input.lower() == '/--rm_context':
            # Keep the current mode when clearing context
            chat_history = transcript_prompt if is_transcript_mode else general_prompt
            print(f"{CYAN}Context cleared. Starting fresh conversation.{RESET}")
            continue
        
        # Check if user wants to switch to free chat mode
        if user_input.lower() == '/--free_chat':
            chat_history = general_prompt
            is_transcript_mode = False
            print(f"{CYAN}Switching to unrestricted conversation mode.{RESET}")
            continue
            
        # Check if user wants to switch back to transcript focus
        if user_input.lower() == '/--focus_transcript':
            chat_history = transcript_prompt
            is_transcript_mode = True
            print(f"{CYAN}Returning to transcript discussion mode.{RESET}")
            continue
        
        # Check if user wants to switch models
        if user_input.lower().startswith('/--llm-model='):
            try:
                # Extract model name
                new_model_name = user_input.split('=', 1)[1].strip()
                
                if new_model_name not in MODEL_REPOSITORIES:
                    print(f"{YELLOW}Error: Model '{new_model_name}' not found. Available models: {', '.join(MODEL_REPOSITORIES.keys())}{RESET}")
                    continue
                
                print(f"{CYAN}Switching to model: {BOLD}{new_model_name}{RESET}")
                new_model = load_llama_model(
                    model_name=new_model_name, 
                    context_size=context_size,
                    n_gpu_layers=args.n_gpu_layers,  # Will need to be passed to function
                    verbose=args.verbose,  # Will need to be passed to function
                    force_reload=False
                )
                current_model = new_model
                print(f"{GREEN}Model switched successfully to {BOLD}{new_model_name}{RESET}")
                continue
            except Exception as e:
                print(f"{YELLOW}Error switching model: {e}{RESET}")
                continue
        
        if not user_input:
            continue
        
        # Append user question to chat history
        current_exchange = f"\nUser: {user_input}\nAssistant: "
        truncated_history = chat_history[-context_size:] if len(chat_history) > context_size else chat_history
        
        # Generate response with modified prompt
        print(f"{GRAY}Generating response...{RESET}")
        response = current_model.create_completion(
            truncated_history + current_exchange,
            max_tokens=1024,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=repeat_penalty,
            stop=["User:", "\nUser"]  # Prevent model from generating additional user messages
        )
        
        # Extract and print response
        assistant_response = response["choices"][0]["text"].strip()
        print(f"{assistant_response}")
        
        # Update chat history more efficiently
        chat_history += current_exchange + assistant_response

def main() -> None:
    """Main function to run the audio summarization pipeline."""
    try:
        global args
        
        show_welcome_message()
        args = parse_arguments()
        
        # Handle --free-chat mode first
        if args.free_chat:
            print(f"{CYAN}Starting in free chat mode{RESET}")
            
            # Parse context size
            context_size = parse_context_size(args.context_size)
            
            # Load LLM model
            llama_model = load_llama_model(
                args.llama_model,
                args.llm_model,
                context_size=context_size,
                n_gpu_layers=args.n_gpu_layers,
                verbose=args.verbose,
                force_reload=False
            )
            
            # Start interactive chat with empty context
            interactive_chat(
                llama_model,
                "",  # Empty transcription
                "",  # Empty summary
                temperature=args.temperature,
                repeat_penalty=args.repeat_penalty,
                context_size=context_size,
                args_param=args,
                force_free_chat=True
            )
            
            print(f"\n{GREEN}Chat session completed!{RESET}")
            return
            
        # Only check for audio/transcript input if not in free chat mode
        if args.audio_path is None and args.transcript_input is None:
            raise ValueError("Either audio_path or --transcript-input must be provided")
            
        # Parse context size expression
        context_size = parse_context_size(args.context_size)
        
        # Check if user just wants to list models
        if args.list_models:
            list_available_models()
            sys.exit(0)
        
        # Handle direct transcript input case
        if args.transcript_input:
            if not os.path.isfile(args.transcript_input):
                raise FileNotFoundError(f"Transcript file not found: {args.transcript_input}")
                
            print(f"{CYAN}Using provided transcript file: {BOLD}{args.transcript_input}{RESET}")
            transcription = read_transcription(args.transcript_input)
            transcription_file = args.transcript_input

        # Handle audio transcription case
        else:
            # Check if the audio file exists
            check_file_exists(args.audio_path)
            
            # Set transcript file path if provided
            transcript_file = args.transcript_file
            
            # Check for existing transcription
            use_existing, transcription_file = check_for_existing_transcription(
                args.audio_path, args.use_existing, transcript_file
            )
            
            # Get transcription (either from file or by transcribing)
            if use_existing:
                transcription = read_transcription(transcription_file)
                print(f"{CYAN}Using existing transcription from: {BOLD}{transcription_file}{RESET}")
            else:
                # Load Whisper model only if we need to transcribe
                whisper_model = load_whisper_model(args.whisper_model, force_reload=False)
                transcription, transcription_file = transcribe_audio(whisper_model, args.audio_path, transcript_file)
        
        # Stop here if transcribe-only mode
        if args.transcribe_only:
            print(f"\n{GREEN}Transcription completed and saved to: {BOLD}{transcription_file}{RESET}")
            if not args.output:
                print(f"\nTRANSCRIPTION:")
                print(f"{'-' * 80}")
                print(transcription)
            elif args.output != transcription_file:  # Don't write twice to same file
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(transcription)
                print(f"{GREEN}Transcription saved to output file: {BOLD}{args.output}{RESET}")
            print(f"\n{GREEN}Operation completed successfully!{RESET}")
            return
            
        # Load LLM model for summarization
      #  global args  # Make args available to interactive_chat
        llama_model = load_llama_model(
            args.llama_model,
            args.llm_model,
            context_size=context_size,
            n_gpu_layers=args.n_gpu_layers,
            verbose=args.verbose,
            force_reload=False
        )
        
        # Summarize transcription
        summary = summarize_text(
            llama_model, 
            transcription,
            temperature=args.temperature,
            repeat_penalty=args.repeat_penalty
        )
        
        # Output results
        if args.output:
            save_output(transcription, summary, args.output)
        else:
            print(f"\nTRANSCRIPTION:")
            print(f"{'-' * 80}")
            print(transcription)
            print(f"\nSUMMARY:")
            print(f"{'-' * 80}")
            print(summary)
        
        # Ask if the user wants to chat about the output
        print(f"\n{CYAN}Would you like to chat with the model about the transcription and summary?{RESET}")
        choice = input("(y/n/x to exit): ").strip().lower()
        
        if choice == 'y':
            interactive_chat(
                llama_model,
                transcription,
                summary,
                temperature=args.temperature,
                repeat_penalty=args.repeat_penalty,
                context_size=context_size,
                args_param=args  # Pass args as a parameter
            )
        elif choice == 'x':
            print("Exiting program.")
            sys.exit(0)
            
        print(f"\n{GREEN}Audio summarization completed successfully!{RESET}")
        
    except FileNotFoundError as e:
        print(f"{YELLOW}Error: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"{YELLOW}Error: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{YELLOW}Error: An unexpected error occurred: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

def get_cache_dir():
    """Get the cache directory for the script."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "audio_summarizer")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def save_model_cache(model_type, model_path, model_name=None):
    """Save model path to cache file."""
    cache_dir = get_cache_dir()
    cache_file = os.path.join(cache_dir, f"{model_type}_cache.json")
    
    # Create or update cache entry
    cache_data = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
    
    cache_data["path"] = model_path
    cache_data["name"] = model_name
    cache_data["updated"] = time.time()
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

def get_model_cache(model_type):
    """Get cached model path if available."""
    cache_dir = get_cache_dir()
    cache_file = os.path.join(cache_dir, f"{model_type}_cache.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        return cache_data
    
    return None

def remove_model_cache(model_type):
    """Remove cached model information."""
    cache_dir = get_cache_dir()
    cache_file = os.path.join(cache_dir, f"{model_type}_cache.json")
    
    if os.path.exists(cache_file):
        os.remove(cache_file)
        return True
    
    return False

def list_available_models():
    """Display a formatted list of all available models."""
    print(f"\n{BOLD}AVAILABLE LLM MODELS{RESET}")
    print("-" * 80)
    print(f"{'Model Name':<30} {'Repository':<50} {'Filename':<30}")
    print("-" * 80)
    
    for model_name, model_info in sorted(MODEL_REPOSITORIES.items()):
        print(f"{model_name:<30} {model_info['repo_id']:<50} {model_info['filename']:<30}")
    
    print("\n" + "-" * 80)
    print(f"\n{BOLD}AVAILABLE WHISPER MODELS{RESET}")
    print("-" * 80)
    print(f"{'Model Name':<15} {'Description':<65}")
    print("-" * 80)
    
    whisper_models = [
        ("tiny", "~39M parameters, English-only, fastest"),
        ("base", "~74M parameters, multilingual, fast"),
        ("small", "~244M parameters, multilingual, good balance of speed/accuracy"),
        ("medium", "~769M parameters, multilingual, more accurate but slower"),
        ("large", "~1.5B parameters, multilingual, most accurate but slowest")
    ]
    
    for model_name, description in whisper_models:
        print(f"{model_name:<15} {description:<65}")
    
    print("\n" + "-" * 80)

if __name__ == "__main__":
    main()
