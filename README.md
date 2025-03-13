# AI Voice Transcription & Autosummary (Open Source)

A **small Python script** that utilizes **OpenAI Whisper** for speech-to-text transcription and an **open-source LLM** (like Llama 3.2 or Mistral) for summarizing audio files. Works offline, doesn´t require expensive hardware & prevents data leakages.
Autoloading of all whisper model sizes & autoloading of LLMs from HuggingFace repo.  
Adaptability: LLM tuneable by model, modelsize & quantization as well as by context window, temperature & repeatition penality to fit all resources and needs.

## Features
- 🎙 **Transcribe** spoken content from an audio file by running whisper locally.
- ✍️ **Summarize** the transcribed text using a local LLM.
- 🚀 **Supports multiple models** (Llama 3.x, Mistral, Starling etc.).
- ⚡ **This script is optimized for Apple Silicon running by leveraging metal acceleration.**
---

## 🛠️ Installation
### **1. Clone the Repository**
```sh
git clone https://github.com/YOUR-USERNAME/AI_VoiceTranscription_Autosummary_OpenSource.git
cd AI_VoiceTranscription_Autosummary_OpenSource
```

### **2. Set Up a Virtual Environment**
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### **3. Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## 🚀 Usage
Run the script with:
```sh
#python3 audio_summarizer.py --whisper-model small --llama-model Llama_3.2_7B my_audio.mp3
python3 audio_summarizer.py my_audio.mp3
```

### **Available Options & Flags**
⚠️ **Note:** Model selection via `--llama-model` is not fully implemented yet. The script currently defaults to the recommended instruct model.  
⚠️ **Note:** Temperature & repeat_penality adjustment using `--temperature` and `--repeat-penalty` is not fully implemented yet. Use the variables TEMPERATURE and RETITION_PENALTY at the top of the script.
| Argument | Description |
|----------|-------------|
| `my_audio.mp3` | Path to the input audio file |
| `--whisper-model` | Select Whisper model (tiny, base, small, medium, large) |
| `--llama-model` | Choose LLM (Llama_3.2_7BINS, Llama_3.2_7B, Llama_3.2_11B, Llama_3.2_13B, Mistral_7B, Mixtral_8x7B, Starling_7B) |
| `--output` | Path to save transcription & summary |
| `--temperature` |Controls randomness (0.0-1.0) |  
| `--repeat-penalty` |Controls how much to penalize repetition 1.0 - 1.4|

### **Detailed Model Descriptions**
#### **Whisper Models**
Whisper is an automatic speech recognition (ASR) system from OpenAI. Different models offer a trade-off between **accuracy and speed**:
- **tiny** → Fastest, but lowest accuracy.
- **base** → Slightly better accuracy, still very fast.
- **small** → Balanced choice between speed and accuracy. **Default**
- **medium** → High accuracy, but slower.
- **large** → Best accuracy, slowest performance.

#### **Default LLM Model for Summarization**
- **L3.2-Rogue-Creative-Instruct-7B-GGUF** (Recommended for Mac M1 16GB)  
Fine-tuned version of LLama3.2 3B at Quant 4, expanded to 67 layers using the Brainstorm 40x method. It outperforms the standard Llama 3.2 7B in conversational coherence and instruction adherence.
This model is optimized for creative writing and structured text summarization** → used when no specific LLM repo is provided**.  
    - max context window of 131,072
    - min input length of 1 token
    - different Quant choices available  

---

## 🧠 Alternative LLM Models
This script will support **different LLMs** for summarization in the future:
- **Llama 3.2 7B (Standard Version)** → The base Llama 3.2 7B model, optimized for general-purpose NLP tasks but less tuned for structured summarization.
- **Llama 3.2 13B** → More detailed summaries with improved comprehension but requires more memory.
- **Llama 3.2 11B** → Enhanced performance with longer context support.
- **Mixtral 8x7B** → A Mixture-of-Experts model, ideal for high-quality, in-depth summarization, **but currently not available.**
- **Mistral 7B** → Lightweight and efficient.
- **Llama 2 7B** → Older but still usable as a fallback option.
- **Starling 7B** → An alternative with optimized prompt-following capabilities.

---

## 💻 Running on NVIDIA (CUDA) Instead of Metal
To run on **NVIDIA GPUs**, install **CUDA dependencies**:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Then, adjust `audio_summarizer.py`:
```python
model_params = {
    "n_gpu_layers": -1,  # Use full GPU acceleration
    "n_ctx": 131072,  # Extend context length
    "n_batch": 512,
}
```
Run:
```sh
python audio_summarizer.py my_audio.mp3
```

---

## 🔧 Troubleshooting
- **Issue: Whisper not found** → Run `pip install openai-whisper`
- **Issue: Out of memory** → Reduce `n_ctx` in `audio_summarizer.py`
- **Issue: CUDA errors** → Check `nvidia-smi` to confirm GPU availability

---

## 📜 License
This project is open-source under the **MIT License**.

---

## 📋 TODO
- 📌 Gather working repos for various open-source LLMs.
- 📌 Implement temperature and rep value selection vie flags `--temperature` & `--repeat-penalty`
- 📌 Fix model selection to work properly via `--llama-model` flag.
- 📌 Conduct intensive testing on stronger Apple Silicon chips and NVIDIA GPUs.


🚀 **Happy Transcribing!**
