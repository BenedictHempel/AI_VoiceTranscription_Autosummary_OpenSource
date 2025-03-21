# AI Voice Transcription & Autosummary (Open Source)

A **small Python script** that utilizes **OpenAI Whisper** for speech-to-text transcription and an **open-source LLM** (like Llama 3.2 or Mistral) for summarizing audio files. Works offline, doesn´t require expensive hardware & prevents data leakages.
Autoloading of all whisper model sizes & autoloading of LLMs from HuggingFace repo.  
Adaptability: LLM tuneable by model, modelsize & quantization as well as by context window, temperature & repeatition penality to fit all resources and needs.

## Features
- **Transcription:** Runs Whisper **locally** without sending data to the cloud.
- **Summarization:** Uses **local Large Language Models (LLMs)** from Hugging Face.
- **Multiple LLM types included:**
  - **Instruction-Tuned Models** (e.g., Mistral, Starling) for structured summarization.
  - **Standard General-Purpose LLMs** (e.g., Llama 3.2) for NLP tasks.
  - **Mixture-of-Experts (MoE) Models** (e.g., Mixtral, Beyonder) for efficient processing.
- **Optimized for Apple Silicon** with **Metal acceleration**.
- **GPU Support:** Works with **NVIDIA CUDA**, **Apple Metal**, and **CPU-only setups**.
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

| Argument | Description |
|----------|-------------|
| `my_audio.mp3` | Path to the input audio file |
| `--whisper-model` | Select Whisper model (tiny, base, small, medium, large) |
| `--llama-model` | Choose LLM (Mistral-7B, Llama-3.2-3B-Instruct, Llama-3.2-1B-Instruct, Llama-2-7B, Starling-7B, Mixtral-8x7B, NexoNimbus-7B, Falcon-Mamba-7B, Beyonder-4x7B, laser-dolphin-mixtral-2x7B, Dr_Samantha-7B, Baichuan-2-13B) |
| `--output` | Path to save transcription & summary |
| `--temperature` |Controls randomness (0.0-1.0) |  
| `--repeat-penalty` |Controls how much to penalize repetition 1.0 - 1.4|
| `--context-size` |Defines the context window size (see below)|

### **Detailed Model Descriptions**
#### **Whisper Models**
Whisper is an automatic speech recognition (ASR) system from OpenAI. Different models offer a trade-off between **accuracy and speed**:
- **tiny** → Fastest, but lowest accuracy.
- **base** → Slightly better accuracy, still very fast.
- **small** → Balanced choice between speed and accuracy. **Default**
- **medium** → High accuracy, but slower.
- **large** → Best accuracy, slowest performance.

  
## 🧠 Default LLM Model for Summarization

### **Mistral-7B**  

- **Developer:** Mistral AI  
- **Open-Source Status:** ✅ Fully Open-Source  
- **Ancestry:** Based on **LLaMA 2**, optimized for **efficiency & summarization**.  

#### **Key Features:**
- 🔹 **Strong summarization & instruction following.**
- 🔹 **Optimized for Mac Metal acceleration.**
- 🔹 **Uses Q4_K_M quantization for best performance.**
- 🔹 **Efficient even on lower-end hardware.**
- 🔹 **Extensible context window support (8192+ tokens).**

📌 **This model gave the most consistent results within a reasonable performance/resource ratio during testing on Mac M1 16GB with metal acceleration.**


---

## **🧠 Alternative LLM Models**
| Model | Developer | Open-Source? | Type | Description |
|-------|-----------|--------------|------|-------------|
| **Mistral-7B (Default)** | Mistral AI | ✅ Open | Instruction-Tuned | Optimized for summarization, best performance on macOS Metal |
| **Llama-3.2-3B-Instruct** | Meta AI | ❌ Closed | Standard | General-purpose NLP & summarization |
| **Llama-3.2-1B-Instruct** | Meta AI | ❌ Closed | Standard | Lighter version for low-memory setups |
| **Llama-2-7B** | Meta AI | ✅ Open | Standard | Older but still effective |
| **Starling-7B** | Berkeley AI | ✅ Open | Instruction-Tuned | Highly optimized for structured summarization |
| **Mixtral-8x7B** | Mistral AI | ✅ Open | MoE | Mixture-of-Experts model, very efficient |
| **NexoNimbus-7B** | Cohere | ✅ Open | Standard | Balanced for summarization & generation |
| **Falcon-Mamba-7B** | TII UAE | ✅ Open | Experimental | High-efficiency model with novel techniques |
| **Beyonder-4x7B** | Stability AI | ✅ Open | MoE | Multi-expert model for complex tasks |
| **laser-dolphin-mixtral-2x7B** | Unknown | ✅ Open | MoE | Hybrid Mixtral variant for efficiency |
| **Dr_Samantha-7B** | Stability AI | ✅ Open | Instruction-Tuned | Optimized for conversational AI & medical text |
| **Baichuan-2-13B** | Baichuan AI | ✅ Open | Standard | Chinese & multilingual NLP model |


---

🛠 Context Window Flag (--context-size)

The context window determines how many tokens the model can process at once.
| Argument | Description |
|----------|-------------|
|"x"|	Base context size (Default: 8192)|
|"x*2"|	Double the base context (Recommended for large summaries)|
|"x*4", "x*8"|	Further increases (requires more VRAM)|
|"x/2", "x/4", "x/8"|	Reduces the context window (for low-memory setups)|
|Specific Number|	e.g., "32768" to manually define the size|

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
- **Return -3 Error (Ollama/Llama.cpp)** → Reduce `--context-size`, check if **VRAM/CPU RAM is overloaded** 
- **Exceeding Context Size (LLM Fails Midway)** → Lower `--context-size` to prevent **exceeding model’s token limit** 

---

## 📜 License
This project is open-source under the **MIT License**.

---

## 📋 TODO
- 📌 add phi4, gemma3, deepseek-r1, qwen
- 📌 integrate visual capabilities with llava
- 📌 Add resource intensive models
- 📌 Conduct intensive testing on stronger Apple Silicon chips and NVIDIA GPUs
- 📌 Add `--quantization` flag to manually select quantization levels (Q4, Q6, Q8)
    - Auto-download best quantized model for user’s hardware
- 📌 Implement adaptive temperature & penalty settings based on input length
- 📌 Enable GPU load balancing between multiple available GPUs
- 📌 add use existing transcription file from transient/from provided file
- 📌 add option to only create transcript
    - add option to provide target path for transcript & and for summary
- 📌 add option to start chat mode with context
- 📌 add prompt modification by providing string via flag
- 📌 add permanent personalization options


🚀 **Happy Transcribing!**
