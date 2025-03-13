# AI Voice Transcription & Autosummary (Open Source)

A **small Python script** that utilizes **OpenAI Whisper** for speech-to-text transcription and an **open-source LLM** (like Llama 3.2 or Mixtral) for summarizing audio files.

## Features
- 🎙 **Transcribe** spoken content from an audio file.
- ✍️ **Summarize** the transcribed text using a local LLM.
- 🚀 **Supports multiple models** (Llama 3.2, Mixtral, etc.).
- ⚡ **Optimized for macOS (Metal) & NVIDIA (CUDA)**.

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
python audio_summarizer.py my_audio.mp3 --whisper-model small --llama-model 7B
```

### **Available Options**
| Argument | Description |
|----------|-------------|
| `my_audio.mp3` | Path to the input audio file |
| `--whisper-model` | Select Whisper model (tiny, base, small, medium, large) |
| `--llama-model` | Choose LLM (7B, 13B, 11B, Mixtral) |
| `--output` | Path to save transcription & summary |

---

## 🧠 Alternative LLM Models
This script supports **different LLMs** for summarization:
- **Llama 3.2 (7B, 13B, 11B)** → Optimized for macOS (Metal) & NVIDIA (CUDA)
- **Mixtral 8x7B** → Mixture of Experts for high performance
- **Mistral 7B** → Lightweight and efficient

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
python audio_summarizer.py my_audio.mp3 --llama-model Mixtral
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

## 📬 Contributing
- **Fork** the repo
- **Create a feature branch** (`git checkout -b my-feature`)
- **Submit a PR** 🎉

🚀 **Happy Transcribing!**
