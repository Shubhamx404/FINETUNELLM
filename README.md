# FINE TUNE LAMMA USING QLORA
# 🦙 Fine-Tuning LLaMA-2-7B with QLoRA on Google Colab

This repository demonstrates how to fine-tune **Meta’s LLaMA-2-7B** model efficiently using **QLoRA** and **PEFT** on a single Google Colab GPU.  
It turns the base model into a custom **instruction-following chatbot** trained on the **Indian Constitution Articles dataset**.

---
Check more models  on Hugging Face 
# colab link : [https://colab.research.google.com/drive/1-lxQKb9P8Xde0ft4hTLFFJ_O6V8vA8uo](https://colab.research.google.com/github/Shubhamx404/FINETUNELLM/blob/main/finetuning%20with%20qlora.ipynb)

## 📘 Dataset

- **Name:** `nisaar/Articles_Constitution_3300_Instruction_Set`
- **Source:** [Hugging Face Datasets](https://huggingface.co/datasets/nisaar/Articles_Constitution_3300_Instruction_Set)
- **Description:**  
  Instruction-based dataset derived from the Indian Constitution, containing ~3300 examples.  
  Each sample includes:
  ```json
  {
    "instruction": "Explain Article 21 of the Indian Constitution.",
    "input": "",
    "output": "Article 21 guarantees the protection of life and personal liberty..."
  }

!pip install -q transformers accelerate bitsandbytes peft trl datasets einops
🧠 Model & Approach



* Base Model: meta-llama/Llama-2-7b-hf

* Quantization: 4-bit (using bitsandbytes)

* Finetuning Strategy: QLoRA (Quantized Low-Rank Adaptation)

* Trainer: SFTTrainer from trl

**Key benefits:**

* Runs on a single GPU (Colab T4 / A100)

* Significantly reduced VRAM usage

* Compatible with PEFT for adapter-based fine-tuning

**Saving model :**
trainer.model.save_pretrained("Llama2-7b-constitution-qlora")
tokenizer.save_pretrained("Llama2-7b-constitution-qlora")

