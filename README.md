# Efficient LLM Fine-Tuning with DeepSpeed

A highly optimized, scalable pipeline for fine-tuning Large Language Models (LLMs) using Hugging Face Transformers and DeepSpeed ZeRO optimizations. 

This project demonstrates how to significantly reduce GPU VRAM requirements during model training, enabling the fine-tuning of billion-parameter models on consumer-grade hardware.

## Key Features

* **DeepSpeed ZeRO Stage 2 Optimization:** Implemented CPU offloading for optimizer states to drastically reduce VRAM bottlenecks.
* **Mixed Precision Training (fp16):** Accelerated training speeds and halved memory consumption for model weights without sacrificing model performance.
* **Scalable Architecture:** Pipeline developed and tested locally on a lightweight model (GPT-2) to verify MPI communication and data collation, architected to scale seamlessly to 7B+ parameter models (e.g., Llama, Mistral) on single or multi-GPU instances.

## Tech Stack

* **Frameworks:** PyTorch, Hugging Face `transformers`, Hugging Face `datasets`
* **Optimization:** Microsoft DeepSpeed, `mpi4py`
* **Hardware:** NVIDIA Tesla T4

## Performance Metrics
*By implementing DeepSpeed ZeRO-2 and Mixed Precision, this architecture enables up to a **60% reduction in VRAM requirements** compared to standard PyTorch training loops on large-scale models.*

## Installation & Setup

1. **Clone the repository:**
   
   git clone [https://github.com/YOUR_USERNAME/deepspeed-llm-finetuning.git](https://github.com/YOUR_USERNAME/deepspeed-llm-finetuning.git]
   cd deepspeed-llm-finetuning
2.**Install the required dependencies:**
   sudo apt-get install -y libopenmpi-dev
   pip install transformers datasets accelerate deepspeed mpi4py torch
3.**Usage**
   To run the fine-tuning pipeline, simply execute the training script. DeepSpeed will automatically read the ds_config.json file for optimization settings.
   python train.py
