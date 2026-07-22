# Mistral-7B Medical QA Fine-Tuning (QLoRA)

Parameter-efficient fine-tuning of Mistral-7B-Instruct on real-world patient-doctor conversations, adapting a general-purpose LLM into a domain-specific medical Q&A assistant using 4-bit quantized LoRA (QLoRA).

📝 Full write-up: [Medium walkthrough](https://medium.com/@diwash.adhi4/fine-tuning-mistral-7b-for-medical-q-a-with-qlora-a-practical-walkthrough-5e97faac9c8b)

## Overview

This project fine-tunes `mistralai/Mistral-7B-Instruct-v0.2` on the ChatDoctor-HealthCareMagic dataset to explore how far a 7B open-weight model can be pushed toward domain specialization under consumer-GPU memory constraints. The focus is on the full fine-tuning pipeline — data curation, quantization-aware training, and evaluation — rather than building a production clinical tool.

**This is a research/portfolio project, not a clinical product.** See [Safety & Limitations](#safety--limitations) below.

## Trained Model

The fine-tuned LoRA adapter is published on Hugging Face Hub:

**[d1wash/mistral-7b-healthcare-qlora](https://huggingface.co/d1wash/mistral-7b-healthcare-qlora)**

Load it directly on top of the base model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_repo = "d1wash/mistral-7b-healthcare-qlora"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, adapter_repo)
```

Or download the adapter files directly with the Hugging Face CLI:

```bash
huggingface-cli download d1wash/mistral-7b-healthcare-qlora
```

## Motivation

General-purpose instruction-tuned LLMs are fluent but not necessarily accurate or well-calibrated on specialized domains like medicine. This project investigates:
- How much domain adaptation QLoRA can achieve on a 7B model with a curated subset of examples under a fixed compute/time budget
- What data quality and safety trade-offs show up when adapting an LLM to a sensitive domain
- How to measure whether fine-tuning actually improved domain performance, rather than just changed writing style

## Dataset

**Source:** [`lavita/ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)

- ~112,000 real patient questions and doctor responses collected from HealthCareMagic.com
- Identity information removed by the original dataset authors; light automated grammar correction already applied upstream
- **License status:** not explicitly specified by the dataset maintainers. Treated here as research-use only — not redistributed, not used for any commercial claim.

### Data Preparation

1. **Instruction formatting** — each example reformatted into Mistral's `[INST]...[/INST]` chat template, framed as a doctor-patient exchange
2. **Quality filtering**:
   - Removed examples with empty input or output fields
   - Removed responses under 20 words (low-information answers)
   - Removed formatted examples exceeding 256 words (memory/time constraints)
3. **Subsampling** — filtered dataset (~109,800 examples) subsampled to 15,000 training / 1,500 evaluation examples to fit training within a fixed time budget on a single free-tier T4 GPU
4. **Train/test split** — 80/20, fixed seed for reproducibility
5. **Tokenization** — fixed-length (256 token) padding/truncation, labels set for causal language modeling

## Model & Method

| Component | Detail |
|---|---|
| Base model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning method | QLoRA (4-bit NF4 quantization + LoRA adapters) |
| Quantization | 4-bit, double quantization, fp16 compute dtype |
| LoRA rank / alpha | r=8, alpha=16 |
| Target modules | `q_proj`, `v_proj` |
| Optimizer | paged AdamW (8-bit) |
| LR schedule | Cosine, peak LR 2e-4, 50 warmup steps |
| Max steps | 900 (~1 epoch over the 15k-example training subset) |
| Effective batch size | 16 (per-device batch 8 × grad accumulation 2) |
| Hardware | 1x NVIDIA T4 (16GB), Lightning AI Studio free tier |

Training and evaluation loss are tracked via TensorBoard; checkpoints are saved on a step interval with the best checkpoint (by eval loss) restored at the end of training.

## Evaluation

Training and validation loss were tracked at each checkpoint and decreased consistently over the course of training, with validation loss tracking closely alongside training loss throughout — indicating the model generalized to the held-out set rather than memorizing the training data.

Beyond loss and perplexity, a qualitative evaluation pass compares base-model vs. fine-tuned outputs on the same held-out prompts, checking for:
- Medical relevance and coherence of the response
- Absence of harmful or contraindicated advice
- Appropriate escalation language for emergency-flagged symptoms (e.g. chest pain, suicidal ideation)

Loss and perplexity alone do not verify medical accuracy or safety — they only indicate how well the model learned the style and structure of the training data.

## Safety & Limitations

This model is **not validated for clinical use** and should never be used as a substitute for professional medical advice, diagnosis, or treatment.

- The training data reflects informal, non-verified patient-doctor forum exchanges, not peer-reviewed clinical guidance.
- The model can hallucinate confident-sounding but medically incorrect information, as any LLM fine-tuned on unstructured text can.
- No formal clinical validation, physician review, or regulatory evaluation has been performed.
- The dataset itself carries no clear usage license — this project treats it as strictly non-commercial and research/educational in nature.
- Training used a 15,000-example subset of the full ~110,000-example dataset (a deliberate time/compute trade-off), so coverage of rarer conditions or edge cases is limited.
- Any input describing a medical emergency should always be redirected to seek immediate professional or emergency care rather than answered by the model directly.

## Tech Stack

- **Model & training:** Hugging Face `transformers`, `peft`, `bitsandbytes`, `trl`, `accelerate`
- **Data:** Hugging Face `datasets`
- **Experiment tracking:** TensorBoard
- **Demo UI:** Gradio
- **Hosting for models/weights:** Hugging Face Hub

## Project Structure

```
├── mistralfinetunning.ipynb     # End-to-end training notebook
├── app.py                       # Gradio demo app
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install transformers datasets peft bitsandbytes accelerate trl gradio huggingface_hub
```

Authenticate with Hugging Face (required — Mistral-7B-Instruct is a gated model):

```python
from huggingface_hub import login
login(token="<your_hf_token>")  # use a secrets manager, never hardcode
```

Accept the Mistral-7B-Instruct-v0.2 license on the [model page](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) before attempting to download weights.

## Running the Demo

```bash
python app.py
```

This launches a Gradio chat interface (with `share=True` for a temporary public link) that loads the base model plus the published LoRA adapter directly from Hugging Face Hub.

## Future Work

- Train on the full ~110,000-example dataset given more compute/time budget
- Expand LoRA target modules beyond attention projections to include MLP layers for stronger domain adaptation
- Move to completion-only loss masking so the model is trained solely on doctor responses, not patient input
- Build out a larger, rubric-scored evaluation set rather than spot-checked qualitative review
- Explore comparison against a retrieval-augmented approach as a complementary or alternative strategy to pure fine-tuning

## License

The code in this repository is released under the [MIT License](LICENSE).

This covers the training pipeline, notebooks, and scripts only. The **dataset** (ChatDoctor-HealthCareMagic-100k) is not covered by this license — its original terms are unspecified by the maintainers, so it is used here strictly for research/educational purposes and is not redistributed. Any fine-tuned model weights inherit the license terms of the base model, Mistral-7B-Instruct-v0.2 (Apache 2.0), in addition to this project's own MIT terms for the training code itself.

## Acknowledgments

- HealthCareMagic dataset authors
- Mistral AI for the base model
- Hugging Face for the `transformers`/`peft`/`trl` ecosystem
