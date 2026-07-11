# Mistral-7B Medical QA Fine-Tuning (QLoRA)

Parameter-efficient fine-tuning of Mistral-7B-Instruct on real-world patient-doctor conversations, adapting a general-purpose LLM into a domain-specific medical Q&A assistant using 4-bit quantized LoRA (QLoRA).

## Overview

This project fine-tunes `mistralai/Mistral-7B-Instruct-v0.2` on the ChatDoctor-HealthCareMagic dataset to explore how far a 7B open-weight model can be pushed toward domain specialization under consumer-GPU memory constraints. The focus is on the full fine-tuning pipeline — data curation, quantization-aware training, and evaluation — rather than building a production clinical tool.

**This is a research/portfolio project, not a clinical product.** See [Safety & Limitations](#safety--limitations) below.

## Motivation

General-purpose instruction-tuned LLMs are fluent but not necessarily accurate or well-calibrated on specialized domains like medicine. This project investigates:
- How much domain adaptation QLoRA can achieve on a 7B model with a few thousand curated examples
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
   - Removed formatted examples exceeding 512 tokens (memory constraints)
3. **Train/test split** — 80/20, fixed seed for reproducibility
4. **Tokenization** — fixed-length (512 token) padding/truncation, labels set for causal language modeling

## Model & Method

| Component | Detail |
|---|---|
| Base model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning method | QLoRA (4-bit NF4 quantization + LoRA adapters) |
| Quantization | 4-bit, double quantization, fp16 compute dtype |
| LoRA rank / alpha | r=8, alpha=16 |
| Target modules | `q_proj`, `v_proj` |
| Optimizer | paged AdamW (32-bit) |
| LR schedule | Cosine, peak LR 2e-4, 100 warmup steps |
| Epochs | 3 |
| Effective batch size | 16 (batch 4 × grad accumulation 4) |

Training and evaluation loss are tracked via TensorBoard; checkpoints are saved on a step interval with the best checkpoint (by eval loss) restored at the end of training.

## Evaluation

Beyond training/eval loss and perplexity, the project includes a qualitative evaluation pass: a held-out sample of prompts is scored manually against a small rubric covering:
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
- Any input describing a medical emergency should always be redirected to seek immediate professional or emergency care rather than answered by the model directly.

## Tech Stack

- **Model & training:** Hugging Face `transformers`, `peft`, `bitsandbytes`, `trl`, `accelerate`
- **Data:** Hugging Face `datasets`
- **Experiment tracking:** TensorBoard
- **Hosting for models/weights:** Hugging Face Hub

## Project Structure

```
├── mistral_finetuning.ipynb     # End-to-end training notebook
├── mistral-7b-healthcare-qlora/ # Training outputs
│   ├── checkpoint-*/            # Intermediate checkpoints
│   ├── final-adapter/           # Trained LoRA adapter + tokenizer
│   └── merged-model/            # Base model merged with LoRA weights
└── README.md
```

## Setup

```bash
pip install transformers datasets peft bitsandbytes accelerate trl huggingface_hub
```

Authenticate with Hugging Face (required — Mistral-7B-Instruct is a gated model):

```python
from huggingface_hub import login
login(token="<your_hf_token>")  # use a secrets manager, never hardcode
```

Accept the Mistral-7B-Instruct-v0.2 license on the [model page](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) before attempting to download weights.

## Future Work

- Expand LoRA target modules beyond attention projections to include MLP layers for stronger domain adaptation
- Move to completion-only loss masking so the model is trained solely on doctor responses, not patient input
- Build out a larger, rubric-scored evaluation set rather than spot-checked qualitative review
- Explore comparison against a retrieval-augmented approach as a complementary or alternative strategy to pure fine-tuning

## License

The code in this repository is released under the [MIT License](LICENSE).

This covers the training pipeline, notebooks, and scripts only. The **dataset** (ChatDoctor-HealthCareMagic-100k) is not covered by this license — its original terms are unspecified by the maintainers, so it is used here strictly for research/educational purposes and is not redistributed. Any fine-tuned model weights inherit the license terms of the base model, Mistral-7B-Instruct-v0.2 (Apache 2.0), in addition to this project's own MIT terms for the training code itself.

## Acknowledgments

- [ChatDoctor project](https://github.com/Kent0n-Li/ChatDoctor) and the HealthCareMagic dataset authors
- Mistral AI for the base model
- Hugging Face for the `transformers`/`peft`/`trl` ecosystem
