"""
Gradio demo for the Mistral-7B-Instruct QLoRA medical Q&A fine-tune.
Loads the base model in 4-bit + the LoRA adapter directly from Hugging Face Hub.

Features:
- Chat tab: talk to the fine-tuned model, with streaming responses and 👍/👎 feedback
- Compare tab: same prompt run through base Mistral vs. the fine-tuned adapter, side by side

Run locally (e.g. in a Lightning Studio, right after training):
    python app.py

Deploy on HF Spaces:
    1. Create a new Space -> SDK: Gradio, Hardware: T4 small (or ZeroGPU if available)
    2. Push this app.py + requirements.txt to the Space repo
    3. (Optional) Set ADAPTER_REPO as a Space variable/secret if you want to override the default below
"""

import os
import copy
import threading

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel

# ---- Config ------------------------------------------------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "d1wash/mistral-7b-healthcare-qlora")

# Detect whether we're running inside an HF Space (used to tweak launch() below)
IS_SPACE = os.environ.get("SPACE_ID") is not None

SYSTEM_PROMPT = "You are a helpful medical assistant."

DISCLAIMER = (
    "⚠️ **Disclaimer:** This is a portfolio/research demo fine-tuned on the "
    "ChatDoctor-HealthCareMagic dataset. It is **not** a substitute for professional "
    "medical advice, diagnosis, or treatment. Always consult a qualified healthcare "
    "provider for medical concerns. If you are experiencing a medical emergency, "
    "contact emergency services immediately."
)

# ---- Load model + adapter (once, at startup) ---------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model in 4-bit...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading LoRA adapter from {ADAPTER_REPO} ...")
finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
finetuned_model.eval()

print("✅ Model + adapter loaded and ready.")


# ---- Shared helpers -----------------------------------------------------------
def build_prompt(message: str) -> str:
    """Format a patient message into Mistral's instruction template."""
    return f"""<s>[INST] {SYSTEM_PROMPT}

Patient: {message} [/INST]

Doctor:"""


def generate_stream(model, message, max_new_tokens, temperature, top_p):
    """Yield partial text as tokens are generated (for streaming in the Chat tab)."""
    prompt = build_prompt(message)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text


def generate_full(model, message, max_new_tokens, temperature, top_p):
    """Non-streaming generation, used for the side-by-side Compare tab."""
    prompt = build_prompt(message)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    response = full_text[len(prompt_text):].strip()
    return response if response else "(No response generated — try rephrasing your question.)"


# ---- Chat tab callback ---------------------------------------------------------
def chat_respond(message, history, max_new_tokens, temperature, top_p):
    if not message or not message.strip():
        yield "Please enter a question."
        return
    try:
        for partial in generate_stream(finetuned_model, message, max_new_tokens, temperature, top_p):
            yield partial
    except Exception as e:
        yield f"Sorry, something went wrong generating a response: {e}"


def on_feedback(is_positive):
    # Placeholder hook -- in a real deployment this would log to a file/DB
    # for later analysis of response quality.
    label = "👍 positive" if is_positive else "👎 negative"
    print(f"[feedback] User marked last response as {label}")
    return gr.Info(f"Thanks for the feedback ({label})!")


# ---- Compare tab callback -------------------------------------------------------
def compare_respond(message, max_new_tokens, temperature, top_p):
    if not message or not message.strip():
        return "Please enter a question.", "Please enter a question."
    try:
        base_response = generate_full(base_model, message, max_new_tokens, temperature, top_p)
    except Exception as e:
        base_response = f"Error generating base model response: {e}"
    try:
        finetuned_response = generate_full(finetuned_model, message, max_new_tokens, temperature, top_p)
    except Exception as e:
        finetuned_response = f"Error generating fine-tuned response: {e}"
    return base_response, finetuned_response


# ---- UI --------------------------------------------------------------------------
with gr.Blocks(title="Medical Q&A Assistant (Mistral-7B QLoRA)") as demo:
    gr.Markdown("# 🩺 Medical Q&A Assistant")
    gr.Markdown(
        "Fine-tuned **Mistral-7B-Instruct** on the ChatDoctor-HealthCareMagic-100k dataset "
        "using QLoRA (4-bit NF4 quantization + LoRA). "
        f"[Model card & training details](https://huggingface.co/{ADAPTER_REPO})."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Tabs():
        # ---- Tab 1: Chat -------------------------------------------------------
        with gr.Tab("Chat"):
            with gr.Accordion("Generation settings", open=False):
                max_new_tokens = gr.Slider(32, 512, value=256, step=8, label="Max new tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

            gr.ChatInterface(
                fn=chat_respond,
                additional_inputs=[max_new_tokens, temperature, top_p],
                examples=[
                    "I've had a persistent headache every morning for a week. What could be causing it?",
                    "What are common early symptoms of type 2 diabetes?",
                    "Is it normal to feel dizzy after standing up quickly?",
                ],
                title=None,
            )

            gr.Markdown("Was the last response helpful?")
            with gr.Row():
                thumbs_up = gr.Button("👍 Helpful")
                thumbs_down = gr.Button("👎 Not helpful")
            feedback_note = gr.Markdown(visible=True)

            thumbs_up.click(fn=lambda: "Thanks for the feedback! 👍", outputs=feedback_note)
            thumbs_down.click(fn=lambda: "Thanks for the feedback! 👎 We'll use this to improve.", outputs=feedback_note)

        # ---- Tab 2: Compare (base vs fine-tuned) --------------------------------
        with gr.Tab("Compare: Base vs. Fine-tuned"):
            gr.Markdown(
                "Enter a medical question and see how the **original Mistral-7B-Instruct** "
                "responds compared to the **QLoRA fine-tuned** version, side by side."
            )

            with gr.Accordion("Generation settings", open=False):
                cmp_max_new_tokens = gr.Slider(32, 512, value=200, step=8, label="Max new tokens")
                cmp_temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                cmp_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

            cmp_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. I've had a persistent headache every morning for a week. What could be causing it?",
                lines=3,
            )
            cmp_button = gr.Button("Compare responses", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Base Mistral-7B-Instruct**")
                    base_output = gr.Textbox(label="", lines=10, interactive=False, show_copy_button=True)
                with gr.Column():
                    gr.Markdown("**Fine-tuned (QLoRA on ChatDoctor)**")
                    finetuned_output = gr.Textbox(label="", lines=10, interactive=False, show_copy_button=True)

            cmp_button.click(
                fn=compare_respond,
                inputs=[cmp_input, cmp_max_new_tokens, cmp_temperature, cmp_top_p],
                outputs=[base_output, finetuned_output],
            )

if __name__ == "__main__":
    # share=True is only useful for local/Studio testing; HF Spaces handles
    # public access itself, so we skip it there.
    demo.launch(share=not IS_SPACE)
