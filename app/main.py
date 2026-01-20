import gradio as gr
import torch
from unsloth import FastLanguageModel

# 1. CONFIGURATION
# Set this to the folder where you saved your Colab adapters (e.g., 'legalese_model_lora')
model_path = "../legalese_model_lora" 

# 2. LOAD THE MODEL & TOKENIZER
# We load the base model and then attach your fine-tuned 'brain' (the adapters)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, 
    max_seq_length = 2048,
    load_in_4bit = True, # Crucial for running on local consumer GPUs
)
FastLanguageModel.for_inference(model) # Enable 2x faster inference

# 3. DEFINE THE INFERENCE FUNCTION
def simplify_legalese(complex_text):
    # This must match the prompt format you used during training!
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a legal aid assistant. Simplify the following legalese into plain, easy-to-understand English.<|eot_id|><|start_header_id|>user<|end_header_id|>
{complex_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 128,
        use_cache = True,
        temperature = 0.1 # Keep it low for factual/legal tasks
    )
    
    # Decode only the new tokens (the assistant's response)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant")[-1].strip()

# 4. BUILD THE GRADIO UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚖️ Legalese Simplifier")
    gr.Markdown("Translate complex legal jargon into plain English instantly.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Complex Legal Clause", 
                placeholder="Paste legalese here (e.g., 'Notwithstanding the foregoing...')",
                lines=8
            )
            submit_btn = gr.Button("Simplify", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Plain English Translation", lines=8)
            
    submit_btn.click(fn=simplify_legalese, inputs=input_text, outputs=output_text)
    
    gr.Examples(
        examples=[
            ["The indemnified party shall be held harmless against any and all claims."],
            ["In the event of a material breach, the non-breaching party may terminate immediately."]
        ],
        inputs=input_text
    )

# 5. LAUNCH
if __name__ == "__main__":
    demo.launch()