"""
Gradio App for SmolLM2-135M - Using .pt checkpoint

Simple inference app that loads the converted .pt checkpoint directly.
"""

import gradio as gr
import torch
import sys
import os

# Add Assignment_13 to path for model imports
current_dir = os.path.dirname(os.path.abspath(__file__))
assignment_13_dir = os.path.join(current_dir, 'Assignment_13')
if os.path.exists(assignment_13_dir):
    sys.path.insert(0, assignment_13_dir)

from model import SmolLM2ForCausalLM
from transformers import AutoTokenizer

# Global variables
model = None
tokenizer = None


def load_model():
    """Load the model and tokenizer from .pt file"""
    global model, tokenizer
    
    # Look for .pt checkpoint
    checkpoint_path = 'shakespeare_smollm2_step50.pt'
    
    if not os.path.exists(checkpoint_path):
        return None, None, f"❌ Checkpoint not found: {checkpoint_path}\n\nPlease run: python simple_convert.py"
    
    try:
        print(f"[1/3] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Tokenizer loaded: vocab={tokenizer.vocab_size:,}")
        
        print(f"\n[2/3] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        config = checkpoint['config']
        state_dict = checkpoint['model_state_dict']
        global_step = checkpoint.get('global_step', 0)
        
        print(f"✓ Checkpoint loaded:")
        print(f"  - Global step: {global_step}")
        print(f"  - Weight tensors: {len(state_dict)}")
        print(f"  - Vocab size: {config['vocab_size']:,}")
        
        print(f"\n[3/3] Creating model...")
        model = SmolLM2ForCausalLM(config)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Keep on CPU
        device = 'cpu'
        model = model.to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        
        return model, tokenizer, f"""Model loaded successfully!
        
Device: {device.upper()}
Checkpoint: {checkpoint_path}
Training steps: 5,050 (trained from scratch)
Current checkpoint step: {global_step}
Tokenizer: {type(tokenizer).__name__} (vocab={tokenizer.vocab_size:,})
Parameters: {param_count:,} ({param_count/1e6:.1f}M)
Architecture: {checkpoint.get('architecture', 'SmolLM2-135M')}
"""
    
    except Exception as e:
        import traceback
        return None, None, f"❌ Error loading model: {str(e)}\n\nFull error:\n{traceback.format_exc()}"


def generate_text_gradio(
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    use_sampling: bool = True
):
    """Generate text using the model"""
    
    if model is None or tokenizer is None:
        return "❌ Model not loaded. Please wait for initialization or check logs."
    
    if not prompt or prompt.strip() == "":
        return "❌ Please enter a prompt"
    
    try:
        # Tokenize input
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        # Get device from model parameters
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Generate with specified parameters
        # Note: Custom generate method only supports max_length, temperature, and top_k
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=max_length,  # This is max new tokens in custom generate
                temperature=temperature if use_sampling else 1.0,
                top_k=top_k if use_sampling else 0
            )
        
        # Decode and return
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
    
    except Exception as e:
        import traceback
        return f"❌ Error during generation: {str(e)}\n\n{traceback.format_exc()}"


# Example prompts for Shakespeare generation
example_prompts = [
    ["ROMEO:", 150, 0.8, 50, True],
    ["JULIET:", 150, 0.8, 50, True],
    ["First Citizen:", 150, 0.8, 50, True],
    ["To be or not to be", 150, 0.8, 50, True],
    ["All the world's a stage", 150, 0.8, 50, True],
]


# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="SmolLM2-135M Shakespeare Generator", theme=gr.themes.Base()) as interface:
        gr.Markdown("""
        # SmolLM2-135M Shakespeare Generator
        
        A 135M parameter language model trained on Shakespeare's works.
        
        **Model Details:**
        - Architecture: SmolLM2-135M with Grouped Query Attention
        - Tokenizer: HuggingFace SmolLM2 BPE (49,152 vocab)
        - Training: 5,050 steps on Shakespeare corpus
        - Parameters: ~135M
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Input")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt (e.g., 'ROMEO:', 'To be or not to be')",
                    lines=3,
                    value="ROMEO:"
                )
                
                generate_btn = gr.Button("Generate Text", variant="primary", size="lg")
                
                # Parameters
                gr.Markdown("### Generation Parameters")
                
                max_length = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=150,
                    step=10,
                    label="Max Length",
                    info="Maximum number of tokens to generate"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative, lower = more deterministic"
                )
                
                top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Top-k",
                    info="Number of top tokens to consider (0 = no filtering)"
                )
                
                use_sampling = gr.Checkbox(
                    value=True,
                    label="Use Sampling",
                    info="If unchecked, uses greedy decoding (temperature=1.0)"
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### Generated Text")
                output_text = gr.Textbox(
                    label="Output",
                    lines=20,
                    interactive=False
                )
                
                # Example prompts
                gr.Markdown("### Example Prompts")
                gr.Examples(
                    examples=example_prompts,
                    inputs=[prompt_input, max_length, temperature, top_k, use_sampling],
                    outputs=output_text,
                    fn=generate_text_gradio,
                    cache_examples=False,
                )
        
        # Button click event
        generate_btn.click(
            fn=generate_text_gradio,
            inputs=[prompt_input, max_length, temperature, top_k, use_sampling],
            outputs=output_text
        )
        
        gr.Markdown("""
        ---
        ### Tips for Best Results:
        - Use character names (ROMEO:, JULIET:) for dialogue
        - Try famous Shakespeare quotes as prompts
        - Temperature 0.7-0.9 works well for creative text
        - top_k 40-50 provides good diversity
        - Lower temperature (0.5-0.7) for more coherent text
        
        ### Notes:
        - Model trained for 5,050 steps on Shakespeare corpus
        - Uses simple generation (temperature + top-k sampling)
        - Best results with Shakespeare-style prompts
        """)
        
        # Load model on startup
        interface.load(
            fn=load_model,
            inputs=[],
            outputs=[gr.State(), gr.State(), status_box]
        )
    
    return interface


if __name__ == "__main__":
    # Load model first
    print("="*80)
    print("Starting SmolLM2-135M Shakespeare Generator")
    print("="*80)
    
    model, tokenizer, status = load_model()
    
    print("\n" + status)
    
    # Create and launch interface
    print("\n" + "="*80)
    print("Launching Gradio interface...")
    print("="*80)
    
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
