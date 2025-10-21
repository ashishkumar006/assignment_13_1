"""
PyTorch Lightning Training Module for SmolLM2-135M

This module implements:
1. LightningModule wrapper for SmolLM2
2. BPE tokenization using official SmolLM2-135M tokenizer (GPT-2 style)
3. Training with prediction every 500 steps
4. Checkpoint saving and loading
5. Mixed precision training (bfloat16)

IMPORTANT: Uses the official SmolLM2-135M tokenizer with 49,152 vocab (BPE)
"""

import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import SmolLM2ForCausalLM, get_model_config
from typing import Optional


class TextDataset(Dataset):
    """Dataset for BPE tokenized language modeling"""
    def __init__(self, text: str, tokenizer: AutoTokenizer, seq_length: int = 256):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize the entire text using BPE
        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"Total tokens: {len(self.tokens):,}")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")
        print(f"Example: First 100 chars -> {len(tokenizer.encode(text[:100], add_special_tokens=False))} tokens")
    
    def __len__(self):
        # Number of sequences we can extract
        return len(self.tokens) - self.seq_length
    
    def __getitem__(self, idx):
        # Get a chunk of tokens
        chunk = self.tokens[idx:idx + self.seq_length + 1]
        
        # Input is all but last token, target is all but first token
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


class SmolLM2Lightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for SmolLM2-135M
    
    Features:
    - Automatic optimization with AdamW
    - Learning rate scheduling with warmup
    - Gradient clipping
    - Mixed precision training
    - Periodic text generation for monitoring
    """
    def __init__(
        self,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 500,
        max_steps: int = 5000,
        generate_every_n_steps: int = 500,
        tokenizer: Optional[AutoTokenizer] = None,
        seq_length: int = 256,
        pretrained_model_name: Optional[str] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        
        # Get SmolLM2-135M config (vocab_size=49152)
        self.config = get_model_config()
        
        # Initialize model from scratch or load pretrained
        if pretrained_model_name:
            print(f"Loading pretrained weights from {pretrained_model_name}...")
            from transformers import AutoModelForCausalLM as HFModel
            hf_model = HFModel.from_pretrained(pretrained_model_name)
            # Copy weights to our model
            self.model = SmolLM2ForCausalLM(self.config)
            self.model.load_state_dict(hf_model.state_dict(), strict=False)
            print("✓ Loaded pretrained weights")
        else:
            # Train from scratch with random initialization
            print("Initializing model from scratch...")
            self.model = SmolLM2ForCausalLM(self.config)
            print("✓ Model initialized with random weights")
        
        # Store tokenizer for generation
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Verify tokenizer vocab matches model
        if tokenizer and tokenizer.vocab_size != self.config['vocab_size']:
            print(f"⚠️  WARNING: Tokenizer vocab ({tokenizer.vocab_size}) != Model vocab ({self.config['vocab_size']})")
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.generate_every_n_steps = generate_every_n_steps
        
        # Track training progress
        self.training_step_count = 0
        
        print(f"📊 Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def forward(self, input_ids, labels=None):
        return self.model(input_ids, labels=labels)
    
    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        
        # Forward pass
        loss, logits = self(input_ids, labels=labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        
        # Increment step counter
        self.training_step_count += 1
        
        # Generate text periodically
        if self.training_step_count % self.generate_every_n_steps == 0:
            self.generate_and_log()
        
        return loss
    
    def generate_and_log(self):
        """Generate sample text and log it"""
        if self.tokenizer is None:
            return
        
        self.model.eval()
        with torch.no_grad():
            # Start with a prompt
            prompt = "First Citizen:\n"
            input_ids = torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=False)], device=self.device)
            
            # Generate
            generated_ids = self.model.generate(
                input_ids,
                max_length=200,
                temperature=0.8,
                top_k=40
            )
            
            # Decode
            generated_text = self.tokenizer.decode(generated_ids[0].cpu().tolist(), skip_special_tokens=True)
            
            # Log
            print("\n" + "="*60)
            print(f"Step {self.training_step_count} - Generated Text:")
            print("="*60)
            print(generated_text)
            print("="*60 + "\n")
        
        self.model.train()
    
    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        loss, logits = self(input_ids, labels=labels)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_perplexity', perplexity, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer with weight decay and learning rate schedule
        
        Uses AdamW with:
        - Weight decay for regularization (except for biases and layer norms)
        - Warmup followed by cosine decay
        """
        # Separate parameters into those that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # Don't apply weight decay to biases and layer norms
            if 'bias' in name or 'norm' in name or 'embed' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
                return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def on_save_checkpoint(self, checkpoint):
        """Add custom data to checkpoint"""
        checkpoint['training_step_count'] = self.training_step_count
        checkpoint['model_config'] = self.config
    
    def on_load_checkpoint(self, checkpoint):
        """Load custom data from checkpoint"""
        self.training_step_count = checkpoint.get('training_step_count', 0)
        print(f"Resumed from step {self.training_step_count}")


def create_dataloaders(
    text: str,
    tokenizer: AutoTokenizer,
    seq_length: int = 256,
    batch_size: int = 32,
    train_split: float = 0.9
):
    """
    Create train and validation dataloaders
    
    Args:
        text: Input text to train on
        tokenizer: Character tokenizer
        seq_length: Sequence length for each sample
        batch_size: Batch size
        train_split: Fraction of data to use for training
    
    Returns:
        train_loader, val_loader
    """
    # Split text into train and validation
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Training characters: {len(train_text):,}")
    print(f"  Validation characters: {len(val_text):,}")
    
    # Create datasets
    train_dataset = TextDataset(train_text, tokenizer, seq_length)
    val_dataset = TextDataset(val_text, tokenizer, seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Colab compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


def load_checkpoint_and_continue(
    checkpoint_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    additional_steps: int = 50
):
    """
    Load a checkpoint and continue training
    
    Args:
        checkpoint_path: Path to checkpoint file
        train_loader: Training dataloader
        val_loader: Validation dataloader
        additional_steps: Number of additional steps to train
    
    Returns:
        Trained model and trainer
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Create new model with same config
    model = SmolLM2Lightning.load_from_checkpoint(checkpoint_path)
    
    print(f"Checkpoint loaded. Resuming from step {model.training_step_count}")
    
    # Create new trainer for additional training
    trainer = pl.Trainer(
        max_steps=model.training_step_count + additional_steps,
        val_check_interval=0.5,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir='./checkpoints_continued',
        enable_progress_bar=True,
    )
    
    print(f"\nContinuing training for {additional_steps} more steps...")
    print(f"Total steps will be: {model.training_step_count + additional_steps}")
    
    return model, trainer


if __name__ == "__main__":
    # Test the Lightning module
    print("Testing Lightning Module...")
    
    # Load official tokenizer
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model = SmolLM2Lightning(
        learning_rate=3e-4,
        tokenizer=tokenizer
    )
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size, seq_length = 2, 128
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    labels = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    
    loss = model.training_step((input_ids, labels), 0)
    print(f"Test training step loss: {loss.item():.4f}")
