# ... (imports remain the same) ...
from lora_train_dataset import LoRAParquetDataset 
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Repository imports
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion

from lora import loraModel  # Assuming you have a module for LoRA application

import types
from ldm.models.diffusion.ddpm import LatentDiffusion, DDPM
import torch

def custom_get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                     cond_key=None, return_original_cond=False, bs=None):
    """
    Patched get_input that:
    1. Encodes images to latents (Fixes the 3 vs 4 channel error).
    2. Handles text lists correctly (Fixes the list shape error).
    """
    # 1. Get the Raw Image (3 channels)
    x = DDPM.get_input(self, batch, k)
    
    # 2. ENCODE TO LATENTS (Restored this step)
    # We move x to the correct device first
    x = x.to(self.device)
    
    # Run the VAE Encoder
    encoder_posterior = self.encode_first_stage(x)
    z = self.get_first_stage_encoding(encoder_posterior).detach()
    
    # 3. Get the Conditioning (Text)
    if cond_key is None:
        cond_key = self.cond_stage_key

    if cond_key != self.first_stage_key:
        if cond_key == "txt":
            # Just grab the list, don't move to GPU
            xc = batch[cond_key]
        else:
            xc = DDPM.get_input(self, batch, cond_key).to(self.device)
    else:
        xc = None

    # 4. Encode the Conditioning (CLIP)
    if not self.cond_stage_trainable or force_c_encode:
        if isinstance(xc, (dict, list)):
             # Handle list of strings -> CLIP Encoder
             c = self.get_learned_conditioning(xc)
        else:
             c = self.get_learned_conditioning(xc.to(self.device))
    else:
        c = xc
        
    if bs is not None:
        c = c[:bs]
        z = z[:bs]

    return z, c # Return latents (z), not pixels (x)

# Apply the fixed patch
LatentDiffusion.get_input = custom_get_input
print("Global patch updated: Now encodes images to latents AND handles text lists.")


    
def enable_gradient_checkpointing(model):
    """
    Forcefully enables gradient checkpointing on all sub-modules 
    that support it (ResBlocks, AttentionBlocks).
    """
    # 1. Enable on the top-level UNet (if it uses it)
    if hasattr(model, "use_checkpoint"):
        model.use_checkpoint = True

    # 2. Iterate through all children to catch ResBlocks and AttentionBlocks
    for name, module in model.named_modules():
        # The CompVis repo uses 'use_checkpoint' in ResBlocks/AttentionBlocks
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = True
            
        # Some older versions or specific blocks might use just 'checkpoint'
        if hasattr(module, "checkpoint"):
            module.checkpoint = True
            
    print("Gradient checkpointing enabled on all capable modules.")



def train_lora_native(
    config_path, 
    ckpt_path, 
    parquet_url="hf://datasets/huggan/few-shot-obama/data/train-00000-of-00001.parquet", 
    batch_size=1, 
    lr=1e-4, 
    epochs=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Config
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    # --- PATCH 1: Fix Get Input (Text Lists) ---
    model.get_input = types.MethodType(custom_get_input, model)

    # --- PATCH 2: Disable Lightning Logging (THE NEW FIX) ---
    # Prevents "NoneType object has no attribute _results"
    model.log_dict = lambda *args, **kwargs: None
    model.log = lambda *args, **kwargs: None

    # --- PATCH 3: Disable Internal Scheduler (THE NEW FIX) ---
    # Prevents "NoneType object has no attribute lightning_optimizers"
    model.use_scheduler = False

    # --- PATCH 3: Fix CLIP Class Name ---
    import ldm.modules.encoders.modules
    ldm.modules.encoders.modules.FrozenCLIPEmbedder = ldm.modules.encoders.modules.FrozenCLIPTextEmbedder

    # 2. Load Weights
    print(f"Loading weights from {ckpt_path}...")
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model.load_state_dict(sd, strict=False)

    # 3. Inject LoRA
    print("Injecting LoRA adapters...")
    model.model = loraModel(model.model, rank=4)
    model.to(device)

    # --- PATCH 4: Move Schedule Tensors to GPU ---
    schedule_vars = [
        "logvar", "betas", "alphas_cumprod", "alphas_cumprod_prev", 
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod", 
        "posterior_variance", "posterior_log_variance_clipped", 
        "posterior_mean_coef1", "posterior_mean_coef2"
    ]
    for var_name in schedule_vars:
        if hasattr(model, var_name):
            attr = getattr(model, var_name)
            if isinstance(attr, torch.Tensor):
                setattr(model, var_name, attr.to(device))
    
    # 4. Freeze & Optimizer
    model.first_stage_model.eval()
    if model.cond_stage_model:
        model.cond_stage_model.eval()
    
    enable_gradient_checkpointing(model.model)
    model.requires_grad_(False)
    model.model.set_trainable_parameters()
    
    trainable_params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    # 5. Dataset
    dataset = LoRAParquetDataset(parquet_url, size=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 6. Training Loop
    model.train()
    print("Starting Training Loop...")
    
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            # Move tensors to GPU
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            with torch.cuda.amp.autocast():
                # OLD: loss, loss_dict = model.training_step(batch, batch_idx=step)
                
                # NEW: Call shared_step directly. 
                # It returns (loss, dict) and skips the Lightning overhead.
                loss, loss_dict = model.shared_step(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    torch.save(model.model.state_dict(), "lora_finetuned.pt")
    print("Training complete. Saved to lora_finetuned.pt")