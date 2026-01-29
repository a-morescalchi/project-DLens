from xml.parsers.expat import model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import os

# Imports from the latent-diffusion repo
from ldm.util import instantiate_from_config

# Import our dataset
from lora_train_dataset import LoRADataset
from lora import loraModel

# --- Configuration ---
CONFIG_PATH = "configs/latent-diffusion/celebahq-ldm-vq-4.yaml" # Check this path!
CKPT_PATH = "celeba/model.ckpt" # Check this path!
OUTPUT_DIR = "lora_checkpoints"
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_URL = "hf://datasets/huggan/few-shot-obama/data/train-00000-of-00001.parquet"

def load_model(config_path, ckpt_path):
    # 1. Load the YAML configuration
    config = OmegaConf.load(config_path)
    
    # --- FIX START ---
    # The config file points to a separate VQ-VAE file (models/first_stage_models/...)
    # which you don't have. We delete this key so the code doesn't try to load it yet.
    # The VAE weights will be loaded from your main 'ckpt_path' in step 3.
    if "first_stage_config" in config.model.params:
        if "ckpt_path" in config.model.params.first_stage_config.params:
            print("Patched config: Removing reference to missing first_stage_model.")
            del config.model.params.first_stage_config.params["ckpt_path"]
    # --- FIX END ---

    # 2. Instantiate the model (now with initialized, random VAE weights)
    model = instantiate_from_config(config.model)
    
    # 3. Load the actual pre-trained weights (which contain both UNet and VAE)
    print(f"Loading weights from {ckpt_path}...")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
        
    # strict=False allows loading even if there are small mismatches
    # (The main checkpoint keys will overwrite the random VAE weights)
    m, u = model.load_state_dict(sd, strict=False)
    
    if len(m) > 0:
        print(f"Missing keys: {len(m)}")
    if len(u) > 0:
        print(f"Unexpected keys: {len(u)}")
        
    model.to(DEVICE)
    model.eval()
    return model

def train(BATCH_SIZE = 4, LR = 1e-4, EPOCHS = 100):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load the Freeze the Base Model
    model = load_model(CONFIG_PATH, CKPT_PATH)
    
    # In CompVis/latent-diffusion, the UNet is typically at model.model.diffusion_model
    unet = model.model.diffusion_model
    
    # Freeze everything first
    #for param in model.parameters():
    #    param.requires_grad = False

    # 2. Inject Your Custom LoRA
    # ======================================================
    # ??? INSERT YOUR LORA INJECTION HERE ???
    # Example: 
    # from my_lora_implementation import inject_lora
    # inject_lora(unet, r=4) 

    unet = loraModel(unet, rank=16, alpha=16, qkv=[True, True, True])
    unet.to(DEVICE)
    unet.set_trainable_parameters()

    model.model.diffusion_model = unet
    print("Checking if LoRA is in the computation graph...")
    print(f"model.model.diffusion_model type: {type(model.model.diffusion_model)}")
    print(f"Is it loraModel? {isinstance(model.model.diffusion_model, loraModel)}")
    

    # ensure only LoRA parameters have requires_grad=True
    # ======================================================
    
    # Verify we have trainable parameters
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")
    if len(trainable_params) == 0:
        print("WARNING: No trainable parameters found. Did you apply the LoRA?")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-2)
    initial_weights = {}
    for name, param in unet.named_parameters():
        if 'lora' in name and param.requires_grad:
            initial_weights[name] = param.data.clone()

    # 3. Dataset
    dataset = LoRADataset(parquet_url=DATASET_URL, size=256)
    
    # Check if dataset loaded correctly
    print(f"Dataset loaded: {len(dataset)} images found.")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 4. Training Loop
    unet.train()  # Instead of model.train()
    model.eval() 
    
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x in pbar:
            x = x.to(DEVICE)
            
            # A. Encode Images to Latent Space
            # The model handles the VQGAN encoding internally via encode_first_stage
            with torch.no_grad():
                z = model.get_first_stage_encoding(model.encode_first_stage(x))
            
            # B. Sample Noise & Timesteps
            t = torch.randint(0, model.num_timesteps, (x.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(z)
            
            # C. Add Noise (Forward Diffusion)
            # q_sample is the method in LDM to add noise at step t
            x_noisy = model.q_sample(x_start=z, t=t, noise=noise)
            
            # D. Predict Noise
            # apply_model runs the UNet
            # Note: LDM CelebA is unconditional, so second argument (context) is usually None
            model_output = model.apply_model(x_noisy, t, cond=None)
            
            # E. Calculate Loss
            # The target depends on the prediction type (epsilon or v), but usually it's noise
            loss = F.mse_loss(model_output, noise)

            # F. Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            grad_norms = [p.grad.norm().item() for p in trainable_params if p.grad is not None]
            if len(grad_norms) > 0:
                avg_grad = sum(grad_norms) / len(grad_norms)
                pbar.set_postfix(loss=loss.item(), grad=f"{avg_grad:.4f}")
            else:
                print("WARNING: No gradients!")

            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())

        # Save LoRA weights only
        # You'll need to write logic to save ONLY your lora layers, not the whole model
        torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, f"lora_epoch_{epoch}.pt"))
        if epoch % 10 == 0:
            weight_changes = []
            for name, param in unet.named_parameters():
                if 'lora' in name and param.requires_grad:
                    change = (param.data - initial_weights[name]).abs().mean().item()
                    weight_changes.append(change)
            avg_change = sum(weight_changes) / len(weight_changes) if weight_changes else 0
            print(f"Epoch {epoch}: Average LoRA weight change: {avg_change:.6f}")
            
    avg_change = sum(weight_changes) / len(weight_changes) if weight_changes else 0
    print(f"Epoch {epoch}: Average LoRA weight change: {avg_change:.6f}")
    torch.save(unet.state_dict(), os.path.join(OUTPUT_DIR, f"lora_final.pt"))
    print('Weights saved in ', OUTPUT_DIR+"/lora_final.pt")

if __name__ == "__main__":
    train()