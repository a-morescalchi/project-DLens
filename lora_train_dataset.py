import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import io

class LoRADataset(Dataset):
    def __init__(self, parquet_url, size=256):
        self.size = size
        
        print(f"Loading dataset from {parquet_url}...")
        # pandas can read directly from hf:// URLs if pyarrow is installed
        self.df = pd.read_parquet(parquet_url)
        
        # Standard LDM transforms
        self.transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get the row
        row = self.df.iloc[idx]
        
        # 2. Extract image bytes
        # Hugging Face image datasets in Parquet format usually store images 
        # as a dictionary/struct with a 'bytes' key.
        image_data = row['image'] 
        
        if isinstance(image_data, dict) and 'bytes' in image_data:
            img_bytes = image_data['bytes']
        elif isinstance(image_data, bytes):
            img_bytes = image_data
        else:
            raise ValueError(f"Unknown image format at index {idx}")

        # 3. Convert bytes to PIL Image
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # 4. Transform to Tensor
        return self.transforms(image)