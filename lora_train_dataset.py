import io
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LoRAParquetDataset(Dataset):
    def __init__(self, parquet_url, size=512):
        print(f"Loading dataset from {parquet_url}...")
        self.df = pd.read_parquet(parquet_url)
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bytes = row['image']['bytes']
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        pixel_values = self.transform(image)
        # Permute to (H, W, C) as we fixed earlier
        pixel_values = pixel_values.permute(1, 2, 0)
        
        return {
            "jpg": pixel_values,
            # --- THE FIX: Add the "txt" key ---
            "txt": "a photo of Barack Obama" 
        }