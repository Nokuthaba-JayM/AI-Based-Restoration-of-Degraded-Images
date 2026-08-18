#!/usr/bin/env python3
"""
Semiconductor Restoration Inference
Usage: python run.py <input_dir> <output_dir>
"""

import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# ---------- MODEL DEFINITION ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class WideUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(base_channels*4, base_channels*8)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = ResidualBlock(base_channels*8, base_channels*4)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = ResidualBlock(base_channels*4, base_channels*2)
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = ResidualBlock(base_channels*2, base_channels)
        self.pre_shuffle = nn.Conv2d(base_channels, (base_channels//4)*4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.final_conv = nn.Conv2d(base_channels//4, out_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        d3 = self.up3(b); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        out = self.pre_shuffle(d1); out = self.pixel_shuffle(out)
        out = self.final_conv(out); out = self.sigmoid(out)
        return out

# ---------- DATASET ----------
class TestDataset(Dataset):
    def __init__(self, input_dir):
        self.files = sorted(Path(input_dir).glob("*.npy"))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        return torch.from_numpy(arr), self.files[idx].name

# ---------- MAIN ----------
def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir, output_dir = sys.argv[1], sys.argv[2]
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path("outputs/checkpoints/best_unet_progressive.pth")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure training has completed and the file exists.")
        sys.exit(1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = WideUNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset = TestDataset(input_path)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Processing {len(dataset)} images...")
    start = time.time()
    
    with torch.inference_mode():
        for images, names in loader:
            images = images.to(device)
            restored = model(images).cpu().numpy()
            for i, name in enumerate(names):
                np.save(output_path / name, restored[i, 0])
    
    total = time.time() - start
    print(f"Done! {len(dataset)} images in {total:.2f}s ({len(dataset)/total:.2f} img/s)")

if __name__ == "__main__":
    main()