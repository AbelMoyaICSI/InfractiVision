import torch
import sys
import os

# Add LPRNet_Peru to path
sys.path.append(os.path.join(os.getcwd(), 'LPRNet_Peru'))

from data.load_data import CHARS
from model.LPRNet import build_lprnet

state_dict = torch.load('models/LPRNet_Peru_MASTER_FINAL.pth', map_location='cpu')

print("Model Layer Shapes in Weights:")
for k, v in state_dict.items():
    if 'backbone.20.weight' in k:
        print(f"Backbone Layer 20 (logits layer): {v.shape} (Output should be class_num)")
    if 'container.0.weight' in k:
        print(f"Container Layer: {v.shape} (Input should be 448 + class_num, Output should be class_num)")

print(f"Len CHARS from current load_data: {len(CHARS)}")
