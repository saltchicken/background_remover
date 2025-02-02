import argparse
from external.BEN2 import BEN2
from PIL import Image
import torch
import os
from pathlib import Path

# Set up argument parser
parser = argparse.ArgumentParser(description='Process an image with BEN2.')
parser.add_argument('input_image', type=str, help='Path to the input image')
parser.add_argument('--checkpoint', type=str, default='./BEN2_Base.pth', help='Path to the model checkpoint')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = BEN2.BEN_Base().to(device).eval()
script_dir = os.path.dirname(os.path.realpath(__file__))
print(script_dir)
checkpoint_path = os.path.join(script_dir, "external", "BEN2", args.checkpoint)
print(script_dir)
model.loadcheckpoints(checkpoint_path)

# Process image
image = Image.open(args.input_image)
foreground = model.inference(image, refine_foreground=False)

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.basename(args.input_image)
output_file = Path(output_file).with_suffix(".png")
output_path = os.path.join(output_dir, output_file)
foreground.save(output_path)

print(f"Foreground saved to {output_path}")
