import torch
import torch.nn as nn
from utils import check_state_dict
from config import Config
from torchvision.ops.deform_conv import DeformConv2d
import deform_conv2d_onnx_exporter

# BiRefNet-DIS 모델 정의 (기존 코드 import)
from models.birefnet import BiRefNet  # <== 모델 클래스 맞게 import 하세요

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set export_onnx flag
config = Config()
config.export_onnx = True

# Register deform_conv2d operator for ONNX export
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

# Initialize model without loading weights
birefnet = BiRefNet(bb_pretrained=False)

# Load BiRefNet-DIS weights
weights_path = 'BiRefNet-DIS-epoch_590.pth'
state_dict = torch.load(weights_path, map_location=device)
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)

# Convert model to float16
birefnet = birefnet.half()
birefnet.to(device)
evaled_birefnet = birefnet.eval()

# Set input size to 256x256 with float16
dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float16).to(device)

# Export
torch.onnx.export(
    evaled_birefnet,
    dummy_input,
    "BiRefNet-DIS-epoch_590_512_float16.onnx",
    export_params=True,
    opset_version=17,  # Increased opset version for better compatibility
    do_constant_folding=True,
    input_names=['input_image'],
    output_names=['output'],
    dynamic_axes={
        'input_image': {2: 'height', 3: 'width'},  # H, W dynamic
        'output': {2: 'height', 3: 'width'}        # output 도 dynamic
    }
)

print("✅ Dynamic ONNX export complete!")
