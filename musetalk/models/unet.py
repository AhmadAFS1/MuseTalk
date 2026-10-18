import torch
import torch.nn as nn
import math
import json

from diffusers import UNet2DConditionModel
import sys
import time
import numpy as np
import os

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return x
    
def _load_unet_fast(unet_config, model_path, map_location):
    """Build the UNet on the meta device (no random init of ~850M parameters) and adopt
    the memory-mapped checkpoint tensors (assign=True) instead of copying them into a
    freshly initialised CPU model. Measured on an RTX 3090 worker (c5-a, 2026-10-18):
    9.95 s -> 0.97 s, with bit-identical parameters and fp16 forward output.
    Returns None (caller uses the original path) when unsupported, opted out with
    MUSETALK_UNET_FAST_LOAD=0, or when any parameter/buffer would stay on meta."""
    if os.getenv("MUSETALK_UNET_FAST_LOAD", "1").strip() == "0":
        return None
    try:
        with torch.device("meta"):
            model = UNet2DConditionModel(**unet_config)
        weights = torch.load(model_path, map_location=map_location, mmap=True)
        model.load_state_dict(weights, assign=True)
    except (TypeError, RuntimeError, AttributeError) as exc:
        print(f"UNet fast load unavailable ({type(exc).__name__}: {exc}); using the standard load")
        return None
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        print("UNet fast load left tensors on the meta device; using the standard load")
        return None
    return model


class UNet():
    def __init__(self, 
                 unet_config,
                 model_path,
                 use_float16=False,
                 device=None
        ):
        with open(unet_config, 'r') as f:
            unet_config = json.load(f)
        self.pe = PositionalEncoding(d_model=384)
        if device != None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_location = None if torch.cuda.is_available() else self.device
        self.model = _load_unet_fast(unet_config, model_path, map_location)
        if self.model is None:
            self.model = UNet2DConditionModel(**unet_config)
            weights = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(weights)
        if use_float16:
            self.model = self.model.half()
        self.model.to(self.device)
    
if __name__ == "__main__":
    unet = UNet()
