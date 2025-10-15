'''
Created on 15 Oct 2025

@author: yang hu
'''

import argparse
import os

from einops.einops import rearrange
from huggingface_hub import login, hf_hub_download
import timm
import torch

import transformers

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class UNI_TileEncoder(nn.Module):
    """
    One of the baselines
    Pathology-specific encoder based on UNI, squeeze the tile encoding to a very small size
        with linear mapping of fixed parameters (like 1.0)
    UNI weights and model loading code: https://huggingface.co/MahmoodLab/UNI
    """

    def __init__(self, weights_dir_path, output_dim=256, norm_out=True):
        """
        Initializes the UNI_TileEncoder.

        Args:
            weights_dir_path (str): The path to the directory of the UNI weights (path for downloading at local).
            output_dim (int): The desired output dimension after encoding. Default is 64.
        """
        super(UNI_TileEncoder, self).__init__()

        assert weights_dir_path is not None, "Please provide the path to the UNI weights directory."
        model_file_name = "pytorch_model.bin"
        model_weights_path = os.path.join(weights_dir_path, model_file_name)

        # Download the UNI weights if the directory does not exist
        if not os.path.exists(model_weights_path):
            # create directory if it does not exist
            os.makedirs(weights_dir_path, exist_ok=True)

            # login()  # might need to do it the very first time so save the token
            hf_hub_download("MahmoodLab/UNI", filename=model_file_name,
                            local_dir=weights_dir_path, force_download=True)
            print(f"UNI weights downloaded to {weights_dir_path}")
        
        assert os.path.isfile(model_weights_path), f"UNI weights not found at {model_weights_path}"

        # create model
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        # load weights
        model.load_state_dict(
            torch.load(
                os.path.join(model_weights_path),
                map_location="cpu"
            ),
            strict=True
        )
        self.feature_extractor = model

        # Add an adaptive average pooling layer to reduce dimensions to (output_dim, 1)
        self.output_dim = output_dim
        self.pool = nn.AdaptiveAvgPool1d(self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)  # add LayerNorm
        self.norm_out = norm_out

    def forward(self, x):
        """
        Performs forward pass through the UNI_TileEncoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: The encoded tensor of shape (batch_size, self.output_dim).
        """
        x = self.feature_extractor(x)  # shape: (batch_size, 1024)
        if x.shape[-1] > self.output_dim:
            # Apply adaptive average pooling
            x = self.pool(x)  # shape: (batch_size, output_dim)
            if self.norm_out:
                x = self.norm(x)  # shape: (batch_size, output_dim)

        return x


class ProvGigaPath_TileEncoder(nn.Module):
    """
    One of the baselines
    Pathology-specific encoder based on Prov-GigaPath, squeeze the tile encoding to a very small size
        with linear mapping of fixed parameters (like 1.0)
    Prov-GigaPath weights and model loading code: https://huggingface.co/prov-gigapath/prov-gigapath

    Important: timm version needs to be > 0.9.8 to load the model correctly (there was a bug in timm 0.9.8)
    """

    def __init__(self, weights_dir_path, output_dim=256, norm_out=True):
        """
        Initializes the Prov-GigaPath_TileEncoder.

        Args:
            weights_dir_path (str): The path to the directory of the Prov-GigaPath weights.
            output_dim (int): The desired output dimension after encoding. Default is 64.
        """
        super(ProvGigaPath_TileEncoder, self).__init__()
        # login()  # might need to do it the very first time so save the token

        print(f"Please ensure you have one of the newer versions of `timm`. Your version is {timm.__version__}. There is a bug when Prov-GigaPath is used with `timm==0.9.8`. Tested to work from `timm>=1.0.3`")
        assert timm.__version__ > '0.9.8', "Please update timm to version > 0.9.8, there was a bug when Prov-GigaPath is used with version 0.9.8"

        assert weights_dir_path is not None, "Please provide the path to the Prov-GigaPath weights directory."
        model_file_name = "pytorch_model.bin" # name of GigaPath tile encoder
        model_weights_path = os.path.join(weights_dir_path, model_file_name)

        # Download the UNI weights if the directory does not exist
        if not os.path.exists(model_weights_path):
            # create directory if it does not exist
            os.makedirs(weights_dir_path, exist_ok=True)

            hf_hub_download("prov-gigapath/prov-gigapath", filename=model_file_name,
                            local_dir=weights_dir_path, force_download=True)
            print(f"Prov-GigaPath weights downloaded to {weights_dir_path}")
        
        assert os.path.isfile(model_weights_path), f"Prov-GigaPath weights not found at {model_weights_path}"

        # create model
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
        # load weights
        model.load_state_dict(
            torch.load(
                os.path.join(model_weights_path),
                map_location="cpu"
            ),
            strict=True
        )
        self.feature_extractor = model

        # Add an adaptive average pooling layer to reduce dimensions to (output_dim, 1)
        self.output_dim = output_dim
        self.pool = nn.AdaptiveAvgPool1d(self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)  # add LayerNorm
        self.norm_out = norm_out

    def forward(self, x):
        """
        Performs forward pass through the UNI_TileEncoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: The encoded tensor of shape (batch_size, self.output_dim).
        """
        x = self.feature_extractor(x)  # shape: (batch_size, 1536)
        if x.shape[-1] > self.output_dim:
            # Apply adaptive average pooling
            x = self.pool(x)  # shape: (batch_size, self.output_dim)
            if self.norm_out:
                x = self.norm(x)  # shape: (batch_size, output_dim)

        return x


if __name__ == '__main__':
    pass