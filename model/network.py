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
from torchvision.transforms.functional import to_pil_image


def _ensure_hf_weights(repo_id, model_file_name, weights_dir_path, display_name, force_download=False):
    assert weights_dir_path is not None, f"Please provide the path to the {display_name} weights directory."
    assert repo_id is not None, f"Please provide the Hugging Face repository id for {display_name} weights."

    os.makedirs(weights_dir_path, exist_ok=True)
    model_weights_path = os.path.join(weights_dir_path, model_file_name)

    if force_download or not os.path.exists(model_weights_path):
        hf_hub_download(repo_id, filename=model_file_name, local_dir=weights_dir_path, force_download=True)
        print(f"{display_name} weights downloaded to {weights_dir_path}")

    assert os.path.isfile(model_weights_path), f"{display_name} weights not found at {model_weights_path}"
    return model_weights_path


def _load_state_dict(model_weights_path):
    state_dict = torch.load(model_weights_path, map_location="cpu")

    if isinstance(state_dict, dict):
        if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]

    return state_dict


def _create_timm_model(repo_id, state_dict, timm_model_name=None, strict=True, model_kwargs=None):
    model_kwargs = model_kwargs or {}
    model_identifier = timm_model_name or f"hf_hub:{repo_id}"
    model = timm.create_model(model_identifier, pretrained=False, **model_kwargs)
    model.load_state_dict(state_dict, strict=strict)
    return model


def _create_phikon_model(repo_id, cache_dir=None, force_download=False):
    cache_kwargs = {}
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_kwargs["cache_dir"] = cache_dir
    if force_download:
        cache_kwargs["force_download"] = True

    processor = transformers.AutoImageProcessor.from_pretrained(repo_id, **cache_kwargs)
    model = transformers.AutoModel.from_pretrained(repo_id, **cache_kwargs)
    model.eval()
    return model, processor


def _create_conch_model(model_name, checkpoint_path):
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as exc:
        raise ImportError(
            "CONCH_TileEncoder requires the `conch` package. "
            "Please install it following the official CONCH instructions."
        ) from exc

    model, preprocess = create_model_from_pretrained(model_name, checkpoint_path)
    model.eval()
    return model, preprocess


def _compress_features(features, pool_layer, norm_layer, norm_out, target_dim, apply_norm_if_same=False):
    if features.dim() == 3:
        features = features.mean(dim=1)
    elif features.dim() > 3:
        features = features.flatten(start_dim=1)

    needs_pool = features.shape[-1] != target_dim
    if needs_pool:
        features = pool_layer(features.unsqueeze(-1)).squeeze(-1)
    if norm_out and (needs_pool or apply_norm_if_same):
        features = norm_layer(features)
    return features


class UNI_TileEncoder(nn.Module):
    """
    One of the baselines
    Pathology-specific encoder based on UNI, squeeze the tile encoding to a very small size
        with linear mapping of fixed parameters (like 1.0)
    UNI weights and model loading code: https://huggingface.co/MahmoodLab/UNI
    """

    def __init__(
        self,
        weights_dir_path,
        output_dim=256,
        norm_out=True,
        repo_id="MahmoodLab/UNI",
        model_file_name="pytorch_model.bin",
        timm_model_name="vit_large_patch16_224",
        strict=True,
        model_kwargs=None,
    ):
        """
        Initializes the UNI_TileEncoder.

        Args:
            weights_dir_path (str): Local directory to cache/download uni weights.
            output_dim (int): The desired output dimension after encoding.
            norm_out (bool): Whether to apply LayerNorm after projection.
        """
        super(UNI_TileEncoder, self).__init__()

        default_model_kwargs = {
            "img_size": 224,
            "patch_size": 16,
            "init_values": 1e-5,
            "num_classes": 0,
            "dynamic_img_size": True,
        }
        effective_model_kwargs = dict(default_model_kwargs)
        if model_kwargs:
            effective_model_kwargs.update(model_kwargs)

        model_weights_path = _ensure_hf_weights(
            repo_id=repo_id,
            model_file_name=model_file_name,
            weights_dir_path=weights_dir_path,
            display_name="UNI",
        )
        state_dict = _load_state_dict(model_weights_path)
        self.feature_extractor = _create_timm_model(
            repo_id=repo_id,
            state_dict=state_dict,
            timm_model_name=timm_model_name,
            strict=strict,
            model_kwargs=effective_model_kwargs,
        )

        self.output_dim = output_dim
        self.norm_out = norm_out
        self.pool = nn.AdaptiveAvgPool1d(self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        """
        Performs forward pass through the UNI_TileEncoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: The encoded tensor of shape (batch_size, self.output_dim).
        """
        features = self.feature_extractor(x)
        return _compress_features(
            features,
            self.pool,
            self.norm,
            self.norm_out,
            self.output_dim,
            apply_norm_if_same=False,
        )


class ProvGigaPath_TileEncoder(nn.Module):
    """
    One of the baselines
    Pathology-specific encoder based on Prov-GigaPath, squeeze the tile encoding to a very small size
        with linear mapping of fixed parameters (like 1.0)
    Prov-GigaPath weights and model loading code: https://huggingface.co/prov-gigapath/prov-gigapath

    Important: timm version needs to be > 0.9.8 to load the model correctly (there was a bug in timm 0.9.8)
    """

    def __init__(
        self,
        weights_dir_path,
        output_dim=256,
        norm_out=True,
        repo_id="prov-gigapath/prov-gigapath",
        model_file_name="pytorch_model.bin",
        timm_model_name=None,
        strict=True,
        model_kwargs=None,
    ):
        """
        Initializes the Prov-GigaPath_TileEncoder.

        Args:
            weights_dir_path (str): Local directory to cache/download Prov-GigaPath weights.
            output_dim (int): The desired output dimension after encoding.
            norm_out (bool): Whether to apply LayerNorm after projection.
        """
        super(ProvGigaPath_TileEncoder, self).__init__()
        # login()  # might need to do it the very first time so save the token

        print(f"Please ensure you have one of the newer versions of `timm`. Your version is {timm.__version__}. There is a bug when Prov-GigaPath is used with `timm==0.9.8`. Tested to work from `timm>=1.0.3`")
        assert timm.__version__ > '0.9.8', "Please update timm to version > 0.9.8, there was a bug when Prov-GigaPath is used with version 0.9.8"

        model_weights_path = _ensure_hf_weights(
            repo_id=repo_id,
            model_file_name=model_file_name,
            weights_dir_path=weights_dir_path,
            display_name="Prov-GigaPath",
        )
        state_dict = _load_state_dict(model_weights_path)
        self.feature_extractor = _create_timm_model(
            repo_id=repo_id,
            state_dict=state_dict,
            timm_model_name=timm_model_name,
            strict=strict,
            model_kwargs=model_kwargs,
        )

        self.output_dim = output_dim
        self.norm_out = norm_out
        self.pool = nn.AdaptiveAvgPool1d(self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        """
        Performs forward pass through the Prov-GigaPath_TileEncoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: The encoded tensor of shape (batch_size, self.output_dim).
        """
        features = self.feature_extractor(x)
        return _compress_features(
            features,
            self.pool,
            self.norm,
            self.norm_out,
            self.output_dim,
            apply_norm_if_same=False,
        )


class PhikonV2_TileEncoder(nn.Module):
    """
    Pathology foundation encoder based on Phikon-v2.
    Automatically downloads the checkpoint from Hugging Face if needed and
    compresses features to a configurable dimensionality.
    """

    def __init__(
        self,
        weights_dir_path,
        output_dim=256,
        norm_out=True,
        repo_id="PaigeAI/Phikon-v2",
        model_file_name="pytorch_model.bin",
        timm_model_name=None,
        strict=True,
    ):
        super(PhikonV2_TileEncoder, self).__init__()

        os.makedirs(weights_dir_path, exist_ok=True)
        self.feature_extractor, self.processor = _create_phikon_model(
            repo_id=repo_id,
            cache_dir=weights_dir_path,
        )

        self.output_dim = output_dim
        self.norm_out = norm_out
        self.pool = nn.AdaptiveAvgPool1d(self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        device = x.device if isinstance(x, torch.Tensor) else next(self.feature_extractor.parameters()).device
        images = self._as_image_list(x)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.feature_extractor(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
        return _compress_features(features, self.pool, self.norm, self.norm_out, self.output_dim, apply_norm_if_same=True)

    @staticmethod
    def _as_image_list(x):
        if isinstance(x, torch.Tensor):
            if x.dim() == 3:
                x = x.unsqueeze(0)
            return [img.detach().cpu() for img in x]

        if isinstance(x, (list, tuple)):
            images = []
            for item in x:
                if isinstance(item, torch.Tensor):
                    if item.dim() == 3:
                        images.append(item.detach().cpu())
                    elif item.dim() == 4:
                        images.extend(img.detach().cpu() for img in item)
                    else:
                        raise ValueError(f"Unsupported tensor shape for PhikonV2_TileEncoder input: {item.shape}")
                else:
                    images.append(item)
            return images

        return [x]


class CONCH_TileEncoder(nn.Module):
    """
    Pathology foundation encoder based on CONCH.
    Downloads weights from Hugging Face if missing and applies dimensionality compression.
    """

    def __init__(
        self,
        weights_dir_path,
        output_dim=256,
        norm_out=True,
        repo_id="TencentARC/CONCH",
        model_file_name="pytorch_model.bin",
        timm_model_name=None,
        strict=True,
    ):
        super(CONCH_TileEncoder, self).__init__()

        model_weights_path = _ensure_hf_weights(
            repo_id=repo_id,
            model_file_name=model_file_name,
            weights_dir_path=weights_dir_path,
            display_name="CONCH",
        )
        model_name = timm_model_name or "conch_ViT-B-16"
        self.feature_extractor, self.preprocess = _create_conch_model(model_name, model_weights_path)

        self.output_dim = output_dim
        self.norm_out = norm_out
        self.pool = nn.AdaptiveAvgPool1d(self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        device = x.device if isinstance(x, torch.Tensor) else next(self.feature_extractor.parameters()).device
        pil_images = self._as_pil_list(x)
        processed = torch.stack([self.preprocess(image) for image in pil_images])
        processed = processed.to(device)

        with torch.inference_mode():
            features = self.feature_extractor.encode_image(processed)
        return _compress_features(features, self.pool, self.norm, self.norm_out, self.output_dim, apply_norm_if_same=True)

    @staticmethod
    def _as_pil_list(x):
        def _tensor_to_pil(tensor):
            if tensor.dim() != 3:
                raise ValueError(f"Unsupported tensor shape for CONCH_TileEncoder input: {tensor.shape}")
            return to_pil_image(tensor.detach().cpu())

        if isinstance(x, torch.Tensor):
            if x.dim() == 3:
                return [_tensor_to_pil(x)]
            if x.dim() == 4:
                return [_tensor_to_pil(img) for img in x]
            raise ValueError(f"Unsupported tensor shape for CONCH_TileEncoder input: {x.shape}")

        if isinstance(x, (list, tuple)):
            images = []
            for item in x:
                if isinstance(item, torch.Tensor):
                    if item.dim() == 3:
                        images.append(_tensor_to_pil(item))
                    elif item.dim() == 4:
                        images.extend(_tensor_to_pil(img) for img in item)
                    else:
                        raise ValueError(f"Unsupported tensor shape for CONCH_TileEncoder input: {item.shape}")
                else:
                    images.append(item)
            return images

        return [x]


class Virchow2_TileEncoder(nn.Module):
    """
    Pathology foundation encoder based on Virchow2.
    Handles automated weight download and optional feature compression.
    """

    def __init__(
        self,
        weights_dir_path,
        output_dim=256,
        norm_out=True,
        repo_id="MSLab/Virchow2",
        model_file_name="pytorch_model.bin",
        timm_model_name=None,
        strict=True,
    ):
        super(Virchow2_TileEncoder, self).__init__()

        model_weights_path = _ensure_hf_weights(
            repo_id=repo_id,
            model_file_name=model_file_name,
            weights_dir_path=weights_dir_path,
            display_name="Virchow2",
        )
        state_dict = _load_state_dict(model_weights_path)
        self.feature_extractor = _create_timm_model(repo_id, state_dict, timm_model_name=timm_model_name, strict=strict)

        self.output_dim = output_dim
        self.norm_out = norm_out
        self.pool = nn.AdaptiveAvgPool1d(self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        return _compress_features(features, self.pool, self.norm, self.norm_out, self.output_dim, apply_norm_if_same=True)


class MStar_TileEncoder(nn.Module):
    """
    Pathology foundation encoder based on mSTAR.
    Supports automatic checkpoint retrieval and feature dimensionality reduction.
    """

    def __init__(
        self,
        weights_dir_path,
        output_dim=256,
        norm_out=True,
        repo_id="medloader/mSTAR",
        model_file_name="pytorch_model.bin",
        timm_model_name=None,
        strict=True,
    ):
        super(MStar_TileEncoder, self).__init__()

        model_weights_path = _ensure_hf_weights(
            repo_id=repo_id,
            model_file_name=model_file_name,
            weights_dir_path=weights_dir_path,
            display_name="mSTAR",
        )
        state_dict = _load_state_dict(model_weights_path)
        self.feature_extractor = _create_timm_model(repo_id, state_dict, timm_model_name=timm_model_name, strict=strict)

        self.output_dim = output_dim
        self.norm_out = norm_out
        self.pool = nn.AdaptiveAvgPool1d(self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        return _compress_features(features, self.pool, self.norm, self.norm_out, self.output_dim, apply_norm_if_same=True)


if __name__ == '__main__':
    pass
