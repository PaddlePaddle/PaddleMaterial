# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
STEM Image Dataset for SFIN model.
Supports both BF (Bright Field) and HAADF (High-Angle Annular Dark Field) modes.
"""

import os
import os.path as osp
from typing import Callable, Optional, Tuple, List, Union

import numpy as np
import paddle
from paddle.io import Dataset
from PIL import Image


class STEMImageDataset(Dataset):
    """
    STEM Image Dataset for image enhancement task.
    
    This dataset loads pairs of noisy and ground truth (GT) images for training
    and validation. Supports both BF and HAADF STEM imaging modes.
    
    Args:
        noisy_dir (str): Directory containing noisy images
        gt_dir (str): Directory containing ground truth images
        mode (str): STEM imaging mode, 'bf' or 'haadf' (default: 'haadf')
        transform (Optional[Callable]): Optional transform to be applied on images
        image_size (Optional[Tuple[int, int]]): Target image size (H, W). If None, use original size.
    
    Example:
        >>> dataset = STEMImageDataset(
        ...     noisy_dir='./data/haadf_data_test/noisy',
        ...     gt_dir='./data/haadf_data_test/gt_enhance',
        ...     mode='haadf'
        ... )
        >>> noisy_img, gt_img = dataset[0]
    """

    def __init__(
        self,
        noisy_dir: str,
        gt_dir: str,
        mode: str = 'haadf',
        transform: Optional[Callable] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        super(STEMImageDataset, self).__init__()
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.mode = mode.lower()
        self.transform = transform
        self.image_size = image_size

        # Validate mode
        if self.mode not in ['bf', 'haadf']:
            raise ValueError(f"mode must be 'bf' or 'haadf', got {self.mode}")

        # Validate directories
        if not osp.exists(noisy_dir):
            raise FileNotFoundError(f"Noisy directory not found: {noisy_dir}")
        if not osp.exists(gt_dir):
            raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

        # Get list of image files
        self.image_files = self._get_image_files(noisy_dir)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {noisy_dir}")

        print(f"Loaded {len(self.image_files)} image pairs from {mode} mode")

    def _get_image_files(self, directory: str) -> List[str]:
        """Get list of image files from directory."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        image_files = []
        
        for filename in sorted(os.listdir(directory)):
            ext = osp.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_files.append(filename)
        
        return image_files

    def _load_image(self, filepath: str) -> np.ndarray:
        """Load image from file and convert to numpy array."""
        img = Image.open(filepath)
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        img_array = np.array(img, dtype=np.float32)
        
        # Resize if specified
        if self.image_size is not None:
            img_pil = Image.fromarray(img_array.astype(np.uint8))
            img_pil = img_pil.resize(self.image_size[::-1], Image.BICUBIC)
            img_array = np.array(img_pil, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        return img_array

    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Get a pair of noisy and ground truth images.
        
        Args:
            idx: Index of the image pair
        
        Returns:
            Tuple of (noisy_image, gt_image) as paddle.Tensor with shape (C, H, W)
        """
        filename = self.image_files[idx]
        
        # Load images
        noisy_path = osp.join(self.noisy_dir, filename)
        gt_path = osp.join(self.gt_dir, filename)
        
        if not osp.exists(gt_path):
            raise FileNotFoundError(f"Ground truth image not found: {gt_path}")
        
        noisy_img = self._load_image(noisy_path)
        gt_img = self._load_image(gt_path)
        
        # Apply transforms if provided
        if self.transform is not None:
            noisy_img = self.transform(noisy_img)
            gt_img = self.transform(gt_img)
        
        # Add channel dimension and convert to tensor
        noisy_tensor = paddle.to_tensor(noisy_img).unsqueeze(0)  # (1, H, W)
        gt_tensor = paddle.to_tensor(gt_img).unsqueeze(0)  # (1, H, W)
        
        return noisy_tensor, gt_tensor


class STEMImageTestDataset(Dataset):
    """
    STEM Image Test Dataset for inference.
    
    This dataset loads only noisy images for testing/inference.
    
    Args:
        noisy_dir (str): Directory containing noisy images
        mode (str): STEM imaging mode, 'bf' or 'haadf' (default: 'haadf')
        transform (Optional[Callable]): Optional transform to be applied on images
        image_size (Optional[Tuple[int, int]]): Target image size (H, W). If None, use original size.
    
    Example:
        >>> dataset = STEMImageTestDataset(
        ...     noisy_dir='./data/haadf_data_test/noisy',
        ...     mode='haadf'
        ... )
        >>> noisy_img, filename = dataset[0]
    """

    def __init__(
        self,
        noisy_dir: str,
        mode: str = 'haadf',
        transform: Optional[Callable] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        super(STEMImageTestDataset, self).__init__()
        self.noisy_dir = noisy_dir
        self.mode = mode.lower()
        self.transform = transform
        self.image_size = image_size

        if self.mode not in ['bf', 'haadf']:
            raise ValueError(f"mode must be 'bf' or 'haadf', got {self.mode}")

        if not osp.exists(noisy_dir):
            raise FileNotFoundError(f"Noisy directory not found: {noisy_dir}")

        self.image_files = self._get_image_files(noisy_dir)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {noisy_dir}")

        print(f"Loaded {len(self.image_files)} test images from {mode} mode")

    def _get_image_files(self, directory: str) -> List[str]:
        """Get list of image files from directory."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        image_files = []
        
        for filename in sorted(os.listdir(directory)):
            ext = osp.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_files.append(filename)
        
        return image_files

    def _load_image(self, filepath: str) -> np.ndarray:
        """Load image from file and convert to numpy array."""
        img = Image.open(filepath)
        
        if img.mode != 'L':
            img = img.convert('L')
        
        img_array = np.array(img, dtype=np.float32)
        
        if self.image_size is not None:
            img_pil = Image.fromarray(img_array.astype(np.uint8))
            img_pil = img_pil.resize(self.image_size[::-1], Image.BICUBIC)
            img_array = np.array(img_pil, dtype=np.float32)
        
        img_array = img_array / 255.0
        
        return img_array

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[paddle.Tensor, str]:
        """
        Get a noisy image and its filename.
        
        Args:
            idx: Index of the image
        
        Returns:
            Tuple of (noisy_image_tensor, filename)
        """
        filename = self.image_files[idx]
        noisy_path = osp.join(self.noisy_dir, filename)
        
        noisy_img = self._load_image(noisy_path)
        
        if self.transform is not None:
            noisy_img = self.transform(noisy_img)
        
        noisy_tensor = paddle.to_tensor(noisy_img).unsqueeze(0)  # (1, H, W)
        
        return noisy_tensor, filename