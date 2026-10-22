"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
#from fastmri.data import transforms
#from sensitivity_module import SensitivityModelV2
from unet import Unet,UnetCA,UnetV2

from scipy.io import loadmat,savemat
from unetdisen import DisenNetwork,DisenNetworkV2_2,DisenNetworkV3_2,DisenNetworkV3_3,DisenNetworkV3_4,DisenNetworkV3_4_3,DisenNetworkV3_4NoStyle,DisenNetworkV3_4NoCont
from unetdisen import DisenNetworkV3_5,DisenNetworkV3_6,DisenNetworkV3_6NoCont,DisenNetworkV3_6NoStyle,DisenNetworkV3_6NoPhase

def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask
def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor: #将从mask_from 到mask_to的区域保留
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask

def batched_mask_center_v2(kspace:torch.Tensor,mask:torch.Tensor,num_low_frequencies=12):
    # kspace: N*Nc*Nx*Ny*2
    # mask: N*1*Nx*Ny*1
    mask = torch.zeros_like(mask)
    Ny,Nx = mask.shape[2],mask.shape[3]
    #mask[:,:,int(Ny/2-num_low_frequencies):int(Ny/2+num_low_frequencies),int(Nx/2-num_low_frequencies):int(Nx/2+num_low_frequencies),:] = 1
    mask[:,:,int(Ny/2-num_low_frequencies):int(Ny/2+num_low_frequencies),:,:] = 1
    mask = torch.tile(mask,(1,kspace.shape[1],1,1,2))
    ksp = torch.zeros_like(kspace)
    ksp = kspace  #
    ksp = ksp * mask
    return ksp


# 在Unet输入上引入Normalization
class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x) #先对x做normalization
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std) #最终结果将其还原
        x = self.chan_complex_to_last_dim(x)

        return x

#transpose conv 换成upsample+conv
class NormUnetV2(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = UnetV2(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x) #先对x做normalization
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std) #最终结果将其还原
        x = self.chan_complex_to_last_dim(x)

        return x



class NormDisenNetwork(nn.Module):
    """
    Normalized U-Net model.
    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        image_size=(256,384),
        version='V2_2'
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()
        if version == 'V2_2':
            self.unet = DisenNetworkV2_2(in_chans=in_chans,out_chans=out_chans,image_size=image_size)
        elif version == 'V3_2':
            self.unet = DisenNetworkV3_2(in_chans=in_chans,out_chans=out_chans,image_size=image_size)
        elif version == 'V3_3':
            self.unet = DisenNetworkV3_3(in_chans=in_chans,out_chans=out_chans,image_size=image_size)
        elif version == 'V3_4' or version == 'V3_4_2':
            self.unet = DisenNetworkV3_4(in_chans=in_chans,out_chans=out_chans,image_size=image_size)
        elif version == 'V3_4_3':
            self.unet = DisenNetworkV3_4_3(in_chans=in_chans,out_chans=out_chans,image_size=image_size)
        elif version == 'V3_5':
            self.unet = DisenNetworkV3_5(in_chans=in_chans,out_chans=out_chans,image_size=image_size)
        elif version == 'V3_6':
            self.unet = DisenNetworkV3_6(in_chans=in_chans,out_chans=out_chans,image_size=image_size)
        else:  #默认情况V1
            self.unet = DisenNetwork(in_chans=in_chans,out_chans=out_chans,image_size=image_size)

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor,inputs_type = None) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x) #先对x做normalization
        x, pad_sizes = self.pad(x)

        x,cont,sty = self.unet(x,inputs_type=inputs_type)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std) #最终结果将其还原
        x = self.chan_complex_to_last_dim(x)

        return x,cont,sty


def fft2c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = torch.fft.ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = torch.fft.fftshift(data, dim=[-3, -2])

    return data
def ifft2c(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = torch.fft.fftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = torch.fft.ifftshift(data, dim=[-3, -2])

    return data
class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad, num_low_frequencies_tensor

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,modified=False
    ) -> torch.Tensor:
        if self.mask_center: #默认是True
         #   pad, num_low_freqs = self.get_pad_and_num_low_freqs(  #待修改
         #       mask, num_low_frequencies
         #   )#
          # masked_kspace: N,Nc,Nx,Ny,2
         #   masked_kspace = batched_mask_center(
         #       masked_kspace, pad, pad + num_low_freqs #待修改
         #   )
            masked_kspace = batched_mask_center_v2( ##20231211 by Xue
                masked_kspace,mask,num_low_frequencies
            )
        # convert to image space 先iFFT,再将shape转换为images: N*Nc,1,Nx,Ny,2
        #if modified:
        #    images, batches = self.chans_to_batch_dim(ifft2c(masked_kspace))
        #else:
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace))
        # estimate sensitivities #输入网络，输出结果转换回N,Nc,Nx,Ny,2,再除以RSOS of coil sensitivity map
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class SensitivityModelV2(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnetV2(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad, num_low_frequencies_tensor

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,modified=False
    ) -> torch.Tensor:
        if self.mask_center: #默认是True
         #   pad, num_low_freqs = self.get_pad_and_num_low_freqs(  #待修改
         #       mask, num_low_frequencies
         #   )#
          # masked_kspace: N,Nc,Nx,Ny,2
         #   masked_kspace = batched_mask_center(
         #       masked_kspace, pad, pad + num_low_freqs #待修改
         #   )
            masked_kspace = batched_mask_center_v2( ##20231211 by Xue
                masked_kspace,mask,num_low_frequencies
            )
        # convert to image space 先iFFT,再将shape转换为images: N*Nc,1,Nx,Ny,2
        #if modified:
        #    images, batches = self.chans_to_batch_dim(ifft2c(masked_kspace))
        #else:
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace))
        # estimate sensitivities #输入网络，输出结果转换回N,Nc,Nx,Ny,2,再除以RSOS of coil sensitivity map
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )
    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = 8,modified=False
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies,modified=modified) #待修改
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps,modified=modified)
            pass
        if modified:
            return fastmri.rss(fastmri.complex_abs(ifft2c(kspace_pred)), dim=1)
        else:
            return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)


class VarNetDualDomain(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.cascades = nn.ModuleList(
            [VarNetBlockDualDomain(NormUnet(chans, pools),Unet(in_chans=2,out_chans=2,chans=12,num_pool_layers=pools)) for _ in range(num_cascades)]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = 8,modified=False
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies,modified=modified) #待修改
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps,modified=modified)
        if modified:
            return fastmri.rss(fastmri.complex_abs(ifft2c(kspace_pred)), dim=1)
        else:
            return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)



class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fft2c(fastmri.complex_mul(x, sens_maps))
        else:
            return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fastmri.complex_mul(
                ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)
        else:
            return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,modified = False
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        #原版：
        #soft_dc = torch.where(mask.to(torch.bool), current_kspace - ref_kspace, zero) * self.dc_weight
        #新版 为了便于反向传播
        soft_dc = mask * (current_kspace - ref_kspace) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps,modified=modified)), sens_maps,modified=modified
        )

        return current_kspace - soft_dc - model_term

class VarNetBlockDualDomain(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module,ksp_model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.ksp_model = ksp_model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.ksp_weight = nn.Parameter(0.1*torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fft2c(fastmri.complex_mul(x, sens_maps))
        else:
            return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fastmri.complex_mul(
                ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)
        else:
            return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,modified = False
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps,modified=modified)), sens_maps,modified=modified
        )
        B,C,H,W,_ = current_kspace.shape
        ksp_in = current_kspace.reshape(B*C,H,W,2)
        ksp_in = ksp_in.permute(0,3,1,2)
        ksp_term = self.ksp_model(ksp_in)
        ksp_term = ksp_term.reshape(B,C,H,W,2)

        return current_kspace - soft_dc - model_term - self.ksp_weight * ksp_term

class VarNetBlockV2(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module,SMRB=None):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.stepsize = nn.Parameter(0.1*torch.ones(1))
        self.SMRB = SMRB

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fft2c(fastmri.complex_mul(x, sens_maps))
        else:
            return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fastmri.complex_mul(
                ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)
        else:
            return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,modified = False,gate=1.0
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps,modified=modified)), sens_maps,modified=modified
        )
        #更新coilmap
        B,C,H, W,_ = sens_maps.shape  #1,12,512,248,2
        maps_ = sens_maps.permute(0,4,1,2,3) #看一下maps是否被修改
        maps_ = maps_.reshape(B*C, 2,H, W)
        #maps = maps.permute(0,3,1,2) #BC, 2, H, W
        
        maps_ = self.SMRB(maps_)
        maps_ = maps_.reshape(B,C,2,H,W)
        maps_ = maps_.permute(0,1,3,4,2) #1,12,512,248,2
        kspace = (current_kspace - soft_dc - model_term)
        x = self.sens_reduce((kspace),sens_maps,modified=modified)  #1,512,248,2
        mask_tmp = torch.tile(mask,[1,12,1,1,2])

        sens_maps = sens_maps - self.stepsize*(2*ifft2c(fastmri.complex_mul(mask_tmp*(mask_tmp*kspace-ref_kspace),fastmri.complex_conj(x))) + 0.1*maps_) 
        sens_maps = sens_maps / (fastmri.rss(sens_maps,dim=1) + 1e-12)
        sens_maps = sens_maps * gate
        return kspace,sens_maps

class VarNetBlockDisen(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: DisenNetwork):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fft2c(fastmri.complex_mul(x, sens_maps))
        else:
            return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fastmri.complex_mul(
                ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)
        else:
            return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,modified = False,
        inputs_type = None
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        #soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        soft_dc = mask * (current_kspace - ref_kspace) * self.dc_weight
        current_kspace = current_kspace - soft_dc
        img_denoised,cont,sty = self.model(self.sens_reduce(current_kspace, sens_maps,modified=modified),inputs_type=inputs_type)
        model_term = self.sens_expand(
            img_denoised, sens_maps,modified=modified
        )
        
        return model_term,cont,sty

#调换DC和网络的顺序
class VarNetBlockDisenV3(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: DisenNetwork):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fft2c(fastmri.complex_mul(x, sens_maps))
        else:
            return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fastmri.complex_mul(
                ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)
        else:
            return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,modified = False,
        inputs_type = None
    ) -> torch.Tensor:
       # zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        #soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        
        img_denoised,cont,sty = self.model(self.sens_reduce(current_kspace, sens_maps,modified=modified),inputs_type=inputs_type)
        current_kspace = self.sens_expand(
            img_denoised, sens_maps,modified=modified
        )
        soft_dc = mask * (current_kspace - ref_kspace) * self.dc_weight
        current_kspace = current_kspace - soft_dc

        return current_kspace,cont,sty



class VarNetBlockDisenV2_3(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: DisenNetwork):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fft2c(fastmri.complex_mul(x, sens_maps))
        else:
            return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor,modified=False) -> torch.Tensor:
        if modified:
            return fastmri.complex_mul(
                ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)
        else:
            return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
            ).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,modified = False,
        inputs_type = None
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight

        img_denoised,cont,sty = self.model(self.sens_reduce(current_kspace, sens_maps,modified=modified),inputs_type=inputs_type)
        model_term = self.sens_expand(
            img_denoised, sens_maps,modified=modified
        )
        current_kspace = current_kspace - soft_dc - model_term
        return current_kspace,cont,sty

class MyLoss(nn.Module):
    def __init__(self,device) -> None:        
        super().__init__()
        self.l1 = nn.L1Loss().to(device)
        self.ssim = fastmri.losses.SSIMLoss().to(device)
        self.device = device
        return 
    def forward(self,x,y):
        lossl1 = self.l1(x,y)
        ssimloss = self.ssim(x,y,data_range = torch.Tensor([1.0]).to(self.device))
        mask = torch.zeros_like(y)
        Nb,Ny,Nx = y.shape
        mask[:,int(Ny/4):int(Ny/4*3),int(Nx/3):int(Nx/3*2)] = 1.0
        lossl1roi = self.l1(mask*x,mask*y)
        loss = lossl1+0.1*ssimloss+lossl1roi*6
        return loss


class MyLossV2(nn.Module):
    def __init__(self,device) -> None:        
        super().__init__()
        self.l1 = nn.L1Loss().to(device)
        #self.ssim = fastmri.losses.SSIMLoss().to(device)
        self.device = device
        return 
    def forward(self,x,y):
        lossl1 = self.l1(x,y)
        #ssimloss = self.ssim(x,y,data_range = torch.Tensor([1.0]).to(self.device))
        mask = torch.zeros_like(y)
        Nb,Ny,Nx = y.shape
        mask[:,int(Ny/4):int(Ny/4*3),int(Nx/3):int(Nx/3*2)] = 1.0
        lossl1roi = self.l1(mask*x,mask*y)
        loss = lossl1+lossl1roi*6
        return loss

class DisenLossPaired(nn.Module):
    def __init__(self,device,parallel=False) -> None:        
        super().__init__()
        self.l1 = nn.L1Loss().to(device)
        #self.ssim = fastmri.losses.SSIMLoss().to(device)
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.ssim = fastmri.losses.SSIMLoss().to(device)
        if parallel:
            self.ssim = nn.DataParallel(self.ssim,device_ids=[3,0,2])
        return 
    
    def TV_loss(self, img, weight):
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
        return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

    def compute_cosine(self, x, y):
        # x = self.compute_compact_s(x)
        # y = self.compute_compact_s(y)
        cosine = F.cosine_similarity(x,y,dim=0)
        return cosine
        #eps = 1e-7
        #x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 0)+eps)
        #x_norm = torch.max(x_norm, eps*torch.ones_like(x_norm))
        #y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 0)+eps)
        #y_norm = torch.max(y_norm, eps*torch.ones_like(y_norm))
        #cosine = torch.sum(x * y, 0) / (x_norm * y_norm)
        #return cosine
    def forward(self,img_pred,img_label,cont_list,sty_list,contrast_label:dict,classifier,sens_maps=None):
        #contrast_label: 
        #各个键为：
        # 'Ns': slice index
        # 'Nt': time frame/contrast index
        # 'contrast': 所属的contrast 编号
        # 0. Aorta 1. Cine 2. T1map1 3.T1map2 4.T1map3 5.T2map1 6.T2map2 7.T2map3 8.Tagging 9.blackblood 10.flow2d
        # 'Nsample': 样本编号 
        # 'class': 分类label
        #暂时考虑batch size=2的情形
        #Reconstruction loss
        loss_rec = self.l1(img_pred,img_label)
        loss_ssim = self.ssim(img_pred.unsqueeze(1),img_label.unsqueeze(1),data_range = torch.Tensor([1.0]).cuda()) #batch size=2是否有效 
        if img_pred.shape[0] == 1: #batch size = 1 直接结束
            loss = 10*loss_rec
            return loss,loss_rec            
        # 从列表中把相同contrast的找出来
        contrast_list = (contrast_label['contrast'])
        Nsample = contrast_label['Nsample']
        Ns = contrast_label['Ns']
        class_label = contrast_label['class'].cuda()
        class_pred = classifier(sty_list[-1])
        loss_aux = self.ce(class_pred,class_label)  #辅助分类loss 
        #sty_list = list(map(lambda x: self.l1(x[0,...],x[1,...]), sty_list))  #对style list 中每个unroll的2个图像的style code求差值
        loss_sty = 0
        loss_cont = 0


        if contrast_list[0] == contrast_list[1]: #如果一个batch的两张图来源于相同模态
            #如果配对就正常计算，不配对的话就返回0
            if Nsample[0] == Nsample[1] and Ns[0] == Ns[1]:
                sty_list = list(map(lambda x: (1-self.compute_cosine(x[0,...],x[1,...])), sty_list)) 
                loss_sty = sum(sty_list)
            else:
                pass

        else:  #如果两张图像contrast不一样，就最大化它们的差异 或者让其正交
            if Nsample[0] == Nsample[1] and Ns[0] == Ns[1]:
                sty_list = list(map(lambda x: torch.abs(self.compute_cosine(x[0,...],x[1,...])), sty_list))
                loss_sty = sum(sty_list)
            else:
                pass
            #在模态不同的情况下，判断是否是配对的图像
            if Nsample[0] == Nsample[1] and Ns[0] == Ns[1]:  #如果是同一个scan的不同contrast配对图像
                cont_list = list(map(lambda x: self.l1(x[0,...],x[1,...]), cont_list))
                loss_cont = sum(cont_list)  #要让content map尽可能接近
            else:
                pass

        loss = 10*loss_rec + 0.025*loss_sty + 0.025*loss_cont + 0.05*loss_aux  + 1*loss_ssim

        return loss,loss_rec



class DisenLoss(nn.Module):
    def __init__(self,device,parallel=False) -> None:        
        super().__init__()
        self.l1 = nn.L1Loss().to(device)
        #self.ssim = fastmri.losses.SSIMLoss().to(device)
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.ssim = fastmri.losses.SSIMLoss().to(device)
        if parallel:
            self.ssim = nn.DataParallel(self.ssim,device_ids=[3,0,2])


        return 
    
    def TV_loss(self, img, weight):
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
        return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

    def compute_cosine(self, x, y):
        # x = self.compute_compact_s(x)
        # y = self.compute_compact_s(y)
        cosine = F.cosine_similarity(x,y,dim=0)
        return cosine
        #eps = 1e-7
        #x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 0)+eps)
        #x_norm = torch.max(x_norm, eps*torch.ones_like(x_norm))
        #y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 0)+eps)
        #y_norm = torch.max(y_norm, eps*torch.ones_like(y_norm))
        #cosine = torch.sum(x * y, 0) / (x_norm * y_norm)
        #return cosine
    def forward(self,img_pred,img_label,cont_list,sty_list,contrast_label:dict,classifier,sens_maps=None):
        #contrast_label: 
        #各个键为：
        # 'Ns': slice index
        # 'Nt': time frame/contrast index
        # 'contrast': 所属的contrast 编号
        # 0. Aorta 1. Cine 2. T1map1 3.T1map2 4.T1map3 5.T2map1 6.T2map2 7.T2map3 8.Tagging 9.blackblood 10.flow2d
        # 'Nsample': 样本编号 
        # 'class': 分类label
        #暂时考虑batch size=2的情形
        #Reconstruction loss
        loss_rec = self.l1(img_pred,img_label)
        loss_ssim = self.ssim(img_pred.unsqueeze(1),img_label.unsqueeze(1),data_range = torch.Tensor([1.0]).cuda()) #batch size=2是否有效 
        if img_pred.shape[0] == 1: #batch size = 1 直接结束
            loss = 10*loss_rec
            return loss,loss_rec            
        # 从列表中把相同contrast的找出来
        contrast_list = (contrast_label['contrast'])
        Nsample = contrast_label['Nsample']
        Ns = contrast_label['Ns']
        class_label = contrast_label['class'].cuda()
        class_pred = classifier(sty_list[-1])
        loss_aux = self.ce(class_pred,class_label)  #辅助分类loss 
        #sty_list = list(map(lambda x: self.l1(x[0,...],x[1,...]), sty_list))  #对style list 中每个unroll的2个图像的style code求差值
        loss_sty = 0
        loss_cont = 0

        if contrast_list[0] == contrast_list[1]: #如果一个batch的两张图来源于相同模态
            sty_list = list(map(lambda x: (1-self.compute_cosine(x[0,...],x[1,...])), sty_list)) 
            loss_sty = sum(sty_list)
        else:  #如果两张图像不一样，就最大化它们的差异 或者让其正交
            sty_list = list(map(lambda x: torch.abs(self.compute_cosine(x[0,...],x[1,...])), sty_list))
            loss_sty = sum(sty_list)
            #在模态不同的情况下，判断是否是配对的图像
            if Nsample[0] == Nsample[1] and Ns[0] == Ns[1]:  #如果是同一个scan的不同contrast配对图像
                cont_list = list(map(lambda x: self.l1(x[0,...],x[1,...]), cont_list))
                loss_cont = sum(cont_list)  #要让content map尽可能接近
        #coilmap 正则项
    #    if sens_maps is not None:
    #        loss_sens = self.TV_loss(fastmri.complex_abs(sens_maps),weight=100)
        #if torch.isnan(loss_sty):
        #    loss_sty = 0.0
        #loss = 20*loss_rec + 0.05*loss_sty + 0.05*loss_cont + 0.1*loss_aux
        loss = 10*loss_rec + 0.025*loss_sty + 0.025*loss_cont + 0.05*loss_aux  + 1*loss_ssim
        #if torch.isnan(loss):
        #    print("NaN!")  #某个地方出错后会导致loss_rec变为NaN
        #    print(loss_rec)
        #    print(loss_sty)
        #    print(loss_cont)
        #    print(loss_aux)
        return loss,loss_rec

class DisenLossV2(nn.Module):
    def __init__(self,device) -> None:        
        super().__init__()
        self.l1 = nn.L1Loss().to(device)
        #self.ssim = fastmri.losses.SSIMLoss().to(device)
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.ssim = fastmri.losses.SSIMLoss().to(device)

        return 
    
    def TV_loss(self, img, weight):
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
        return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

    def compute_cosine(self, x, y):
        # x = self.compute_compact_s(x)
        # y = self.compute_compact_s(y)
        cosine = F.cosine_similarity(x,y,dim=0)
        return cosine

    def forward(self,img_pred,img_label,cont_list,sty_list,contrast_label:dict,classifier,sens_maps=None):
        #contrast_label: 
        #各个键为：
        # 'Ns': slice index
        # 'Nt': time frame/contrast index
        # 'contrast': 所属的contrast 编号
        # 0. Aorta 1. Cine 2. T1map1 3.T1map2 4.T1map3 5.T2map1 6.T2map2 7.T2map3 8.Tagging 9.blackblood 10.flow2d
        # 'Nsample': 样本编号 
        # 'class': 分类label
        #暂时考虑batch size=2的情形
        #Reconstruction loss
        loss_rec = self.l1(img_pred,img_label)
        loss_ssim = self.ssim(img_pred.unsqueeze(1),img_label.unsqueeze(1),data_range = torch.Tensor([1.0]).cuda()) #batch size=2是否有效 
        if img_pred.shape[0] == 1: #batch size = 1 直接结束
            loss = 10*loss_rec
            return loss,loss_rec,[torch.tensor(0.)]*4     
        # 从列表中把相同contrast的找出来
        contrast_list = (contrast_label['contrast'])
        Nsample = contrast_label['Nsample']
        Ns = contrast_label['Ns']
        class_label = contrast_label['class'].cuda()
        class_pred = classifier(sty_list[-1])
        loss_aux = self.ce(class_pred,class_label)  #辅助分类loss 
        #sty_list = list(map(lambda x: self.l1(x[0,...],x[1,...]), sty_list))  #对style list 中每个unroll的2个图像的style code求差值
        loss_sty = torch.tensor(0.)
        loss_cont = torch.tensor(0.)

        if contrast_list[0] == contrast_list[1]: #如果一个batch的两张图来源于相同模态
            sty_list = list(map(lambda x: (1-self.compute_cosine(x[0,...],x[1,...])), sty_list)) 
            loss_sty = sum(sty_list)
        else:  #如果两张图像不一样，就最大化它们的差异 或者让其正交
            sty_list = list(map(lambda x: torch.abs(self.compute_cosine(x[0,...],x[1,...])), sty_list))
            loss_sty = sum(sty_list)
            #在模态不同的情况下，判断是否是配对的图像
            if Nsample[0] == Nsample[1] and Ns[0] == Ns[1]:  #如果是同一个scan的不同contrast配对图像
                cont_list = list(map(lambda x: (1-self.compute_cosine(torch.flatten(x[0,...]),torch.flatten(x[1,...]))), cont_list))
                loss_cont = sum(cont_list)  #要让content map尽可能接近
        #coilmap 正则项
    #    if sens_maps is not None:
    #        loss_sens = self.TV_loss(fastmri.complex_abs(sens_maps),weight=100)
        #if torch.isnan(loss_sty):
        #    loss_sty = 0.0
        #loss = 20*loss_rec + 0.05*loss_sty + 0.05*loss_cont + 0.1*loss_aux
        loss = 10*loss_rec + 0.025*loss_sty + 0.025*loss_cont + 0.05*loss_aux  + 1*loss_ssim
        return loss,loss_rec,[loss_sty,loss_cont,loss_aux,loss_ssim]


class DisenLossV3(nn.Module):
    def __init__(self,device,parallel=False,lm1=0.025,lm2=0.025,lm3=0.05) -> None:        
        super().__init__()
        self.l1 = nn.L1Loss().to(device)
        #self.ssim = fastmri.losses.SSIMLoss().to(device)
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.ssim = fastmri.losses.SSIMLoss().to(device)
        self.lm1 = lm1
        self.lm2 = lm2
        self.lm3 = lm3
        if parallel:
            self.ssim = nn.DataParallel(self.ssim,device_ids=[3,0,2])


        return 
    
    def TV_loss(self, img, weight):
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
        return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

    def compute_cosine(self, x, y):
        # x = self.compute_compact_s(x)
        # y = self.compute_compact_s(y)
        cosine = F.cosine_similarity(x,y,dim=0)
        return cosine

    def forward(self,img_pred,img_label,cont_list,sty_list,contrast_label:dict,classifier,sens_maps=None):
        #contrast_label: 
        #各个键为：
        # 'Ns': slice index
        # 'Nt': time frame/contrast index
        # 'contrast': 所属的contrast 编号
        # 0. Aorta 1. Cine 2. T1map1 3.T1map2 4.T1map3 5.T2map1 6.T2map2 7.T2map3 8.Tagging 9.blackblood 10.flow2d
        # 'Nsample': 样本编号 
        # 'class': 分类label
        #暂时考虑batch size=2的情形
        #Reconstruction loss
        loss_rec = self.l1(img_pred,img_label)
        loss_ssim = self.ssim(img_pred.unsqueeze(1),img_label.unsqueeze(1),data_range = torch.Tensor([1.0]).cuda()) #batch size=2是否有效 
        if img_pred.shape[0] == 1: #batch size = 1 直接结束
            loss = 10*loss_rec
            return loss,loss_rec            
        # 从列表中把相同contrast的找出来
        contrast_list = (contrast_label['contrast'])
        Nsample = contrast_label['Nsample']
        Ns = contrast_label['Ns']
        class_label = contrast_label['class'].cuda()
        class_pred = classifier(sty_list[-1])
        loss_aux = self.ce(class_pred,class_label)  #辅助分类loss 
        #sty_list = list(map(lambda x: self.l1(x[0,...],x[1,...]), sty_list))  #对style list 中每个unroll的2个图像的style code求差值
        loss_sty = 0
        loss_cont = 0

        if contrast_list[0] == contrast_list[1]: #如果一个batch的两张图来源于相同模态
            sty_list = list(map(lambda x: (1-self.compute_cosine(x[0,...],x[1,...])), sty_list)) 
            loss_sty = sum(sty_list)
        else:  #如果两张图像不一样，就最大化它们的差异 或者让其正交
            sty_list = list(map(lambda x: torch.abs(self.compute_cosine(x[0,...],x[1,...])), sty_list))
            loss_sty = sum(sty_list)
            #在模态不同的情况下，判断是否是配对的图像
            if Nsample[0] == Nsample[1] and Ns[0] == Ns[1]:  #如果是同一个scan的不同contrast配对图像
                cont_list = list(map(lambda x: self.l1(x[0,...],x[1,...]), cont_list))
                loss_cont = sum(cont_list)  #要让content map尽可能接近
        #coilmap 正则项
    #    if sens_maps is not None:
    #        loss_sens = self.TV_loss(fastmri.complex_abs(sens_maps),weight=100)
        #if torch.isnan(loss_sty):
        #    loss_sty = 0.0
        #loss = 20*loss_rec + 0.05*loss_sty + 0.05*loss_cont + 0.1*loss_aux
        loss = 10*loss_rec + self.lm1*loss_sty + self.lm2*loss_cont + self.lm3*loss_aux  + 1*loss_ssim
        #if torch.isnan(loss):
        #    print("NaN!")  #某个地方出错后会导致loss_rec变为NaN
        #    print(loss_rec)
        #    print(loss_sty)
        #    print(loss_cont)
        #    print(loss_aux)
        return loss,loss_rec


class DisenLossNoCont(nn.Module):
    def __init__(self,device) -> None:        
        super().__init__()
        self.l1 = nn.L1Loss().to(device)
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.ssim = fastmri.losses.SSIMLoss().to(device)
        return 
    
    def TV_loss(self, img, weight):
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
        return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

    def compute_cosine(self, x, y):
        cosine = F.cosine_similarity(x,y,dim=0)
        return cosine

    def forward(self,img_pred,img_label,cont_list,sty_list,contrast_label:dict,classifier,sens_maps=None):
        #contrast_label: 
        #各个键为：
        # 'Ns': slice index
        # 'Nt': time frame/contrast index
        # 'contrast': 所属的contrast 编号
        # 0. Aorta 1. Cine 2. T1map1 3.T1map2 4.T1map3 5.T2map1 6.T2map2 7.T2map3 8.Tagging 9.blackblood 10.flow2d
        # 'Nsample': 样本编号 
        # 'class': 分类label
        #暂时考虑batch size=2的情形
        #Reconstruction loss
        loss_rec = self.l1(img_pred,img_label)
        loss_ssim = self.ssim(img_pred.unsqueeze(1),img_label.unsqueeze(1),data_range = torch.Tensor([1.0]).cuda()) #batch size=2是否有效 
        if img_pred.shape[0] == 1: #batch size = 1 直接结束
            loss = 10*loss_rec
            return loss,loss_rec            
        # 从列表中把相同contrast的找出来
        contrast_list = (contrast_label['contrast'])
        #Nsample = contrast_label['Nsample']
        #Ns = contrast_label['Ns']
        class_label = contrast_label['class'].cuda()
        class_pred = classifier(sty_list[-1])
        loss_aux = self.ce(class_pred,class_label)  #辅助分类loss 
        #sty_list = list(map(lambda x: self.l1(x[0,...],x[1,...]), sty_list))  #对style list 中每个unroll的2个图像的style code求差值
        loss_sty = 0
        #loss_cont = 0
        if contrast_list[0] == contrast_list[1]: #如果一个batch的两张图来源于相同模态
            sty_list = list(map(lambda x: (1-self.compute_cosine(x[0,...],x[1,...])), sty_list)) 
            loss_sty = sum(sty_list)
        else:  #如果两张图像不一样，就最大化它们的差异 或者让其正交
            sty_list = list(map(lambda x: torch.abs(self.compute_cosine(x[0,...],x[1,...])), sty_list))
            loss_sty = sum(sty_list)
            #在模态不同的情况下，判断是否是配对的图像
            #if Nsample[0] == Nsample[1] and Ns[0] == Ns[1]:  #如果是同一个scan的不同contrast配对图像
            #    cont_list = list(map(lambda x: self.l1(x[0,...],x[1,...]), cont_list))
            #    loss_cont = sum(cont_list)  #要让content map尽可能接近
        #coilmap 正则项
    #    if sens_maps is not None:
    #        loss_sens = self.TV_loss(fastmri.complex_abs(sens_maps),weight=100)
        #if torch.isnan(loss_sty):
        #    loss_sty = 0.0
        loss = 10*loss_rec + 0.025*loss_sty +  0.05*loss_aux  + 1*loss_ssim
        return loss,loss_rec

class DisenLossNoStyle(nn.Module):
    def __init__(self,device) -> None:        
        super().__init__()
        self.l1 = nn.L1Loss().to(device)
        self.device = device
        self.ssim = fastmri.losses.SSIMLoss().to(device)
        return 
    
    def TV_loss(self, img, weight):
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
        return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

    def compute_cosine(self, x, y):
        cosine = F.cosine_similarity(x,y,dim=0)
        return cosine
    def forward(self,img_pred,img_label,cont_list,sty_list,contrast_label:dict,classifier,sens_maps=None):
        #contrast_label: 
        #各个键为：
        # 'Ns': slice index
        # 'Nt': time frame/contrast index
        # 'contrast': 所属的contrast 编号
        # 0. Aorta 1. Cine 2. T1map1 3.T1map2 4.T1map3 5.T2map1 6.T2map2 7.T2map3 8.Tagging 9.blackblood 10.flow2d
        # 'Nsample': 样本编号 
        # 'class': 分类label
        #暂时考虑batch size=2的情形
        #Reconstruction loss
        loss_rec = self.l1(img_pred,img_label)
        loss_ssim = self.ssim(img_pred.unsqueeze(1),img_label.unsqueeze(1),data_range = torch.Tensor([1.0]).cuda()) #batch size=2是否有效 
        if img_pred.shape[0] == 1: #batch size = 1 直接结束
            loss = 10*loss_rec
            return loss,loss_rec            
        # 从列表中把相同contrast的找出来
        contrast_list = (contrast_label['contrast'])
        Nsample = contrast_label['Nsample']
        Ns = contrast_label['Ns']
        loss_cont = torch.tensor(0.)

        if contrast_list[0] == contrast_list[1]: #如果一个batch的两张图来源于相同模态
            pass
        else:  #如果两张图像不一样，就最大化它们的差异 或者让其正交
            if Nsample[0] == Nsample[1] and Ns[0] == Ns[1]:  #如果是同一个scan的不同contrast配对图像
                cont_list = list(map(lambda x: self.l1(x[0,...],x[1,...]), cont_list))
                loss_cont = sum(cont_list)  #要让content map尽可能接近
        loss = 10*loss_rec  + 0.025*loss_cont  + 1*loss_ssim
        return loss,loss_rec

class Classifier(nn.Module):
    def __init__(self, size=512,out_features=8) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features=size,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=64),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=64,out_features=32),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=32,out_features=16),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=16,out_features=out_features),
        nn.Sigmoid()
        )
    def forward(self,x):
        x = self.net(x)
        return x

#the implementation of DRASTIC
class VarNetDisenV2(nn.Module):
    """
    A full variational network model with disentanglement representation learning
    """
    def __init__(
        self,
        num_cascades: int = 6,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        version='V2',
        dc_after_net = False, #dc_after_infer=False
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
       # self.dc_after_infer = dc_after_infer
        if version != 'V3_6_2':
            self.sens_net = SensitivityModel(
                    chans=sens_chans,
                    num_pools=sens_pools,
                    mask_center=mask_center,
                )
        else:
           self.sens_net = SensitivityModelV2(
               chans=sens_chans,
               num_pools=sens_pools,
               mask_center=mask_center,
           )
        if dc_after_net:
            self.cascades = nn.ModuleList(
            [VarNetBlockDisenV3(NormDisenNetwork(chans, pools,version=version)) for _ in range(num_cascades)] #unshared weights
        )
        elif 'V3_4_2' in version or 'V3_5' in version or 'V3_6' in version:
            self.cascades = nn.ModuleList(
            [VarNetBlockDisen(NormDisenNetwork(chans, pools,version=version)) for _ in range(num_cascades)] #unshared weights
        )
        else:
            self.cascades = nn.ModuleList(
            [VarNetBlockDisen(NormDisenNetwork(chans, pools,version=version))]*num_cascades #shared weights
        )
        self.content_list = []
        self.style_list = []
    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = 8,inputs_type=None
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies) #待修改
        kspace_pred = masked_kspace.clone()
        self.content_list.clear()
        self.style_list.clear()

        for cascade in self.cascades:
            kspace_pred,cont,sty = cascade(kspace_pred, masked_kspace, mask, sens_maps,inputs_type=inputs_type)
            #将中间结果的content和style加进列表
            self.content_list.append(cont)
            self.style_list.append(sty)
        # if self.dc_after_infer:  # 若训练时最后一步不是DC而测试时最后一步是DC，则会导致表现的下降
        #     #处理ksp
        #     soft_dc = mask * (kspace_pred - masked_kspace) * 1
        #     kspace_pred = kspace_pred - soft_dc
        #if modified:
        #    return fastmri.rss(fastmri.complex_abs(ifft2c(kspace_pred)), dim=1),self.content_list,self.style_list
        #else:
        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1),self.content_list,self.style_list,sens_maps
    def load_ckpt(self,modelList):
        for idx,submodel in enumerate(self.cascades):
            submodel.model.unet.content_encoder_enc.load_state_dict(torch.load(modelList[0]))
            submodel.model.unet.content_encoder_dec.load_state_dict(torch.load(modelList[1]))
            submodel.model.unet.style_encoder.load_state_dict(torch.load(modelList[2]))
            submodel.model.unet.decoder.load_state_dict(torch.load(modelList[3]))
        self.sens_net.load_state_dict(torch.load(modelList[4]))
        return
    def load_ckpt_V3_6(self,modelList):
        for idx,submodel in enumerate(self.cascades):
            submodel.model.unet.content_encoder.load_state_dict(torch.load(modelList[0]))
            submodel.model.unet.style_encoder.load_state_dict(torch.load(modelList[1]))
        self.sens_net.load_state_dict(torch.load(modelList[2]))
        return
    def freeze_params(self):
        for idx,submodel in enumerate(self.cascades):
            for param in submodel.model.unet.content_encoder_enc.parameters():
                param.requires_grad = False
            for param in submodel.model.unet.content_encoder_dec.parameters():
                param.requires_grad = False
        return
    def unfreeze_params(self):
        for idx,submodel in enumerate(self.cascades):
            for param in submodel.model.unet.content_encoder_enc.parameters():
                param.requires_grad = True
            for param in submodel.model.unet.content_encoder_dec.parameters():
                param.requires_grad = True
        return
        




class BaselineLoss(nn.Module):
    def __init__(self,device) -> None:        
        super().__init__()
        self.l1 = nn.L1Loss().to(device)
        #self.ssim = fastmri.losses.SSIMLoss().to(device)
        self.device = device
        self.ssim = fastmri.losses.SSIMLoss().to(device)
        return 

    def forward(self,img_pred,img_label):
    
        loss_rec = self.l1(img_pred,img_label)
        loss_ssim = self.ssim(img_pred.unsqueeze(1),img_label.unsqueeze(1),data_range = torch.Tensor([1.0]).to(self.device)) #batch size=2是否有效 
                
        loss = 10*loss_rec + 1*loss_ssim

        return loss,loss_rec



import numpy as np
def nmse(image_gt,image_recon): #NMSE
    SE=np.sum(np.abs(image_gt-image_recon)**2)
    NMSE = SE/np.sum(np.abs(image_gt)**2)
    return NMSE