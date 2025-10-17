# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from lib.backbones.silk.silk import SiLKVGG
from lib.backbones.superpoint.vgg import ParametricVGG
from pytorch_lightning.utilities.cloud_io import load
from silk.utils import CHECKPOINT_PATH

__all__ = ['load_model']

def load_model():
    backbone = ParametricVGG(
        use_max_pooling=False,
        padding=0,
        normalization_fn=[
            torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)
        ]
    )
    model = SiLKVGG(
        in_channels=1,
        backbone=backbone,
        detection_threshold=1.0,
        detection_top_k=10000,
        nms_dist=0,
        border_dist=0,
        default_outputs=("sparse_positions", "sparse_descriptors"),
        descriptor_scale_factor=1.41,
        padding=0
    )
    state_dict = load(CHECKPOINT_PATH, 'cpu')['state_dict']
    state_dict = {
        k[len('_mods.model.'):] : v
            for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    return model
