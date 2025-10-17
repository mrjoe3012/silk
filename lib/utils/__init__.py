# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from importlib import resources
CHECKPOINT_PATH = os.path.join(
    resources.files('silk'), 'coco-rgb-aug.ckpt'
)

__all__ = ['CHECKPOINT_PATH']
