# Copyright (c) OpenMMLab. All rights reserved.
from mmpl.registry import MODELS

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
PLERS = MODELS
RETRIEVER = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_pler(cfg):
    """Build classifier."""
    return PLERS.build(cfg)


def build_retriever(cfg):
    """Build retriever."""
    return RETRIEVER.build(cfg)
