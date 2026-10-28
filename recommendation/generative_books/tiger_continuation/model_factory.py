"""Construct the existing 14.28M architecture; load all trained tensors afterward."""
import copy

from transformers import T5ForConditionalGeneration


def create_wide_model(codes, base_factory):
    model = base_factory('tiger', codes)
    config = copy.deepcopy(model.net.config)
    config.d_model, config.d_ff = 256, 2048
    model.net = T5ForConditionalGeneration(config)
    if sum(p.numel() for p in model.parameters()) != 14281344:
        raise RuntimeError('Unexpected continuation model parameter count')
    return model
