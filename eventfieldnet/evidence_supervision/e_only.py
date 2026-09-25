"""Deterministic auxiliary E branches; the ordinary model keeps dropout=0.5."""
import torch
from eventfieldnet.model.e_readout_scale import scale_e_readout_input

def evidence_tokens(selector, video, text, video_padding_mask, text_padding_mask):
    if not selector.learned_e or not selector.a_local_evidence or selector.c_latent_composition:
        raise ValueError("Requires fixed deployed local E")
    if selector.e_readout_scale != "fixed_init" or video.shape[-1] != 512:
        raise ValueError("Requires raw512/noTEF and fixed_init scaling")
    dropout=selector.e_interaction.dropout
    previous=dropout.training
    try:
        dropout.train(False)
        z=selector.e_interaction.encode(video,text,video_padding_mask,text_padding_mask)
        z,_=scale_e_readout_input(z,~video_padding_mask.bool(),selector.e_readout_scale)
        return torch.tanh(selector.e_interaction.readout(z).squeeze(-1).float())
    finally:
        dropout.train(previous)
