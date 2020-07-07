
import torch


def mask_seq_batch(seq, mask):
    return seq[:, mask]
