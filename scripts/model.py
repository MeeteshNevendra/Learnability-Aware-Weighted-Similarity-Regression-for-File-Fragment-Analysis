"""Raw-byte Siamese regression model and training loss.

Copied without functional change from the archived implementation:
  <archive>/revision/common.py  (states it is copied unchanged from
  <archive>/Regression_new/new/run_continuous_similarity_ablation.py).
Every width, dropout rate and pooling choice below comes from that code, not from the manuscript text:
  CNN branch: Conv1d(1->64,k9,p4)-BN-GELU-MaxPool2, (64->128,k7,p3)-BN-GELU-MaxPool2, (128->256,k5,p2)-BN-GELU-MaxPool2,
              (256->512,k3,p1)-BN-GELU-AdaptiveAvgPool1d(1)            [no max pooling after the 4th block]
  Transformer branch: Linear(1->128) byte projection + sinusoidal positions, 3 pre-norm encoder layers
              (8 heads, FFN 256, dropout 0.1, GELU, batch_first), mean over positions
  Fusion: Linear(640->512)-LayerNorm-GELU-Dropout(0.25)-Linear(512->256)-LayerNorm-GELU-Dropout(0.15), then L2 norm
  Pair head: [z1, z2, |z1-z2|, z1*z2, cos] (1025) -> Linear 256 - GELU - Dropout(0.25) - Linear 128 - GELU -
              Dropout(0.15) - Linear 1 - Sigmoid
  Loss: SmoothL1(beta=0.05)(pred, y) + 0.3 * margin ranking (random in-batch pairs, margin 0.02)
        + 0.2 * SmoothL1(beta=0.05)((cos + 1) / 2, y)
The pair head is not symmetric in (z1, z2): the archived evaluation used ordered pairs for regression metrics and
0.5 * (S + S^T) for provenance tasks; scripts/evaluate_local.py keeps both conventions and reports the asymmetry.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
EMBED_DIM = 256
LAMBDA_REG, LAMBDA_RANK, LAMBDA_COS = 1.0, 0.30, 0.20
HUBER_BETA, RANK_MARGIN = 0.05, 0.02


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class HybridFragmentEncoder(nn.Module):
    def __init__(self, input_len=512, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1,
                 embed_dim=256):
        super().__init__()
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, padding=4), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2), nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1), nn.BatchNorm1d(512), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.byte_embed = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=input_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fusion = nn.Sequential(
            nn.Linear(512 + d_model, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(0.15))

    checkpoint_transformer = False   # execution detail only (see ContinuousSimilarityModel.set_activation_checkpointing)

    def forward(self, x):
        x_cnn = self.cnn_branch(x.unsqueeze(1)).squeeze(-1)
        h = self.pos_enc(self.byte_embed(x.unsqueeze(-1)))
        if self.training and self.checkpoint_transformer:
            # Recompute parts of each encoder layer in the backward pass instead of storing their activations. The
            # layers have no running statistics and dropout masks are replayed (preserve_rng_state), so outputs and
            # gradients are those of self.transformer(h) up to floating-point non-determinism. TransformerEncoder has
            # no final norm here (norm=None). With norm_first=True a layer computes
            #   h = h + _sa_block(norm1(h));  h = h + _ff_block(norm2(h))      (PyTorch TransformerEncoderLayer)
            from torch.utils.checkpoint import checkpoint
            mode = self.checkpoint_transformer
            for layer in self.transformer.layers:
                if mode == "layer":
                    h = checkpoint(layer, h, use_reentrant=False, preserve_rng_state=True)
                    continue
                sa = lambda t, L=layer: L._sa_block(L.norm1(t), None, None, False)
                ff = lambda t, L=layer: L._ff_block(L.norm2(t))
                h = h + (checkpoint(sa, h, use_reentrant=False, preserve_rng_state=True) if mode in ("sa", "both") else sa(h))
                h = h + (checkpoint(ff, h, use_reentrant=False, preserve_rng_state=True) if mode in ("ff", "both") else ff(h))
        else:
            h = self.transformer(h)
        z = self.fusion(torch.cat([x_cnn, h.mean(dim=1)], dim=1))
        return F.normalize(z, p=2, dim=1)


class ContinuousSimilarityModel(nn.Module):
    def __init__(self, input_len=512, embed_dim=EMBED_DIM):
        super().__init__()
        self.encoder = HybridFragmentEncoder(input_len=input_len, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
                                             dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, embed_dim=embed_dim)
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim * 4 + 1, 256), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 1), nn.Sigmoid())

    def set_activation_checkpointing(self, mode="ff"):
        """Memory-saving execution of the transformer layers during training (shared GPU): False, 'layer', 'sa', 'ff'
        or 'both'. Does not change the model, the batch, BatchNorm behaviour (the CNN branch is not recomputed) or
        the loss."""
        self.encoder.checkpoint_transformer = mode

    def head(self, z1, z2):
        cosine = F.cosine_similarity(z1, z2, dim=1, eps=1e-8).unsqueeze(1)
        feat = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2, cosine], dim=1)
        return self.reg_head(feat).squeeze(1), cosine.squeeze(1)

    def forward(self, x1, x2):
        z1, z2 = self.encoder(x1), self.encoder(x2)
        pred, cosine = self.head(z1, z2)
        return z1, z2, pred, cosine


def n_parameters(model=None):
    model = model or ContinuousSimilarityModel()
    return sum(p.numel() for p in model.parameters())


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ContinuousSimilarityModel().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def pairwise_rank_loss(y_true, y_pred, margin=RANK_MARGIN, generator=None):
    n = y_true.size(0)
    if n < 2:
        return torch.tensor(0.0, device=y_true.device)
    idx_i = torch.randint(0, n, (n,), device=y_true.device, generator=generator)
    idx_j = torch.randint(0, n, (n,), device=y_true.device, generator=generator)
    sign = torch.sign(y_true[idx_i] - y_true[idx_j])
    valid = sign != 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=y_true.device)
    diff = sign[valid] * (y_pred[idx_i][valid] - y_pred[idx_j][valid])
    return F.relu(margin - diff).mean()


def combined_loss(pred, cosine, y_true, generator=None):
    reg = F.smooth_l1_loss(pred, y_true, beta=HUBER_BETA)
    cos = F.smooth_l1_loss((cosine + 1.0) / 2.0, y_true, beta=HUBER_BETA)
    rank = pairwise_rank_loss(y_true, pred, margin=RANK_MARGIN, generator=generator)
    return LAMBDA_REG * reg + LAMBDA_RANK * rank + LAMBDA_COS * cos
