"""The regression network (manuscript Section 3.3, Eqs. 19-21).

    h1 = ReLU(W1 s~ + b1)        Eq. (19), 64 units
    h2 = ReLU(W2 h1 + b2)        Eq. (20), 32 units
    y^ = sigma(W3 h2 + b3)       Eq. (21), sigmoid output in [0, 1]

The input scales w (Eq. 5, s~ = w * s) are applied outside the network and stored with each
checkpoint.  With four inputs the network has 2,433 trainable parameters.
"""
import numpy as np
import torch
from torch import nn

from .config import HIDDEN_UNITS


class LAWSRNet(nn.Module):
    def __init__(self, n_inputs=4, hidden=HIDDEN_UNITS):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], 1)

    def forward(self, s):
        h1 = torch.relu(self.fc1(s))
        h2 = torch.relu(self.fc2(h1))
        return torch.sigmoid(self.out(h2)).squeeze(-1)


def n_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ScaledScorer:
    """A fitted network together with its frozen input scales: score = f(w * s)."""

    def __init__(self, state_dict, scales, n_inputs):
        self.scales = np.asarray(scales, dtype=np.float64)
        self.net = LAWSRNet(n_inputs)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    @torch.no_grad()
    def __call__(self, S, batch=1 << 18):
        S = np.asarray(S)
        flat = S.reshape(-1, S.shape[-1])
        out = np.empty(len(flat), dtype=np.float32)
        w = torch.as_tensor(self.scales, dtype=torch.float32)
        for s in range(0, len(flat), batch):
            x = torch.as_tensor(np.ascontiguousarray(flat[s:s + batch], dtype=np.float32)) * w
            out[s:s + batch] = self.net(x).numpy()
        return out.reshape(S.shape[:-1])


def load_scorer(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ScaledScorer(ck['state_dict'], ck['scales'], ck['n_inputs'])
