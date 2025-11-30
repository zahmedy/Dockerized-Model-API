import torch
import torch.nn as nn

class TinyBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),    # 4 input features -> 8 hidden units
            nn.ReLU(),
            nn.Linear(8, 1),    # 8 units -> 1 output (binary)
            nn.Sigmoid()        # Convert to probability (0-1)
        )

    def forward(self, x):
        return self.net(x)