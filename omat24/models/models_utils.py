import torch.nn as nn
import torch.nn.functional as F

class OutputModule(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(OutputModule, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.output_1 = nn.Linear(self.in_features, self.hidden_dim)
        self.output_2 = nn.Linear(self.hidden_dim, self.out_features)
        nn.init.xavier_normal_(self.output_1.weight)
        nn.init.xavier_normal_(self.output_1.weight)
        if self.output_1.bias is not None:
            nn.init.zeros_(self.output_1.bias)
        if self.output_2.bias is not None:
            nn.init.zeros_(self.output_2.bias)
    
    def forward(self, output):
        return self.output_2(F.tanh(self.output_1(output)))

# Example usage:
# force_module = ForceModule(input_dim=..., hidden_dim=..., output_dim=...)
# forces = force_module(output)
