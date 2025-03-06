import torch.nn as nn
import torch.nn.functional as F

class OutputModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OutputModule, self).__init__()
        self.output_1 = nn.Linear(input_dim, hidden_dim)
        self.output_2 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_normal_(self.output_1.weight)
        nn.init.xavier_normal_(self.output_1.weight)
        if self.output_1.bias is not None:
            nn.init.zeros_(self.output_1.bias)
        if self.output_2.bias is not None:
            nn.init.zeros_(self.output_2.bias)
    
    def forward(self, output):
        return self.output_2(F.leaky_relu(self.output_1(output), negative_slope=0.01))

# Example usage:
# force_module = ForceModule(input_dim=..., hidden_dim=..., output_dim=...)
# forces = force_module(output)
