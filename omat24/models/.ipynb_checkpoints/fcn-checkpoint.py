import torch
import torch.nn as nn


class MetaFCNModels:
    def __init__(self, vocab_size=119, use_factorized=False):
        self.configurations = [
            # 970 params 
            {"embedding_dim": 8, "hidden_dim": 16, "depth": 2},
            # 11.6k parameters (includes embedding layer)
            {"embedding_dim": 24, "hidden_dim": 48, "depth": 4},
            # 104k parameters
            {"embedding_dim": 60, "hidden_dim": 108, "depth": 8},
            # # 470k parameters 
            # {"embedding_dim": 96, "hidden_dim": 192, "depth": 12},
            # 1M parameters
            {"embedding_dim": 128, "hidden_dim": 256, "depth": 15},
            # # 4.3M parameters 
            # {"embedding_dim": 256, "hidden_dim": 512, "depth": 16},
            # 10.9M parameters
            {"embedding_dim": 384, "hidden_dim": 768, "depth": 18},
            # 100M parameters 
            # {"embedding_dim": 1024, "hidden_dim": 2048, "depth": 24}
        ]
        self.vocab_size = vocab_size
        self.use_factorized = use_factorized

    def __getitem__(self, idx):
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range")
            
        config = self.configurations[idx]
        
        # Create the model first to get accurate parameter count
        model = FCNModel(
            vocab_size=self.vocab_size,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            depth=config["depth"],
            use_factorized=self.use_factorized,
            use_layer_norm=False,
            dropout_rate=0.0,
            weight_decay=0.0,
            spectral_norm=False,
        )
        
        # Scale regularization based on actual parameter count
        num_params = model.num_params
        
        # Progressive regularization - increases with model size
        if num_params < 100_000:  # Small models
            dropout_rate = 0.0
            weight_decay = 0.0
            use_layer_norm = False
            spectral_norm = False
        elif num_params < 1_000_000:  # Medium models
            dropout_rate = 0.1
            weight_decay = 1e-5
            use_layer_norm = True
            spectral_norm = False
        elif num_params < 10_000_000:  # Large models
            dropout_rate = 0.2
            weight_decay = 5e-5
            use_layer_norm = True
            spectral_norm = False
        else:  # Very large models
            dropout_rate = 0.3
            weight_decay = 1e-4
            use_layer_norm = True
            spectral_norm = True
            
        # Create and return the properly regularized model
        return FCNModel(
            vocab_size=self.vocab_size,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            depth=config["depth"],
            use_factorized=self.use_factorized,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            spectral_norm=spectral_norm,
        )

    def __len__(self):
        return len(self.configurations)

    def __iter__(self):
        for idx in range(len(self.configurations)):
            yield self[idx]


class FCNModel(nn.Module):
    def __init__(
        self,
        vocab_size=119,
        embedding_dim=128,
        hidden_dim=256,
        depth=4,
        use_factorized=False,
        use_layer_norm=False,
        dropout_rate=0.0,
        weight_decay=0.0,
        spectral_norm=False,
    ):
        super(FCNModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_factorized = use_factorized
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.spectral_norm = spectral_norm

        # Embedding for atomic numbers
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim
        )  # Assuming atomic numbers < 119

        # Initial layer
        if self.use_factorized:
            self.fc1 = nn.Linear(embedding_dim + 5, hidden_dim)
        else:
            self.fc1 = nn.Linear(embedding_dim + 3, hidden_dim)

        # Inner layers with residual connections
        self.inner_layers = nn.ModuleList()
        for _ in range(depth):
            # Create the linear layer first for potential spectral normalization
            linear = nn.Linear(hidden_dim, hidden_dim)
            
            # Apply spectral normalization if requested
            if self.spectral_norm:
                linear = nn.utils.spectral_norm(linear)
                
            # Create the sequential layer with all components
            if self.use_layer_norm:
                if self.dropout_rate > 0:
                    layer = nn.Sequential(
                        linear,
                        nn.LayerNorm(hidden_dim),
                        nn.LeakyReLU(),
                        nn.Dropout(self.dropout_rate),
                    )
                else:
                    layer = nn.Sequential(
                        linear,
                        nn.LayerNorm(hidden_dim),
                        nn.LeakyReLU(),
                    )
            else:
                if self.dropout_rate > 0:
                    layer = nn.Sequential(
                        linear,
                        nn.LeakyReLU(),
                        nn.Dropout(self.dropout_rate),
                    )
                else:
                    layer = nn.Sequential(
                        linear,
                        nn.LeakyReLU(),
                    )
            self.inner_layers.append(layer)

        # Output layers - Two layers for each output head
        # Force prediction layers
        self.force_output_1 = nn.Linear(hidden_dim, hidden_dim)
        self.force_output_2 = nn.Linear(hidden_dim, 3)
        
        # Energy prediction layers
        self.energy_output_1 = nn.Linear(hidden_dim, hidden_dim)
        self.energy_output_2 = nn.Linear(hidden_dim, 1)
        
        # Stress prediction layers
        self.stress_output_1 = nn.Linear(hidden_dim, hidden_dim)
        self.stress_output_2 = nn.Linear(hidden_dim, 6)
        
        # Initialize weights with Xavier initialization
        nn.init.xavier_normal_(self.force_output_1.weight)
        nn.init.zeros_(self.force_output_1.bias)
        nn.init.xavier_normal_(self.force_output_2.weight)
        nn.init.zeros_(self.force_output_2.bias)
        
        nn.init.xavier_normal_(self.energy_output_1.weight)
        nn.init.zeros_(self.energy_output_1.bias)
        nn.init.xavier_normal_(self.energy_output_2.weight)
        nn.init.zeros_(self.energy_output_2.bias)
        
        nn.init.xavier_normal_(self.stress_output_1.weight)
        nn.init.zeros_(self.stress_output_1.bias)
        nn.init.xavier_normal_(self.stress_output_2.weight)
        nn.init.zeros_(self.stress_output_2.bias)
        
        # Also initialize the FC1 and inner layers
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        # Initialize inner layer weights
        for layer in self.inner_layers:
            # Get the Linear layer (first component in the Sequential)
            linear = layer[0]
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

        # Calculate number of parameters
        self.num_params = sum(
            p.numel() for name, p in self.named_parameters() if "embedding" not in name
        )

    def force_output(self, x):
        """Two-layer force output head with LeakyReLU activation and dropout."""
        x = nn.functional.leaky_relu(self.force_output_1(x), negative_slope=0.01)
        if self.dropout_rate > 0 and self.training:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        return self.force_output_2(x)
    
    def energy_output(self, x):
        """Two-layer energy output head with LeakyReLU activation and dropout."""
        x = nn.functional.leaky_relu(self.energy_output_1(x), negative_slope=0.01)
        if self.dropout_rate > 0 and self.training:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        return self.energy_output_2(x)
    
    def stress_output(self, x):
        """Two-layer stress output head with LeakyReLU activation and dropout."""
        x = nn.functional.leaky_relu(self.stress_output_1(x), negative_slope=0.01)
        if self.dropout_rate > 0 and self.training:
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        return self.stress_output_2(x)

    def forward(self, atomic_numbers, positions, distance_matrix=None, mask=None):
        """
        Args:
            atomic_numbers: Tensor of shape [batch_size, vocab_size]
            positions: Tensor of shape [batch_size, vocab_size, 3]
            distance_matrix: Factorized Inverse Distances [batch_size, vocab_size, 5]

        Returns:
            forces: Tensor of shape [batch_size, vocab_size, 3]
            energy: Tensor of shape [batch_size]
            stress: Tensor of shape [batch_size, 6]
        """
        # Create a mask for valid atoms (non-padded)
        new_mask = mask.unsqueeze(-1)  # Shape: [batch_size, vocab_size, 1]

        # Embed atomic numbers
        atomic_embeddings = self.embedding(
            atomic_numbers
        )  # [batch_size, vocab_size, embedding_dim]
        
        # Apply embedding dropout if specified
        if self.dropout_rate > 0 and self.training:
            atomic_embeddings = nn.functional.dropout(
                atomic_embeddings, p=self.dropout_rate, training=self.training
            )
        
        # Concatenate embeddings with positions or distances
        if self.use_factorized:
            x = torch.cat(
                [atomic_embeddings, distance_matrix], dim=-1
            )  # [batch_size, vocab_size, embedding_dim + 5]
        else:
            x = torch.cat(
                [atomic_embeddings, positions], dim=-1
            )  # [batch_size, vocab_size, embedding_dim + 3]

        # Initial layer
        x = self.fc1(x)
        
        # Apply stochastic depth (similar to dropout but for entire layers)
        # Only applicable for very deep networks
        layer_dropout_prob = 0.0
        if self.depth > 10 and self.dropout_rate > 0 and self.training:
            layer_dropout_prob = self.dropout_rate * 0.5  # Reduced strength for layer dropout
        
        # Pass through inner layers with residual connections
        for i, layer in enumerate(self.inner_layers):
            # Stochastic depth - randomly skip some layers during training
            if self.training and layer_dropout_prob > 0:
                if torch.rand(1).item() < layer_dropout_prob:
                    continue
                    
            # Regular forward pass with residual connection    
            x_res = layer(x)
            
            # Scale residual connections for stability in deeper networks
            if self.depth > 10:
                # Scale factor that decreases with depth
                scale = 1.0 / (self.depth ** 0.5)
                x = x + scale * x_res  # Scaled residual connection
            else:
                x = x + x_res  # Standard residual connection

        # Predict forces
        forces = self.force_output(x)  # [batch_size, vocab_size, 3]
        forces = forces * new_mask.float()  # Mask padded atoms

        # Predict per-atom energy contributions and sum
        energy_contrib = self.energy_output(x).squeeze(-1)  # [batch_size, vocab_size]
        energy_contrib = energy_contrib * mask.squeeze(-1).float()
        energy = energy_contrib.sum(dim=1)  # [batch_size]

        # Predict per-atom stress contributions and sum
        stress_contrib = self.stress_output(x)  # [batch_size, vocab_size, 6]
        stress_contrib = stress_contrib * new_mask.float()
        stress = stress_contrib.sum(dim=1)  # [batch_size, 6]

        return forces, energy, stress