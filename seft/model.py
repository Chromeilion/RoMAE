import torch
import torch.nn as nn
import torch.nn.functional as F


class SEFTConfig:
    def __init__(self,
                 tubelet_size: tuple[int, int, int] = (1, 1, 16),
                 d_model: int = 768,
                 depth: int = 12,
                 nhead: int = 12,
                 dim_feedforward: int = 3072,
                 activation: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 max_len: int = 1500,
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.,
                 mlp_ratio: float = 4.,
                 attn_drop_rate: float = 0.,
                 norm_layer = nn.LayerNorm,
                 init_values: float = 0.,
                 head_drop_rate: float = 0.,
                 num_classes: int = 0,
                 init_scale: float = 0.0,
                 *args, **kwargs):
        self.tubelet_size = tubelet_size
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.max_len = max_len
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.init_values = init_values
        self.init_scale = init_scale
        self.head_drop_rate = head_drop_rate
        self.num_classes = num_classes


class SEFT(nn.Module):
    def __init__(self, config: SEFTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.d_model % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with "
                             f"an odd dim ({config.d_model})")
        self.config: SEFTConfig = config
        self.num_classes = config.num_classes
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        transformer_layer = nn.TransformerEncoderLayer(
            config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            activation=F.gelu,
            batch_first=True,
            dropout=config.attn_drop_rate
        )
        self.encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=config.depth
        )
        self.fc_norm = config.norm_layer(config.d_model)
        self.head_dropout = nn.Dropout(config.head_drop_rate)
        self.head = nn.Linear(config.d_model, config.num_classes)
        self.projection = nn.Linear(config.tubelet_size[0] * config.tubelet_size[1], config.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(config.max_len, config.d_model))
        self.cls = nn.Parameter(torch.zeros(config.d_model))

    def patchify(self, x):
        """
        Expected input format: BTHW
        Converts the input video-like format to a sequence of patches
        """
        B, T, H, W = x.shape

        T_p = T // self.config.tubelet_size[0]
        H_p = H // self.config.tubelet_size[1]
        W_p = W // self.config.tubelet_size[2]
        N = T_p * H_p * W_p
        x = x.reshape(B, T_p, H_p, W_p, *self.config.tubelet_size)
        return x.reshape(B, N, -1)

    def forward(self, values, positions, pad_mask, label=None):
        B = values.shape[0]
        # Here we add one to all the positions and masks because we're
        # inserting the cls token at the start.
        positions += 1
        zeros = torch.zeros((B, 1), device=positions.device, dtype=torch.long)
        positions = torch.cat([zeros, positions], dim=1)
        pad_mask = torch.cat([zeros.bool(), pad_mask], dim=1)

        x = self.patchify(values)
        x = self.projection(x)
        x = torch.cat([self.cls.expand(B, 1, -1), x], dim=1)
        x += self.pos_embedding[positions]
        x = self.pos_drop(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.fc_norm(x[:, 0, :])
        x = self.head_dropout(x)
        logits = self.head(x)

        loss = None
        if label is not None:
            loss = F.cross_entropy(logits, label)

        return logits, loss
