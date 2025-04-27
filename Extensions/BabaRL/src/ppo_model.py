import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import MAX_RULES, NUM_OBJECT_TYPES, PAD_INDEX


class ActorCritic(nn.Module):
    def __init__(
        self,
        grid_shape,
        num_actions,
        rule_emb_dim=64,
        transformer_heads=4,
        transformer_layers=1,
        cnn_out_channels=64,
        hidden_dim=256,
    ):
        super(ActorCritic, self).__init__()

        grid_channels, _, _ = grid_shape
        self.num_actions = num_actions

        # Grid Encoder (CNN)
        self.conv1 = nn.Conv2d(grid_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, cnn_out_channels, kernel_size=3, stride=1, padding=1)
        self.cnn_flat_size = self._get_cnn_output_size(grid_shape, cnn_out_channels)
        self.cnn_linear = nn.Linear(self.cnn_flat_size, hidden_dim // 2)

        # Rule Encoder (Embedding + Transformer)
        self.rule_part_embedding = nn.Embedding(
            NUM_OBJECT_TYPES, rule_emb_dim, padding_idx=PAD_INDEX
        )
        # Each rule is (N, O, T), so 3 embeddings concatenated
        transformer_input_dim = rule_emb_dim * 3

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_input_dim * 2,
            batch_first=True,  # Input shape (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.rule_linear = nn.Linear(transformer_input_dim, hidden_dim // 2)

        fusion_input_dim = (hidden_dim // 2) * 2
        self.fusion_linear1 = nn.Linear(fusion_input_dim, hidden_dim)
        self.fusion_linear2 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Output Heads
        self.actor_head = nn.Linear(hidden_dim // 2, num_actions)  # Policy logits
        self.critic_head = nn.Linear(hidden_dim // 2, 1)  # Value estimate

    def _get_cnn_output_size(self, shape, out_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.size()[1:]))  # Calculate flattened size

    def forward(self, grid, rules, rule_mask):
        # grid: (Batch, C, H, W)
        # rules: (Batch, MAX_RULES, 3) - indices
        # rule_mask: (Batch, MAX_RULES) - 0 for pad, 1 for valid

        # CNN
        # print("Grid shape in:", grid.shape)
        x_cnn = F.relu(self.conv1(grid))
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = F.relu(self.conv3(x_cnn))
        x_cnn = x_cnn.view(x_cnn.size(0), -1)
        # print("CNN flat shape:", x_cnn.shape)
        x_cnn = F.relu(self.cnn_linear(x_cnn))

        # Rule Transformer
        # print("Rules shape in:", rules.shape) # Debug
        # print("Rule Mask shape in:", rule_mask.shape) # Debug
        batch_size = rules.size(0)
        # Embed rule parts: (B, MAX_RULES, 3) -> (B, MAX_RULES, 3, EmbDim)
        rule_parts_embedded = self.rule_part_embedding(rules)
        # Concatenate N, O, T embeddings: -> (B, MAX_RULES, 3 * EmbDim)
        rule_embedded = rule_parts_embedded.view(batch_size, MAX_RULES, -1)
        # print("Rule embedded shape:", rule_embedded.shape) # Debug

        # Create attention mask for transformer (True where padded)
        # Transformer expects mask where True indicates positions TO BE IGNORED
        # Our rule_mask is 1 for valid, 0 for pad. So we need to invert it.
        attention_mask = rule_mask == 0
        # print("Attention mask shape:", attention_mask.shape) # Debug
        # print("Attention mask example:", attention_mask[0]) # Debug

        # Transformer expects (Batch, Seq, Feature) due to batch_first=True
        transformer_output = self.transformer_encoder(
            rule_embedded, src_key_padding_mask=attention_mask
        )
        # print("Transformer output shape:", transformer_output.shape) # Debug

        # Masked Mean Pooling: Average only the valid rule outputs
        mask_expanded = rule_mask.unsqueeze(-1).float()
        # Sum valid outputs: (B, MAX_RULES, Feature) * (B, MAX_RULES, 1) -> sum over dim 1
        summed_output = (transformer_output * mask_expanded).sum(dim=1)
        valid_rule_count = rule_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        rule_context = summed_output / valid_rule_count  # (B, Feature)
        # print("Rule context shape:", rule_context.shape)
        rule_context = F.relu(self.rule_linear(rule_context))

        # print("CNN features shape:", x_cnn.shape)
        # print("Rule context shape:", rule_context.shape)
        fused = torch.cat((x_cnn, rule_context), dim=1)
        # print("Fused shape:", fused.shape)
        x_fused = F.relu(self.fusion_linear1(fused))
        x_fused = F.relu(self.fusion_linear2(x_fused))

        # Heads 
        action_logits = self.actor_head(x_fused)
        value = self.critic_head(x_fused)

        return action_logits, value
