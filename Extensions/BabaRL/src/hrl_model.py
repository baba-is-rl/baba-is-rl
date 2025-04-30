import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import MAX_RULES, NUM_OBJECT_TYPES, PAD_INDEX, NUM_OPTIONS


class HierarchicalActorCritic(nn.Module):
    def __init__(
        self,
        grid_shape,
        num_actions,         
        num_options,         
        rule_emb_dim=64,
        transformer_heads=4,
        transformer_layers=1,
        cnn_out_channels=64,
        hidden_dim=256,
        option_emb_dim=32    
    ):
        super(HierarchicalActorCritic, self).__init__()

        grid_channels, _, _ = grid_shape
        self.num_actions = num_actions
        self.num_options = num_options
        self.option_emb_dim = option_emb_dim

        self.conv1 = nn.Conv2d(grid_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, cnn_out_channels, kernel_size=3, stride=1, padding=1)
        self.cnn_flat_size = self._get_cnn_output_size(grid_shape, cnn_out_channels)
        self.cnn_linear = nn.Linear(self.cnn_flat_size, hidden_dim // 2)

        self.rule_part_embedding = nn.Embedding(
            NUM_OBJECT_TYPES, rule_emb_dim, padding_idx=PAD_INDEX
        )
        transformer_input_dim = rule_emb_dim * 3
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_input_dim * 2,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.rule_linear = nn.Linear(transformer_input_dim, hidden_dim // 2)

        fusion_input_dim = (hidden_dim // 2) * 2
        self.fusion_linear = nn.Linear(fusion_input_dim, hidden_dim)

        self.manager_actor_head = nn.Linear(hidden_dim, num_options) 
        self.manager_critic_head = nn.Linear(hidden_dim, 1)          

        self.option_embedding = nn.Embedding(num_options, option_emb_dim)

        worker_input_dim = hidden_dim + option_emb_dim
        self.worker_shared_layer = nn.Linear(worker_input_dim, hidden_dim) 

        self.worker_actor_head = nn.Linear(hidden_dim, num_actions) 
        self.worker_critic_head = nn.Linear(hidden_dim, 1)         

    def _get_cnn_output_size(self, shape, out_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.size()[1:]))

    def _get_shared_features(self, grid, rules, rule_mask):
        
        x_cnn = F.relu(self.conv1(grid))
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = F.relu(self.conv3(x_cnn))
        x_cnn = x_cnn.view(x_cnn.size(0), -1)
        x_cnn = F.relu(self.cnn_linear(x_cnn))

        batch_size = rules.size(0)
        rule_parts_embedded = self.rule_part_embedding(rules)
        rule_embedded = rule_parts_embedded.view(batch_size, MAX_RULES, -1)
        attention_mask = rule_mask == 0 
        transformer_output = self.transformer_encoder(
            rule_embedded, src_key_padding_mask=attention_mask
        )
        mask_expanded = rule_mask.unsqueeze(-1).float()
        summed_output = (transformer_output * mask_expanded).sum(dim=1)
        valid_rule_count = rule_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        rule_context = summed_output / valid_rule_count
        rule_context = F.relu(self.rule_linear(rule_context))

        fused = torch.cat((x_cnn, rule_context), dim=1)
        shared_features = F.relu(self.fusion_linear(fused))
        return shared_features

    def forward(self, grid, rules, rule_mask, current_option):

        shared_features = self._get_shared_features(grid, rules, rule_mask)

        manager_logits = self.manager_actor_head(shared_features)
        manager_value = self.manager_critic_head(shared_features)

        option_emb = self.option_embedding(current_option) 

        worker_input_features = torch.cat((shared_features, option_emb), dim=1)
        worker_hidden = F.relu(self.worker_shared_layer(worker_input_features))

        worker_logits = self.worker_actor_head(worker_hidden)
        worker_value = self.worker_critic_head(worker_hidden)

        return worker_logits, worker_value, manager_logits, manager_value

    def get_worker_action_value(self, grid, rules, rule_mask, current_option):
        shared_features = self._get_shared_features(grid, rules, rule_mask)
        option_emb = self.option_embedding(current_option)
        worker_input_features = torch.cat((shared_features, option_emb), dim=1)
        worker_hidden = F.relu(self.worker_shared_layer(worker_input_features))
        return self.worker_actor_head(worker_hidden), self.worker_critic_head(worker_hidden)

    def get_manager_option_value(self, grid, rules, rule_mask):
        shared_features = self._get_shared_features(grid, rules, rule_mask)
        return self.manager_actor_head(shared_features), self.manager_critic_head(shared_features)

    def get_manager_value(self, grid, rules, rule_mask):
         shared_features = self._get_shared_features(grid, rules, rule_mask)
         return self.manager_critic_head(shared_features)

    def get_worker_value(self, grid, rules, rule_mask, current_option):
        shared_features = self._get_shared_features(grid, rules, rule_mask)
        option_emb = self.option_embedding(current_option)
        worker_input_features = torch.cat((shared_features, option_emb), dim=1)
        worker_hidden = F.relu(self.worker_shared_layer(worker_input_features))
        return self.worker_critic_head(worker_hidden)