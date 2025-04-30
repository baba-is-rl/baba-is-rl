import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import MAX_RULES, NUM_OBJECT_TYPES, PAD_INDEX

class SubgoalHierarchicalActorCritic(nn.Module):
    def __init__(
        self,
        grid_shape,          
        num_actions,         
        rule_emb_dim=64,
        transformer_heads=4,
        transformer_layers=1,
        cnn_out_channels=64,
        hidden_dim=256,
        generator_hidden_channels=64 
    ):
        super(SubgoalHierarchicalActorCritic, self).__init__()

        grid_channels, grid_height, grid_width = grid_shape
        self.num_actions = num_actions
        self.grid_shape = grid_shape
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(grid_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, cnn_out_channels, kernel_size=3, stride=1, padding=1)
        self.cnn_flat_size = self._get_cnn_output_size(grid_shape, cnn_out_channels)

        self.rule_part_embedding = nn.Embedding(
            NUM_OBJECT_TYPES, rule_emb_dim, padding_idx=PAD_INDEX
        )
        transformer_input_dim = rule_emb_dim * 3
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim, nhead=transformer_heads,
            dim_feedforward=transformer_input_dim * 2, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.rule_linear = nn.Linear(transformer_input_dim, hidden_dim // 4) 

        manager_fusion_input_dim = self.cnn_flat_size + (hidden_dim // 4)
        self.manager_fusion_linear = nn.Linear(manager_fusion_input_dim, hidden_dim)

        self.manager_critic_head = nn.Linear(hidden_dim, 1)

        s1_h = self._calc_input_dim_for_convtranspose(grid_height, k=4, s=2, p=1)
        s1_w = self._calc_input_dim_for_convtranspose(grid_width, k=4, s=2, p=1)
        s0_h = self._calc_input_dim_for_convtranspose(s1_h, k=4, s=2, p=1)
        s0_w = self._calc_input_dim_for_convtranspose(s1_w, k=4, s=2, p=1)

        s0_h = max(1, s0_h)
        s0_w = max(1, s0_w)

        self.generator_linear = nn.Linear(hidden_dim, generator_hidden_channels * s0_h * s0_w)
        self.generator_reshape_dims = (generator_hidden_channels, s0_h, s0_w)

        self.generator_deconv1 = nn.ConvTranspose2d(generator_hidden_channels, generator_hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        self.generator_deconv2 = nn.ConvTranspose2d(generator_hidden_channels // 2, grid_channels, kernel_size=4, stride=2, padding=1)

        worker_fusion_input_dim = self.cnn_flat_size * 2 + (hidden_dim // 4) 
        self.worker_fusion_linear1 = nn.Linear(worker_fusion_input_dim, hidden_dim)
        self.worker_fusion_linear2 = nn.Linear(hidden_dim, hidden_dim // 2)

        self.worker_actor_head = nn.Linear(hidden_dim // 2, num_actions)
        self.worker_critic_head = nn.Linear(hidden_dim // 2, 1)

    def _get_cnn_output_size(self, shape, out_channels):
        with torch.no_grad():
            c, h, w = shape
            temp_conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)
            temp_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            temp_conv3 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
            dummy_input = torch.zeros(1, *shape)
            x = F.relu(temp_conv1(dummy_input))
            x = F.relu(temp_conv2(x))
            x = F.relu(temp_conv3(x))
            return int(np.prod(x.size()[1:])) 

    def _calc_input_dim_for_convtranspose(self, target_dim, k, s, p):
        return (target_dim + 2 * p - k) // s + 1

    def _get_base_features(self, grid):
        x_cnn = F.relu(self.conv1(grid))
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = F.relu(self.conv3(x_cnn))
        x_cnn_flat = x_cnn.view(x_cnn.size(0), -1)
        return x_cnn_flat 

    def _get_rule_context(self, rules, rule_mask):
        batch_size = rules.size(0)
        if batch_size == 0:
            return torch.zeros(0, self.hidden_dim // 4, device=rules.device)

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
        return rule_context

    def get_manager_value_and_subgoal(self, grid, rules, rule_mask):
        u_features_flat = self._get_base_features(grid)
        rule_context = self._get_rule_context(rules, rule_mask)

        manager_input_features = torch.cat((u_features_flat, rule_context), dim=1)
        manager_hidden = F.relu(self.manager_fusion_linear(manager_input_features))

        manager_value = self.manager_critic_head(manager_hidden)

        gen_lin_out = F.relu(self.generator_linear(manager_hidden))
        gen_reshaped = gen_lin_out.view(gen_lin_out.size(0), *self.generator_reshape_dims)
        gen_deconv1 = F.relu(self.generator_deconv1(gen_reshaped))
        generated_subgoal_w = torch.tanh(self.generator_deconv2(gen_deconv1)) 

        return manager_value, generated_subgoal_w

    def get_worker_action_value(self, current_grid_u, rules_u, rule_mask_u, target_grid_w):
        u_features_flat = self._get_base_features(current_grid_u)
        w_features_flat = self._get_base_features(target_grid_w)
        rule_context = self._get_rule_context(rules_u, rule_mask_u)

        worker_input_features = torch.cat((u_features_flat, w_features_flat, rule_context), dim=1)
        worker_hidden = F.relu(self.worker_fusion_linear1(worker_input_features))
        worker_hidden = F.relu(self.worker_fusion_linear2(worker_hidden))

        worker_logits = self.worker_actor_head(worker_hidden)
        worker_value = self.worker_critic_head(worker_hidden)

        return worker_logits, worker_value

    def get_manager_value(self, grid, rules, rule_mask):
         u_features_flat = self._get_base_features(grid)
         rule_context = self._get_rule_context(rules, rule_mask)
         manager_input_features = torch.cat((u_features_flat, rule_context), dim=1)
         manager_hidden = F.relu(self.manager_fusion_linear(manager_input_features))
         return self.manager_critic_head(manager_hidden)

    def get_worker_value(self, current_grid_u, rules_u, rule_mask_u, target_grid_w):
         u_features_flat = self._get_base_features(current_grid_u)
         w_features_flat = self._get_base_features(target_grid_w)
         rule_context = self._get_rule_context(rules_u, rule_mask_u)
         worker_input_features = torch.cat((u_features_flat, w_features_flat, rule_context), dim=1)
         worker_hidden = F.relu(self.worker_fusion_linear1(worker_input_features))
         worker_hidden = F.relu(self.worker_fusion_linear2(worker_hidden))
         return self.worker_critic_head(worker_hidden)
