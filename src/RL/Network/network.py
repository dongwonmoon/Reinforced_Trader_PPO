import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config.user_settings import network_setting


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.TransformerEncoderLayer):
        for name, param in module.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        agent_state_dim: int,
        num_actions: int,
        d_model: int = network_setting["d_model"],
        actor_hidden_dim: int = network_setting["actor_hidden_dim"],
        critic_hidden_dim: int = network_setting["critic_hidden_dim"],
        transformer_layers: int = network_setting["transformer_layers"],
        nhead: int = network_setting["nhead"],
        dropout: float = network_setting["dropout"],
        max_seq_length: int = network_setting["max_seq_length"],
    ) -> None:
        super(ActorCritic, self).__init__()
        self.input_linear = nn.Linear(state_dim, d_model)
        self.max_seq_length = max_seq_length
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model), requires_grad=True
        )
        nn.init.xavier_uniform_(self.pos_embedding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.actor = nn.Sequential(
            nn.Linear(d_model + agent_state_dim, actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(actor_hidden_dim, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(d_model + agent_state_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, 1),
        )
        self.apply(initialize_weights)

    def _get_feature(
        self, state: torch.Tensor, agent_state: torch.Tensor
    ) -> torch.Tensor:
        state_projected = self.input_linear(state)
        seq_len = state_projected.size(1)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        state_with_pos = state_projected + pos_emb
        trans_out = self.transformer_encoder(state_with_pos)
        norm_out = self.norm(trans_out)
        feature = norm_out.mean(dim=1)
        feature = torch.nan_to_num(feature, nan=0.0, posinf=1e6, neginf=-1e6)
        feature = self.dropout(feature)
        return torch.cat([feature, agent_state], dim=1)

    def forward(
        self, state: torch.Tensor, agent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self._get_feature(state, agent_state)
        logits = self.actor(feature)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e1, neginf=-1e1)
        action_probs = F.softmax(logits, dim=-1)
        state_value = self.critic(feature)
        return action_probs, state_value

    def act(
        self, state: torch.Tensor, agent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs, state_value = self.forward(state, agent_state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return (
            action.detach(),
            action_probs.detach(),
            action_logprob.detach(),
            state_value.detach(),
        )

    def evaluate(
        self, state: torch.Tensor, agent_state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self._get_feature(state, agent_state)
        logits = self.actor(feature)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e1, neginf=-1e1)
        action_probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(feature)
        return action_logprobs, state_values, dist_entropy
