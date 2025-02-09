from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.user_settings import network_setting

from .encoder import TransformerEncoder


def initialize_weights(module: nn.Module) -> None:
    """
    Initialize weights for Linear and TransformerEncoderLayer modules using Xavier uniform
    for weights and constant zero for biases.
    """
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
    """
    ActorCritic network that processes state and agent state
    via an encoder and returns action probabilities, state value, and actions.
    """

    def __init__(
        self,
        state_dim: int,
        agent_state_dim: int,
        num_actions: int,
        temperature: float,
        d_model: int = network_setting["d_model"],
        actor_hidden_dim: int = network_setting["actor_hidden_dim"],
        critic_hidden_dim: int = network_setting["critic_hidden_dim"],
        transformer_layers: int = network_setting["transformer_layers"],
        nhead: int = network_setting["nhead"],
        dropout: float = network_setting["dropout"],
        max_seq_length: int = network_setting["max_seq_length"],
    ) -> None:
        super(ActorCritic, self).__init__()
        self.temperature = temperature

        # Initialize encoder for processing state
        self.encoder = TransformerEncoder(
            state_dim,
            d_model,
            nhead,
            transformer_layers,
            max_seq_length,
            dropout,
        )

        # Define the actor module that outputs logits for action selection.
        self.actor = nn.Sequential(
            nn.Linear(d_model + agent_state_dim, actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(actor_hidden_dim, num_actions),
        )

        # Define the critic module that outputs state value estimate.
        self.critic = nn.Sequential(
            nn.Linear(d_model + agent_state_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, 1),
        )
        # Apply weight initialization
        self.apply(initialize_weights)

    def _get_feature(
        self, state: torch.Tensor, agent_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Process state through the encoder, add positional embedding and concatenate with agent_state.
        Uses torch.nan_to_num to replace non-finite numbers.
        """
        feature = self.encoder(state)
        feature = torch.nan_to_num(feature, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.cat([feature, agent_state], dim=1)

    def _compute_actor_output(
        self, feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute logits using the actor network and convert to action probabilities.
        Returns a tuple of (logits, action_probs).
        """
        logits = self.actor(feature)
        action_probs = F.softmax(logits / self.temperature, dim=-1)
        return logits, action_probs

    def forward(
        self, state: torch.Tensor, agent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the network.
        Returns (action_probs, state_value).
        """
        feature = self._get_feature(state, agent_state)
        logits, action_probs = self._compute_actor_output(feature)
        state_value = self.critic(feature)
        return action_probs, state_value

    def act(
        self, state: torch.Tensor, agent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action based on the action probability distribution.
        Returns a tuple of (action, action_probs, action_logprob, state_value) with detached tensors.
        """
        action_probs, state_value = self.forward(state, agent_state)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        action_logprob = distribution.log_prob(action)
        return (
            action.detach(),
            action_probs.detach(),
            action_logprob.detach(),
            state_value.detach(),
        )

    def evaluate(
        self,
        state: torch.Tensor,
        agent_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the given action for specified state and agent_state.
        Computes the log probability of the action, the state value, and the entropy of the distribution.
        """
        feature = self._get_feature(state, agent_state)
        _, action_probs = self._compute_actor_output(feature)
        distribution = torch.distributions.Categorical(action_probs)
        action_logprobs = distribution.log_prob(action)
        dist_entropy = distribution.entropy()
        state_values = self.critic(feature)
        return action_logprobs, state_values, dist_entropy
