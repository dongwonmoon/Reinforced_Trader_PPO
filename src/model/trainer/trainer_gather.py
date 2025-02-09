from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.user_settings import trainer_setting

from ..Network import ActorCritic
from .rollout_buffer import RolloutBuffer


class PPO:
    def __init__(
        self,
        state_dim: int,
        agent_state_dim: int,
        num_actions: int,
        lr_encoder: float = trainer_setting["lr_encoder"],
        lr_actor: float = trainer_setting["lr_actor"],
        lr_critic: float = trainer_setting["lr_critic"],
        gamma: float = trainer_setting["gamma"],
        K_epochs: int = trainer_setting["K_epochs"],
        batch_size: int = trainer_setting["batch_size"],
        eps_clip: float = trainer_setting["eps_clip"],
    ) -> None:
        self.gamma: float = gamma
        self.eps_clip: float = eps_clip
        self.K_epochs: int = K_epochs
        self.batch_size: int = batch_size
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.buffer: RolloutBuffer = RolloutBuffer()

        self.policy: ActorCritic = ActorCritic(
            state_dim, agent_state_dim, num_actions
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.policy.encoder.parameters(),
                    "lr": lr_encoder,
                },
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old: ActorCritic = ActorCritic(
            state_dim, agent_state_dim, num_actions
        ).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss: nn.Module = nn.MSELoss()

    def update(self) -> torch.Tensor:
        # Use torch.stack to create tensors from the stored list of tensors.
        old_states = torch.cat(self.buffer.states).detach().to(self.device)
        old_agent_states = (
            torch.cat(self.buffer.agent_states).detach().to(self.device)
        )
        old_actions = torch.cat(self.buffer.actions).detach().to(self.device)
        old_logprobs = torch.cat(self.buffer.logprobs).detach().to(self.device)
        old_state_values = (
            torch.cat(self.buffer.state_values).detach().to(self.device)
        )
        rewards = torch.cat(self.buffer.rewards).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()

        # Ensure that rewards and advantages are shaped correctly by using TensorDataset.
        dataset = TensorDataset(
            old_states,
            old_agent_states,
            old_actions,
            old_logprobs,
            rewards,
            advantages,
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        losses: list[torch.Tensor] = []
        for epoch in range(self.K_epochs):
            for (
                mini_old_states,
                mini_old_agent_states,
                mini_old_actions,
                mini_old_logprobs,
                mini_rewards,
                mini_advantages,
            ) in dataloader:
                # Evaluate the current policy on the mini-batch
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    mini_old_states, mini_old_agent_states, mini_old_actions
                )
                state_values = torch.squeeze(state_values)

                # Calculate the probability ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(logprobs - mini_old_logprobs.detach())

                # Compute surrogate losses via clipping strategy
                surr1 = ratios * mini_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * mini_advantages
                )
                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * self.mse_loss(state_values, mini_rewards)
                    - 0.01 * dist_entropy
                )

                # Gradient descent step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                losses.append(loss.mean())
            # Update the old policy to the new policy after each epoch
            self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear the rollout buffer after update
        self.buffer.clear()

        return sum(losses) / len(losses)

    def save(
        self, checkpoint_path: str = trainer_setting["checkpoint_path"]
    ) -> None:
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(
        self, checkpoint_path: str = trainer_setting["checkpoint_path"]
    ) -> None:
        self.policy_old.load_state_dict(
            torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
        )
        self.policy.load_state_dict(
            torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )
        )
