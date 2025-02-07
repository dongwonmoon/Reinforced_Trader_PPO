from typing import Any, List, Tuple

import torch
import torch.nn as nn

from ..RolloutBuffer import RolloutBuffer
from ..Network import ActorCritic

from config.user_settings import trainer_setting


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
        eps_clip: float = trainer_setting["eps_clip"],
    ) -> None:
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, agent_state_dim, num_actions).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.policy.transformer_encoder.parameters(),
                    "lr": lr_encoder,
                },
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(state_dim, agent_state_dim, num_actions).to(
            self.device
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()

    def select_action(self, state, agent_state) -> int:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            agent_state = torch.FloatTensor(agent_state).unsqueeze(0).to(self.device)

            action, action_probs, action_logprob, state_val = self.policy_old.act(
                state, agent_state
            )

        self.buffer.states.append(state)
        self.buffer.agent_states.append(agent_state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item(), action_probs

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.dones)
        ):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0))
            .detach()
            .to(self.device)
        )
        old_agent_states = (
            torch.squeeze(torch.stack(self.buffer.agent_states, dim=0))
            .detach()
            .to(self.device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(self.device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(self.device)
        )

        advantages = rewards.detach() - old_state_values.detach()

        losses = []

        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_agent_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.mse_loss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            losses.append(loss.mean())

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return sum(losses) / len(losses)

    def save(self, checkpoint_path: str = trainer_setting["checkpoint_path"]):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path: str = trainer_setting["checkpoint_path"]):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
