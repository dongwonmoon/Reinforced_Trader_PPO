import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config.companies import companies
from config.logger import TradingLogger
from src.agent import Agent
from src.environment import Environment
from src.model.trainer import RolloutBuffer
from src.model.trainer.trainer import PPO


@dataclass
class SimulationConfig:
    num_epochs: int = 100
    initial_balance: int = 10_000_000
    data_dir: str = "./data"


@dataclass
class TrainerConfig:
    state_dim: int = 180
    agent_state_dim: int = 4
    num_actions: int = 3
    temperature: float = 2
    temperature_decay: float = 0.95
    update_timestep: int = 4


def get_discounted_reward(rewards, dones, device):
    rewards_list: list[float] = []
    discounted_reward: float = 0.0
    # Reverse traversing ensures proper discounting, convert done to bool if needed.
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            discounted_reward = 0.0
        discounted_reward = float(reward) + (0.99 * discounted_reward)
        rewards_list.insert(0, discounted_reward)

    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
    return rewards


class Trainer:
    """
    Trading system main class that manages the training and execution process.
    """

    def __init__(
        self,
        sim_config: SimulationConfig = SimulationConfig(),
        trainer_config: TrainerConfig = TrainerConfig(),
    ) -> None:
        self.sim_config = sim_config
        self.trainer_config = trainer_config
        self.logger = TradingLogger()
        self.trainer = self._initialize_trainer()

        self.save_model_freq = 4
        self.current_epoch = 0

    def _initialize_trainer(self) -> PPO:
        """
        Initialize the PPO trainer with configuration parameters.
        """
        return PPO(
            state_dim=self.trainer_config.state_dim,
            agent_state_dim=self.trainer_config.agent_state_dim,
            num_actions=self.trainer_config.num_actions,
            temperature=self.trainer_config.temperature,
        )

    def get_data_paths(self) -> Dict[str, str]:
        """
        Retrieve company data file paths from the specified data directory.
        """
        if not os.path.exists(self.sim_config.data_dir):
            raise FileNotFoundError(
                f"데이터 디렉토리를 찾을 수 없습니다: {self.sim_config.data_dir}"
            )

        files = os.listdir(self.sim_config.data_dir)
        # Using companies as a dict; keys represent company identifiers.
        return {
            key: next(
                (
                    os.path.join(self.sim_config.data_dir, file)
                    for file in files
                    if file.startswith(key)
                ),
                None,
            )
            for key in companies.keys()
        }

    def simulate_episode(
        self, company: str, initial_balance: float, file_path: str
    ) -> float:
        """
        Executes a full simulation episode for a given company.

        Parameters:
            company (str): Company identifier.
            initial_balance (float): Starting balance for the agent.
            file_path (str): Path to the company's data file.

        Returns:
            float: The training loss after simulating the episode.
        """
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(
                f"회사 {company}의 데이터 파일을 찾을 수 없습니다: {file_path}"
            )

        df = pd.read_csv(file_path)
        if "Close" not in df.columns:
            raise ValueError(f"데이터 파일에 'Close' 열이 없습니다: {file_path}")

        # Separate features and target column
        X = df.drop(columns=["Close"])
        X = pd.DataFrame(
            np.clip(X.values, -1e12, 1e12),
            columns=X.columns,
        )
        y = df["Close"]

        # Normalize features and recombine with target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_scaled = pd.concat(
            [
                pd.DataFrame(X_scaled, columns=X.columns),
                y.reset_index(drop=True),
            ],
            axis=1,
        )

        env = Environment(df_scaled)
        agent = Agent(env, initial_balance)
        buffer = RolloutBuffer()

        # Simulation loop for the episode
        while not env.done:

            observation = env.observe().values
            agent_state = list(agent.get_states())
            with torch.no_grad():
                observation = (
                    torch.FloatTensor(observation).unsqueeze(0).to(self.trainer.device)
                )
                agent_state = (
                    torch.FloatTensor(agent_state).unsqueeze(0).to(self.trainer.device)
                )
                action, action_probs, action_logprob, state_val = (
                    self.trainer.policy.act(observation, agent_state)
                )
                reward = agent.act(action.item(), action_probs)
            buffer.push(
                observation,
                agent_state,
                action,
                action_logprob,
                state_val,
                reward,
                env.done,
            )

        rewards = get_discounted_reward(
            buffer.rewards, buffer.dones, self.trainer.device
        )
        # Record reward and termination status in trainer buffer
        self.trainer.buffer.push(
            torch.stack(buffer.states).squeeze(),
            torch.stack(buffer.agent_states).squeeze(),
            torch.stack(buffer.actions).squeeze(),
            torch.stack(buffer.logprobs).squeeze(),
            torch.stack(buffer.state_values).squeeze(),
            rewards,
            buffer.dones,
        )
        # Update observation and state for next iteration

        self.logger.metrics.returns.append(reward)
        self.logger.log_portfolio_performance(
            company, initial_balance, agent.portfolio_value
        )
        return agent.num_buy, agent.num_hold, agent.num_sell

    def run(self) -> None:
        """
        Execute the main training loop over the specified number of epochs.
        """
        data_paths = self.get_data_paths()
        company_list = list(data_paths.keys())

        losses = []
        for epoch in range(self.sim_config.num_epochs):
            self.current_epoch = epoch

            buys = 0
            holdings = 0
            sells = 0

            # Iterate over companies and simulate each episode
            for company in company_list:
                file_path = data_paths.get(company)
                initial_balance = self.sim_config.initial_balance

                b, h, s = self.simulate_episode(company, initial_balance, file_path)

                buys += b
                holdings += h
                sells += s

            print(f"Num Buys: {buys}, Num Holdings: {holdings}, Num Sells: {sells}")

            loss = self.trainer.update()
            losses.append(loss)

            self.trainer_config.temperature *= self.trainer_config.temperature_decay

            self.trainer.buffer.clear()
            self.logger.log_training_metrics(epoch, loss)

            # Periodically save the trainer model
            if epoch % self.save_model_freq == 0:
                self.trainer.save()


def main() -> None:
    """
    Main entry point for the trading system.
    """
    trading_system = Trainer()
    trading_system.run()


if __name__ == "__main__":
    main()
