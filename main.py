import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from config.companies import companies
from logger import TradingLogger
from src.Agent.agent import Agent
from src.Agent.environment import Environment
from src.RL.trainer import PPO


@dataclass
class SimulationConfig:
    num_epochs: int = 100
    min_initial_balance: int = 10_000_000
    max_initial_balance: int = 100_000_000
    data_dir: str = "./data"


@dataclass
class TrainerConfig:
    state_dim: int = 180
    agent_state_dim: int = 4
    num_actions: int = 3
    update_timestep: int = 4


class TradingSystem:
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
            np.clip(X.values, -1e8, 1e8),
            columns=X.columns,
        )
        y = df["Close"]

        # Normalize features and recombine with target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_scaled = pd.concat(
            [pd.DataFrame(X_scaled, columns=X.columns), y.reset_index(drop=True)],
            axis=1,
        )

        env = Environment(df_scaled)
        agent = Agent(env, initial_balance)

        observation = env.observe().values
        agent_state = list(agent.get_states())

        # Simulation loop for the episode
        while not env.done:
            action, action_probs = self.trainer.select_action(observation, agent_state)
            reward = agent.act(action, action_probs)

            # Record reward and termination status in trainer buffer
            self.trainer.buffer.rewards.append(reward)
            self.trainer.buffer.dones.append(env.done)

            # Update observation and state for next iteration
            observation = env.observe().values
            agent_state = list(agent.get_states())

        self.logger.metrics.returns.append(reward)
        self.logger.log_portfolio_performance(
            company, initial_balance, agent.portfolio_value
        )
        loss = self.trainer.update()

        return loss

    def run(self) -> None:
        """
        Execute the main training loop over the specified number of epochs.
        """
        data_paths = self.get_data_paths()
        company_list = list(data_paths.keys())

        for epoch in range(self.sim_config.num_epochs):
            self.current_epoch = epoch
            losses = []

            # Iterate over companies and simulate each episode
            for company in company_list:
                file_path = data_paths.get(company)
                initial_balance = np.random.randint(
                    self.sim_config.min_initial_balance,
                    self.sim_config.max_initial_balance,
                )

                loss = self.simulate_episode(company, initial_balance, file_path)
                losses.append(loss)

            epoch_loss = sum(losses) / len(losses)
            self.trainer.buffer.clear()
            self.logger.log_training_metrics(epoch, epoch_loss)

            # Periodically save the trainer model
            if epoch % self.save_model_freq == 0:
                self.trainer.save()


def main() -> None:
    """
    Main entry point for the trading system.
    """
    trading_system = TradingSystem()
    trading_system.run()


if __name__ == "__main__":
    main()
