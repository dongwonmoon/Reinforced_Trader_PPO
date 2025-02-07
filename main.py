import concurrent.futures
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from config import companies
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
    """Trading system main class that manages the training and execution process"""

    def __init__(
        self,
        sim_config: SimulationConfig = SimulationConfig(),
        trainer_config: TrainerConfig = TrainerConfig(),
    ):
        self.sim_config = sim_config
        self.trainer_config = trainer_config
        self.logger = TradingLogger()
        self.trainer = self._initialize_trainer()

        self.save_model_freq = 4

    def _initialize_trainer(self) -> PPO:
        """Initialize the SAC trainer with configuration parameters"""
        return PPO(
            state_dim=self.trainer_config.state_dim,
            agent_state_dim=self.trainer_config.agent_state_dim,
            num_actions=self.trainer_config.num_actions,
        )

    def get_data_paths(self) -> Dict[str, str]:
        """Retrieve company data file paths"""
        if not os.path.exists(self.sim_config.data_dir):
            raise FileNotFoundError(
                f"데이터 디렉토리를 찾을 수 없습니다: {self.sim_config.data_dir}"
            )

        files = os.listdir(self.sim_config.data_dir)
        return {
            key: next(
                (
                    os.path.join(self.sim_config.data_dir, file)
                    for file in files
                    if file.startswith(str(key))
                ),
                None,
            )
            for key in companies.companies.keys()
        }

    def run_agent_episode(
        self, company: str, initial_balance: float, data_paths: Dict[str, str]
    ) -> Agent:
        """Execute a single episode for a company"""
        file_path = data_paths
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(
                f"회사 {company}의 데이터 파일을 찾을 수 없습니다: {file_path}"
            )

        env = Environment(pd.read_csv(file_path))
        agent = Agent(env, initial_balance)

        observation = env.observe().values
        agent_state = list(agent.get_states())
        done = False

        while not done:
            action, action_probs = self.trainer.select_action(observation, agent_state)
            reward = agent.act(action, action_probs)
            self.logger.metrics.returns.append(reward)
            next_observation = env.observe().values
            next_agent_state = list(agent.get_states())

            observation = next_observation
            agent_state = next_agent_state
            done = env.done

            self.trainer.buffer.rewards.append(reward)
            self.trainer.buffer.dones.append(done)

        self.logger.log_portfolio_performance(
            company, initial_balance, agent.portfolio_value
        )
        loss = self.trainer.update()

        return loss

    def run(self) -> None:
        """Execute the main training loop"""
        data_paths = self.get_data_paths()
        companies_list = list(data_paths.keys())

        for epoch in range(self.sim_config.num_epochs):
            self.current_epoch = epoch

            losses = []
            for company in companies_list:
                loss = self.run_agent_episode(
                    company,
                    np.random.randint(
                        self.sim_config.min_initial_balance,
                        self.sim_config.max_initial_balance,
                    ),
                    data_paths[company],
                )
                losses.append(loss)
            epoch_loss = sum(losses) / len(losses)

            # Save progress
            self.trainer.memory.reset()
            self.logger.log_training_metrics(epoch, epoch_loss)
            if epoch % self.save_model_freq == 0:
                self.trainer.save()


def main() -> None:
    """Main entry point for the trading system"""
    trading_system = TradingSystem()
    trading_system.run()


if __name__ == "__main__":
    main()
