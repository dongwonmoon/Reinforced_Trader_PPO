from typing import Tuple

import torch
import torch.nn as nn

from config.user_settings import network_setting


def initialize_weights(module: nn.Module) -> None:
    """
    선형(Linear), 1차원 합성곱(Conv1d), GRU 레이어에 대해 Xavier uniform 방식으로 가중치를 초기화합니다.

    매개변수:
        module (nn.Module): 초기화를 수행할 모듈.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        agent_state_dim: int,
        num_actions: int,
        actor_hidden_dim: int = network_setting["actor_hidden_dim"],
        critic_hidden_dim: int = network_setting["critic_hidden_dim"],
        transformer_layers: int = network_setting["transformer_layers"],
        nhead: int = network_setting["nhead"],
        dropout: float = network_setting["dropout"],
    ) -> None:
        """
        ActorCritic 네트워크 초기화

        매개변수:
            state_dim (int): 환경 상태 차원
            agent_state_dim (int): 에이전트 상태 차원
            num_actions (int): 가능한 행동의 수
            actor_hidden_dim (int): 액터(hidden) 레이어 차원
            critic_hidden_dim (int): 크리틱(hidden) 레이어 차원
            transformer_layers (int): Transformer 인코더 레이어 수
            nhead (int): Transformer의 헤드 수
            dropout (float): 드롭아웃 확률
        """
        super(ActorCritic, self).__init__()

        # Transformer를 이용한 Feature Extractor
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=state_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        # Actor 네트워크 구성: 상태와 에이전트 상태를 결합하여 행동 확률 산출
        self.actor: nn.Sequential = nn.Sequential(
            nn.Linear(state_dim + agent_state_dim, actor_hidden_dim),
            nn.Mish(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.Mish(),
            nn.Linear(actor_hidden_dim, num_actions),
            nn.Softmax(dim=-1),
        )

        # Critic 네트워크 구성: 상태와 에이전트 상태를 결합하여 상태 가치 산출
        self.critic: nn.Sequential = nn.Sequential(
            nn.Linear(state_dim + agent_state_dim, critic_hidden_dim),
            nn.Mish(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.Mish(),
            nn.Linear(critic_hidden_dim, 1),
        )

        # 가중치 초기화
        self.apply(initialize_weights)

    def _get_feature(
        self, state: torch.Tensor, agent_state: torch.Tensor
    ) -> torch.Tensor:
        """
        입력된 상태와 에이전트 상태를 기반으로 특성(feature)을 추출합니다.

        매개변수:
            state (torch.Tensor): 환경 상태 텐서 (batch, seq_len, state_dim)
            agent_state (torch.Tensor): 에이전트 상태 텐서 (batch, agent_state_dim)

        반환값:
            torch.Tensor: 추출된 특성 텐서
        """
        # Transformer 인코더를 통해 sequence로부터 특성 추출
        trans_out: torch.Tensor = self.transformer_encoder(state)
        feature: torch.Tensor = trans_out.mean(dim=1)  # 평균 풀링
        feature = torch.nan_to_num(feature, nan=0.0, posinf=1e6, neginf=-1e6)
        feature = self.dropout(feature)

        # 에이전트 상태와 결합
        feature_agent: torch.Tensor = torch.cat([feature, agent_state], dim=1)
        return feature_agent

    def forward(
        self, state: torch.Tensor, agent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파(forward) 메서드 구현.

        매개변수:
            state (torch.Tensor): 환경 상태 텐서 (batch, seq_len, state_dim)
            agent_state (torch.Tensor): 에이전트 상태 텐서 (batch, agent_state_dim)

        반환값:
            Tuple[torch.Tensor, torch.Tensor]: (actor의 행동 확률, critic의 상태 가치)
        """
        feature: torch.Tensor = self._get_feature(state, agent_state)
        action_probs: torch.Tensor = self.actor(feature)
        state_value: torch.Tensor = self.critic(feature)
        return action_probs, state_value

    def act(
        self, state: torch.Tensor, agent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        행동을 선택합니다.

        매개변수:
            state (torch.Tensor): 환경 상태 텐서 (batch, seq_len, state_dim)
            agent_state (torch.Tensor): 에이전트 상태 텐서 (batch, agent_state_dim)

        반환값:
            Tuple:
                - 선택한 행동 (torch.Tensor, detach 된 상태)
                - 행동 로그 확률 (torch.Tensor, detach 된 상태)
                - 상태 가치 (torch.Tensor, detach 된 상태)
        """
        action_probs, state_value = self.forward(state, agent_state)
        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            action_probs
        )
        action: torch.Tensor = dist.sample()
        action_logprob: torch.Tensor = dist.log_prob(action)
        return (
            action.detach(),
            action_probs.detach(),
            action_logprob.detach(),
            state_value.detach(),
        )

    def evaluate(
        self, state: torch.Tensor, agent_state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        주어진 상태와 행동에 대해 평가를 수행합니다.

        매개변수:
            state (torch.Tensor): 환경 상태 텐서 (batch, seq_len, state_dim)
            agent_state (torch.Tensor): 에이전트 상태 텐서 (batch, agent_state_dim)
            action (torch.Tensor): 선택된 행동 텐서

        반환값:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (행동 로그 확률, 상태 가치, 분포 엔트로피)
        """
        feature: torch.Tensor = self._get_feature(state, agent_state)
        action_probs: torch.Tensor = self.actor(feature)
        dist: torch.distributions.Categorical = torch.distributions.Categorical(
            action_probs
        )
        action_logprobs: torch.Tensor = dist.log_prob(action)
        dist_entropy: torch.Tensor = dist.entropy()
        state_values: torch.Tensor = self.critic(feature)
        return action_logprobs, state_values, dist_entropy
