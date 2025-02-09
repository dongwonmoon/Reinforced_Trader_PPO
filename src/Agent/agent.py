from typing import Tuple

import numpy as np
import torch

from config.user_settings import trading


class Agent:
    # 거래 관련 설정 값 (거래 수수료, 거래 세금)
    TRADING_CHARGE: float = trading["TRADING_CHARGE"]
    TRADING_TAX: float = trading["TRADING_TAX"]

    # 행동 정의 (매수, 매도, 관망)
    ACTION_BUY: int = 0
    ACTION_SELL: int = 1
    ACTION_HOLD: int = 2
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS: int = len(ACTIONS)

    def __init__(self, environment, initial_balance: float) -> None:
        """
        에이전트를 초기화합니다.

        Parameters:
            environment: 환경 정보 (예: reset, observe, get_price 메서드 포함)
            initial_balance (float): 초기 계좌 잔액
        """
        self.environment = environment
        self.initial_balance = initial_balance
        self._reset_state()

        # 상태 차원: 잔고, 포트폴리오 가치, 손익, 평균 매수 단가
        self.state_dim = len(self.get_states())

    def _reset_state(self) -> None:
        """에이전트의 거래 관련 상태를 초기화합니다."""
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.avg_buy_price = 0
        self.profitloss = 0
        self.ratio_hold = 0  # 추후 사용을 위해 남겨둠

    def reset(self) -> None:
        """에이전트의 상태를 초기값으로 재설정합니다."""
        self._reset_state()

    def set_balance(self, balance: float) -> None:
        """초기 잔액을 변경하고 상태를 재설정합니다."""
        self.initial_balance = balance
        self.reset()

    def get_states(self) -> Tuple[float, float, float, float]:
        """
        에이전트의 현재 상태를 반환합니다.

        return:
            (잔고, 포트폴리오 가치, 손익, 평균 매수 단가)
        """
        return (
            self.balance,
            self.portfolio_value,
            self.profitloss,
            self.avg_buy_price,
        )

    def decide_action(self, pred_policy: np.ndarray) -> int:
        """
        예측된 정책을 기반으로 행동을 결정합니다.

        parameters:
            pred_policy (np.ndarray): 행동 확률 배열
        return:
            int: Argmax policy
        """
        return int(np.argmax(pred_policy))

    def validate_action(self, action: int) -> bool:
        """
        현재 상태를 바탕으로 선택된 행동의 유효성을 확인합니다.

        parameters:
            action (int): 행동 인덱스
        return:
            bool: 유효한 행동이면 True, 그렇지 않으면 False
        """
        curr_price = self.environment.get_price()

        if action == Agent.ACTION_BUY:
            # 잔고가 부족하면 매수 불가
            if self.balance < curr_price * (1 + self.TRADING_CHARGE):
                return False
        elif action == Agent.ACTION_SELL:
            # 보유 주식이 없으면 매도 불가
            if self.num_stocks <= 0:
                return False
        return True

    def decide_buy_unit(self, confidence: float) -> int:
        """
        매수 시 거래 단위를 결정합니다.

        parameters:
            confidence (float): 정규화된 확률 값
        return:
            int: 매수할 주식 수
        """
        curr_price = self.environment.get_price()
        if self.balance <= 0:
            return 0
        max_unit = np.trunc(
            self.balance / (curr_price * (1 + self.TRADING_CHARGE))
        )
        assert max_unit >= 0

        return np.trunc(confidence * max_unit)

    def decide_sell_unit(self, confidence: float) -> int:
        """
        매도 시 거래 단위를 결정합니다.

        parameters:
            confidence (float): policy[action]
        return:
            int: 매도할 주식 수
        """
        trading_unit = np.trunc(confidence * self.num_stocks)

        assert trading_unit >= 0
        return trading_unit

    def _update_portfolio_state(self, curr_price: float) -> None:
        """
        포트폴리오 가치 및 손익률을 업데이트합니다.

        parameter:
            curr_price (float): 현재 주가
        """
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = self.portfolio_value / self.initial_balance - 1

    def _handle_buy(self, curr_price: float, confidence: float) -> None:
        """
        매수 행동을 처리합니다.

        parameter:
            curr_price (float): 현재 주가
            confidence (float): policy[action]
        """
        trading_unit = self.decide_buy_unit(confidence)
        invest_cost = curr_price * (1 + self.TRADING_CHARGE) * trading_unit

        assert invest_cost >= 0

        self.balance -= invest_cost

        # 평균 매수 단가 갱신
        total_cost = (
            self.avg_buy_price * self.num_stocks + curr_price * trading_unit
        )
        self.num_stocks += trading_unit
        self.avg_buy_price = (
            total_cost / self.num_stocks if self.num_stocks > 0 else 0
        )
        self.num_buy += 1

    def _handle_sell(self, curr_price: float, confidence: float) -> None:
        """
        매도 행동을 처리합니다.

        parameters:
            curr_price (float): 현재 주가
            confidence (float): 정규화된 확률 값
        """
        trading_unit = self.decide_sell_unit(confidence)

        invest_gain = (
            curr_price
            * (1 - (self.TRADING_TAX + self.TRADING_CHARGE))
            * trading_unit
        )
        assert invest_gain >= 0

        if self.num_stocks > trading_unit:
            total_cost = (
                self.avg_buy_price * self.num_stocks - curr_price * trading_unit
            )
            self.num_stocks -= trading_unit
            self.avg_buy_price = (
                total_cost / self.num_stocks if self.num_stocks > 0 else 0
            )
        else:
            # 모든 주식을 매도하는 경우
            self.num_stocks = 0
            self.avg_buy_price = 0

        self.balance += invest_gain
        self.num_sell += 1

    def _handle_hold(self) -> None:
        """관망 행동을 처리합니다."""
        self.num_hold += 1

    def act(self, action: int, prob: torch.Tensor) -> float:
        """
        선택된 행동을 실행하고 포트폴리오 상태를 업데이트합니다.

        parameter:
            action (int): 선택된 행동 인덱스
            prob (torch.Tensor): 각 행동의 확률을 담은 텐서 (형태: (1, 행동 차원))

        return:
            float: 실행 후 손익률
        """
        confidence = prob.squeeze()[action].item()

        # 선택한 행동이 유효하지 않으면 관망으로 전환
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        curr_price = self.environment.get_price()

        if action == Agent.ACTION_BUY:
            self._handle_buy(curr_price, confidence)
        elif action == Agent.ACTION_SELL:
            self._handle_sell(curr_price, confidence)
        else:  # Agent.ACTION_HOLD
            self._handle_hold()

        self._update_portfolio_state(curr_price)
        return self.profitloss
