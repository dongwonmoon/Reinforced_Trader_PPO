from typing import Tuple
import numpy as np
import torch

from config.user_settings import trading
from utils import utils


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

        매개변수:
            environment: 환경 정보 (예: reset, observe, get_price 메서드 포함)
            initial_balance (float): 초기 계좌 잔액
        """
        self.environment = environment
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.num_stocks = 0
        self.portfolio_value = initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0

        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0

        # 상태 차원: 잔고, 포트폴리오 가치, 손익, 평균 매수 단가
        self.state_dim = len(self.get_states())

    def reset(self) -> None:
        """에이전트의 상태를 초기값으로 재설정합니다."""
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0

    def set_balance(self, balance: float) -> None:
        """초기 잔액을 변경합니다."""
        self.initial_balance = balance

    def get_states(self) -> Tuple[float, float, float, float]:
        """
        에이전트의 현재 상태를 반환합니다.

        반환값:
            (잔고, 포트폴리오 가치, 손익, 평균 매수 단가)
        """
        return (self.balance, self.portfolio_value, self.profitloss, self.avg_buy_price)

    def decide_action(self, pred_policy: np.ndarray) -> int:
        """
        예측된 정책을 기반으로 행동을 결정합니다.

        매개변수:
            pred_policy (np.ndarray): 행동 확률 배열

        반환값:
            int: 선택된 행동 인덱스
        """
        return int(np.argmax(pred_policy))

    def validate_action(self, action: int) -> bool:
        """
        현재 상태를 바탕으로 선택된 행동의 유효성을 확인합니다.

        매개변수:
            action (int): 행동 인덱스

        반환값:
            bool: 유효한 행동이면 True, 아니면 False
        """
        if action == Agent.ACTION_BUY:
            # 현재 가격에 거래 수수료 포함 시 잔고가 부족하면 매수 불가
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE):
                return False
        elif action == Agent.ACTION_SELL:
            # 보유 주식이 없으면 매도 불가
            if self.num_stocks <= 0:
                return False
        return True

    def decide_buy_unit(self, confidence: float) -> int:
        """
        매수 시 거래 단위를 결정합니다.

        매개변수:
            confidence (float): 정규화된 확률 값

        반환값:
            int: 매수할 주식 수
        """
        curr_price = self.environment.get_price()
        max_unit = int(self.balance / (curr_price * (1 + self.TRADING_CHARGE)))
        return int(confidence * max_unit)

    def decide_sell_unit(self, confidence: float) -> int:
        """
        매도 시 거래 단위를 결정합니다.

        매개변수:
            confidence (float): 정규화된 확률 값

        반환값:
            int: 매도할 주식 수
        """
        trading_unit = int(confidence * self.num_stocks)
        if trading_unit < 0:
            raise ValueError(f"잘못된 매도 단위: {trading_unit}")
        return trading_unit

    def _update_portfolio_state(self, curr_price: float) -> None:
        """
        포트폴리오 가치 및 손익률을 업데이트합니다.

        매개변수:
            curr_price (float): 현재 주가
        """
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = self.portfolio_value / self.initial_balance - 1

    def _handle_buy(self, curr_price: float, confidence: float) -> None:
        """
        매수 행동을 처리합니다.

        매개변수:
            curr_price (float): 현재 주가
            confidence (float): 정규화된 확률 값
        """
        trading_unit = self.decide_buy_unit(confidence)
        invest_cost = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
        self.balance -= invest_cost

        if invest_cost > 0:
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

        매개변수:
            curr_price (float): 현재 주가
            confidence (float): 정규화된 확률 값
        """
        trading_unit = self.decide_sell_unit(confidence)
        invest_gain = (
            curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
        )

        if invest_gain > 0 and trading_unit > 0:
            if self.num_stocks > trading_unit:
                total_cost = (
                    self.avg_buy_price * self.num_stocks - curr_price * trading_unit
                )
                self.num_stocks -= trading_unit
                self.avg_buy_price = (
                    total_cost / self.num_stocks if self.num_stocks > 0 else 0
                )
            else:
                # 보유 주식을 모두 매도하는 경우
                self.num_stocks = 0
                self.avg_buy_price = 0

            self.balance += invest_gain
            self.num_sell += 1

    def _handle_hold(self) -> None:
        """관망 행동을 처리합니다."""
        self.num_hold += 1

    def act(self, action: int, prob: torch.tensor) -> float:
        """
        선택된 행동을 실행하고 포트폴리오 상태를 업데이트합니다.

        매개변수:
            action (int): 선택된 행동 인덱스
            prob (torch.tensor): 각 행동의 확률을 담은 텐서 (형태: (1, 행동 차원))

        반환값:
            float: 실행 후 손익률
        """
        action_prob = prob[0][action]
        confidence = utils.min_max_scaling(action_prob).numpy()

        # 선택한 행동이 유효하지 않으면 관망 행동으로 전환
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        curr_price = self.environment.get_price()

        if action == Agent.ACTION_BUY:
            self._handle_buy(curr_price, confidence)
        elif action == Agent.ACTION_SELL:
            self._handle_sell(curr_price, confidence)
        elif action == Agent.ACTION_HOLD:
            self._handle_hold()

        self._update_portfolio_state(curr_price)
        return self.profitloss
