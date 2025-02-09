import pandas as pd

from config.user_settings import setting


class Environment:
    PRICE_IDX = setting["PRICE_IDX"]

    def __init__(
        self, chart_data: pd.DataFrame = None, window: int = setting["window"]
    ):
        self.chart_data = chart_data
        self.observation = None
        self.window = window
        self.done = 0
        self.idx = -1

    def reset(self):
        self.observation = None
        self.done = 0
        self.idx = -1

    def observe(self):
        if len(self.chart_data) - self.window >= self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[
                self.idx : self.idx + self.window
            ]
            self.done = (
                0 if len(self.chart_data) - self.window > self.idx else 1
            )
            return self.observation
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation.iloc[-1, self.PRICE_IDX]
        return None
