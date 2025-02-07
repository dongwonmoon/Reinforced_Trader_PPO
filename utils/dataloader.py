from typing import List

import FinanceDataReader as fdr
import pandas as pd
from ta import add_all_ta_features


class DataLoader:
    def __init__(self, code: str, start: str = "2000") -> None:
        """
        DataLoader 인스턴스를 초기화합니다.

        매개변수:
            code (str): 주요 데이터셋의 티커 심볼 또는 코드.
            start (str): 데이터 조회 시작 날짜. 기본값은 "2000".
        """
        self.code: str = code
        self.start: str = start
        self.df: pd.DataFrame = pd.DataFrame()
        self.df_origin: pd.DataFrame = pd.DataFrame()

        # 추가 데이터 소스 리스트: 지수 및 국채 수익률
        self.indices: List[str] = [
            "KS11",
            "KQ11",
            "KS200",
            "DJI",
            "IXIC",
            "S&P500",
            "RUT",
            "VIX",
            "N225",
            "SSEC",
        ]
        self.yt: List[str] = ["US5YT", "US10YT", "US30YT"]

    def read_data(self) -> None:
        """
        주어진 code와 start 날짜를 이용해 메인 데이터셋을 읽어옵니다.
        읽어온 데이터는 Date 컬럼을 datetime 형식으로 변환하고 인덱스로 설정한 후,
        self.df 및 self.df_origin에 저장합니다.
        """
        df = fdr.DataReader(self.code, self.start)
        df = df.reset_index().rename(columns={"index": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        self.df = df.copy()
        self.df_origin = df.copy()

    def _interpolate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        주어진 DataFrame의 결측치를 보간(interpolate)하고 전후 결측치를 채웁니다.

        매개변수:
            df (pd.DataFrame): 입력 DataFrame.

        반환:
            pd.DataFrame: 보간 및 결측치 채움이 완료된 DataFrame.
        """
        return df.interpolate().bfill().ffill()

    def append_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        입력 DataFrame에 기술적 분석(TA) 피처를 추가하고,
        특정 컬럼(예: trend_psar_up, trend_psar_down)을 제거한 후 결측치를 처리합니다.

        매개변수:
            df (pd.DataFrame): 처리할 DataFrame.

        반환:
            pd.DataFrame: 기술적 분석 피처가 추가된 업데이트된 DataFrame.
        """
        df = df.copy()
        df = add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
        )
        # 데이터 분포에 영향을 줄 수 있는 컬럼 제거
        columns_to_drop = ["trend_psar_up", "trend_psar_down"]
        df.drop(columns=columns_to_drop, errors="ignore", inplace=True)
        df = self._interpolate_df(df)
        return df

    def _fetch_external_data(self, ticker: str) -> pd.DataFrame:
        """
        주어진 티커의 외부 데이터를 조회하고, 컬럼 이름에 티커를 접미사로 추가합니다.

        매개변수:
            ticker (str): 외부 데이터의 티커 심볼.

        반환:
            pd.DataFrame: 컬럼명이 수정된 외부 데이터 DataFrame.
        """
        external_df = fdr.DataReader(ticker, self.start)
        external_df.rename(columns=lambda col: f"{col}_{ticker}", inplace=True)
        return external_df

    def append_external_data(
        self, df: pd.DataFrame, codes: List[str]
    ) -> pd.DataFrame:
        """
        주어진 DataFrame에 외부 데이터를 병합합니다.
        codes 리스트에 있는 각 티커 심볼에 대해 외부 데이터를 조회하여,
        컬럼 단위로 병합(concatenate)하고 결측치를 보간합니다.

        매개변수:
            df (pd.DataFrame): 기본 DataFrame.
            codes (List[str]): 외부 티커 심볼 리스트.

        반환:
            pd.DataFrame: 외부 데이터가 추가된 통합 DataFrame.
        """
        df = df.copy()
        for ticker in codes:
            ext_df = self._fetch_external_data(ticker)
            # 외부 데이터를 컬럼 기준으로 병합
            df = pd.concat([df, ext_df], axis=1)
            df = self._interpolate_df(df)
        return df

    def expand_df(self) -> pd.DataFrame:
        """
        원본 DataFrame을 확장하는 과정:
            1. 기본 데이터셋을 읽어옴.
            2. 기술적 분석 피처를 추가함.
            3. 외부 지수 및 국채 수익률 데이터를 병합함.
            4. 원본 날짜만 유지하도록 필터링함.

        반환:
            pd.DataFrame: 확장 및 통합된 DataFrame.
        """
        # 1단계: 기본 데이터 읽기
        self.read_data()
        original_dates = self.df.index

        # 2단계: 기술적 분석 피처 추가
        df = self.append_ta_features(self.df)

        # 3단계: 외부 지수 및 국채 수익률 데이터 추가
        df = self.append_external_data(df, self.indices)
        df = self.append_external_data(df, self.yt)

        # 4단계: 원본 날짜만 유지하도록 필터링 및 재정렬
        df = df[df.index.isin(original_dates)]
        df = df.reindex(original_dates)

        self.df = df
        return df


if __name__ == "__main__":
    code = "005930"  # 삼성전자
    start_date = "2000"
    loader = DataLoader(code, start_date)
    expanded_df = loader.expand_df()
    print("확장된 데이터프레임의 상위 5개 행:")
    print(expanded_df.head())
