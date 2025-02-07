import os
import sys

import pandas as pd

from config.companies import companies
from src.utils.dataloader import DataLoader


def main():
    # CSV 파일들을 저장할 디렉토리 지정
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    # companies 딕셔너리를 순회하며 데이터 다운로드
    for company, code in companies.items():
        dl = DataLoader(code=code, start="2015")
        df = dl.expand_df()
        save_path = f"{data_dir}/{company}_{dl.start}.csv"
        df.to_csv(save_path, index=False)
        print(f"Saved {company} data to {save_path}")


if __name__ == "__main__":
    main()
