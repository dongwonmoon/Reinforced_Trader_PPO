import datetime
import time

import numpy as np
import torch

# 날짜, 시간 관련 문자열 형식
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"


def get_today_str():
    today = datetime.datetime.combine(
        datetime.date.today(), datetime.datetime.min.time()
    )
    today_str = today.strftime("%Y%m%d")
    return today_str


def get_time_str():
    return datetime.datetime.fromtimestamp(int(time.time())).strftime(
        FORMAT_DATETIME
    )


def sigmoid(x):
    x = max(min(x, 10), -10)
    return 1.0 / (1.0 + np.exp(-x))


def min_max_scaling(x):
    # 1/3 => 1/num_actions
    return (x - 1 / 3) / (1 - 1 / 3)


def combine_feature_with_scalars(feature, agent_state):
    agent_state_tensor = torch.tensor(
        agent_state, dtype=feature.dtype, device=feature.device
    )

    return torch.cat([feature.squeeze(), agent_state_tensor], dim=0).unsqueeze(
        0
    )


def map_companies_to_paths(data_dir="./data"):
    """
    Maps companies to their corresponding CSV file paths in the given data directory.
    """
    paths = os.listdir(data_dir)
    return {
        key: next(
            ("./data/" + path for path in paths if path.startswith(str(key))),
            None,
        )
        for key in companies.companies.keys()
    }
