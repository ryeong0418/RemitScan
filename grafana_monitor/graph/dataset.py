# graph/dataset.py

import os
import pandas as pd
from torch_geometric.data import Dataset, Data
from graph.graph_utils import build_graph_from_df

class FraudGraphDataset(Dataset):
    def __init__(self, csv_path, transform=None, pre_transform=None):
        """
        Args:
            csv_path (str): 전처리된 CSV 파일 경로 (예: client_0.csv)
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.pre_transform = pre_transform

        # 그래프 데이터는 전체를 하나로 보므로 len은 1
        self.graph_data = build_graph_from_df(self.df)
        
        self._indices = None

    def len(self):
        return 1

    def get(self, idx):
        assert idx == 0, "FraudGraphDataset은 단일 그래프만을 포함합니다."
        return self.graph_data
