# graph/graph_utils.py

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler

def build_graph_from_df(df: pd.DataFrame, edge_cols=['card1', 'addr1']) -> Data:
    """
    거래 데이터를 기반으로 PyG용 그래프 객체(Data)를 생성한다.
    동일한 card1/addr1 값을 공유하는 거래 간 edge를 생성한다.
    """
    df = df.copy()
    df['node_id'] = np.arange(len(df))

    # 1. 엣지 구성 (예: 같은 card1끼리 엣지 생성)
    edges = []
    for col in edge_cols:
        if col not in df.columns:
            continue
        col_groups = df.groupby(col)['node_id'].apply(list)
        for group in col_groups:
            for i in range(len(group) - 1):
                edges.append([group[i], group[i + 1]])

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).T  # shape: [2, num_edges]

    # 2. 노드 특징 구성
    # 기본적으로 수치형 feature만 사용
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(['TransactionID'])
    scaler = StandardScaler()
    x = scaler.fit_transform(df[numeric_cols].fillna(0))
    x = torch.tensor(x, dtype=torch.float)

    # 3. 라벨 (이상거래 여부)
    y = torch.tensor(df['isFraud'].values, dtype=torch.long) if 'isFraud' in df.columns else None

    return Data(x=x, edge_index=edge_index, y=y)
