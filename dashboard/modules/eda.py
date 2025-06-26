import os
from dashboard.modules.utils import SystemUtils
import pandas as pd


class FraudDataEDA:

    def __init__(self, filename = 'synthetic_transaction_data'):
        base_path = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base_path))
        self.file = os.path.join(project_root, 'dashboard/data', f'{filename}.csv')

    def extract_eda_df(self):
        fds_raw_df = SystemUtils.load_csv_files(self.file)
        return fds_raw_df

    def extract_rule_combo(self):
        fds_raw_df = self.extract_eda_df()

        #1. 조건 정의
        fds_raw_df['is_high_amt'] = fds_raw_df['TransactionAmt'] >= 700000
        fds_raw_df['is_high_ratio'] = fds_raw_df['amount_ratio_to_bank_avg'] >= 2.0
        fds_raw_df['has_risk_factor'] = (fds_raw_df['vpn'] == True) | (fds_raw_df['is_new_device'] == True) | (fds_raw_df['rooting'] == True)

        fds_raw_df['rule_combo']=(
                fds_raw_df['is_high_amt'].astype(int).astype(str)+'_' +
                fds_raw_df['is_high_ratio'].astype(int).astype(str) + '_' +
                fds_raw_df['has_risk_factor'].astype(int).astype(str)
        )
        rule_based_df = fds_raw_df
        # rule_based_df = fds_raw_df[fds_raw_df['rule_combo'] == '1_1_1']
        #fds_raw_df.to_csv("rule_combined_fds2.csv", index=False, encoding="utf-8-sig")
        # print('*'*100)
        # print(rule_based_df)

        return rule_based_df

    # def transaction_count(self):
    #     self.eda_df['transaction_time'] = pd.to_datetime(self.eda_df['transaction_time'])
    #     self.eda_df['month'] = self.eda_df['transaction_time'].dt.to_period('M')
    #
    #     return self.eda_df

eda = FraudDataEDA()
print(eda.extract_rule_combo())
# eda.transaction_count()
