import os
import pandas as pd


class SystemUtils:
    @staticmethod
    def load_csv_files(file_path):
        print(file_path)

        # file_path = f'{folder_path}\dataset\{file_name}.csv'
        # print(file_path)
        if not os.path.exists(file_path):
            print(f"[WARNING] 파일이 존재하지 않습니다")
            return None

        try:
            df =pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"[ERROR]  읽기 실패:{e}")
            return None