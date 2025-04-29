import sys
import os
# 明确指定项目根目录的绝对路径
project_root = '/root/'
sys.path.append(project_root)
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
from utils.timefeatures import time_features  

variable_mapping = {
    'timestamp': 'T',
    'Active_Power': 'AP',
    'Diffuse_Horizontal_Radiation': 'DHR',
    'Global_Horizontal_Radiation': 'GHR',
    'Radiation_Diffuse_Tilted': 'RDT',
    'Radiation_Global_Tilted': 'RGT',
    'Weather_Temperature_Celsius': 'WTC'
}

input_features = ['Diffuse_Horizontal_Radiation', 'Global_Horizontal_Radiation',
                  'Radiation_Diffuse_Tilted', 'Radiation_Global_Tilted', 'Weather_Temperature_Celsius']
target_var = 'Active_Power'


def preprocess_data(input_path, output_filename='.csv', freq='5min'):

    """
    # 读取数据
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    selected_cols = ['timestamp'] + input_features + [target_var]
    df = df[selected_cols]
    full_index = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq=freq)
    df = df.set_index('timestamp').reindex(full_index)
    df['Active_Power'] = df['Active_Power'].interpolate(method='time').fillna(0)  # 夜间功率补0
    for col in input_features:
        df[col] = df[col].interpolate(method='time')
    df['Active_Power'] = df['Active_Power'].clip(lower=0)
    for col in ['Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation',
                'Radiation_Global_Tilted', 'Radiation_Diffuse_Tilted']:
        df[col] = df[col].clip(lower=0)
    df = df.reset_index().rename(columns={'index': 'timestamp'})
    df_stamp = df[['timestamp']]
    time_feat = time_features(df_stamp, timeenc=0, freq=freq)  # [month, day, weekday, hour, minute, is_weekend]
    time_cols = ['month', 'day', 'weekday', 'hour', 'minute', 'is_weekend']
    for i, col in enumerate(time_cols):
        df[col] = time_feat[:, i].astype(int) 
    scaler = MinMaxScaler()
    numeric_cols_to_scale = input_features + [target_var]
    df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
    if df.isna().any().any():
        print("Warning: Dataset contains NaN values!")
        print(df.isna().sum())
    if np.isinf(df[numeric_cols_to_scale].values).any():
        print("Warning: Dataset contains Inf values!")

   
    path = Path(input_path)
    output_path = path.parent / output_filename
    df.to_csv(output_path, index=False)
    scaler_path = path.parent / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)

    print(f"预处理完成，数据已保存至：{output_path}")
    print(f"Scaler已保存至：{scaler_path}")
    return df, scaler


if __name__ == '__main__':
    # 配置路径
    root_path = '/root/dataset'
    input_filename = 'Solar.csv'
    input_path = os.path.join(root_path, input_filename)

    # 运行预处理
    df_processed, scaler = preprocess_data(input_path)
    print("预处理后的数据预览：")
    print(df_processed.head())
