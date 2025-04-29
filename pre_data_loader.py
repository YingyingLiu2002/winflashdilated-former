import sys
import os
# 明确指定项目根目录的绝对路径
project_root = '/root/autodl-tmp/fwin_experiment'
sys.path.append(project_root)
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
from utils.timefeatures import time_features  # 假设与 data_loader.py 共享此模块

# 定义变量缩写
variable_mapping = {
    'timestamp': 'T',
    'Active_Power': 'AP',
    'Diffuse_Horizontal_Radiation': 'DHR',
    'Global_Horizontal_Radiation': 'GHR',
    'Radiation_Diffuse_Tilted': 'RDT',
    'Radiation_Global_Tilted': 'RGT',
    'Weather_Temperature_Celsius': 'WTC'
}

# 选择主要影响因素作为输入特征
input_features = ['Diffuse_Horizontal_Radiation', 'Global_Horizontal_Radiation',
                  'Radiation_Diffuse_Tilted', 'Radiation_Global_Tilted', 'Weather_Temperature_Celsius']
target_var = 'Active_Power'


def preprocess_data(input_path, output_filename='32CanadianSolar_processed.csv', freq='5min'):
    """
    预处理光伏数据集，生成适合后续模型输入的CSV文件，仅保留主要影响因素。

    参数：
    - input_path: 原始数据集路径（例如 '/root/autodl-tmp/fwin_experiment/dataset/32CanadianSolar.csv'）
    - output_filename: 预处理后数据集的文件名（默认 '32CanadianSolar_processed.csv'）
    - freq: 数据采样频率（默认 '5min'）

    返回：
    - df_processed: 预处理后的DataFrame
    - scaler: 用于归一化的MinMaxScaler对象
    """
    # 读取数据
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 筛选主要特征和目标变量
    selected_cols = ['timestamp'] + input_features + [target_var]
    df = df[selected_cols]

    # 检查并填补缺失时间点，保持 DatetimeIndex
    full_index = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq=freq)
    df = df.set_index('timestamp').reindex(full_index)

    # 缺失值处理（基于 DatetimeIndex）
    df['Active_Power'] = df['Active_Power'].interpolate(method='time').fillna(0)  # 夜间功率补0
    for col in input_features:
        df[col] = df[col].interpolate(method='time')

    # 异常值处理
    df['Active_Power'] = df['Active_Power'].clip(lower=0)
    for col in ['Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation',
                'Radiation_Global_Tilted', 'Radiation_Diffuse_Tilted']:
        df[col] = df[col].clip(lower=0)

    # 重置索引以添加时间特征
    df = df.reset_index().rename(columns={'index': 'timestamp'})

    # 时间特征提取（使用 timeenc=0，与 data_loader.py 一致）
    df_stamp = df[['timestamp']]
    time_feat = time_features(df_stamp, timeenc=0, freq=freq)  # [month, day, weekday, hour, minute, is_weekend]
    time_cols = ['month', 'day', 'weekday', 'hour', 'minute', 'is_weekend']
    for i, col in enumerate(time_cols):
        df[col] = time_feat[:, i].astype(int)  # 确保为整数

    # 标准化（仅对输入特征和目标变量进行归一化，不包括时间特征）
    scaler = MinMaxScaler()
    numeric_cols_to_scale = input_features + [target_var]
    df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
    # 在保存之前添加检查
    if df.isna().any().any():
        print("Warning: Dataset contains NaN values!")
        print(df.isna().sum())
    if np.isinf(df[numeric_cols_to_scale].values).any():
        print("Warning: Dataset contains Inf values!")

    # 保存预处理结果和scaler
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
    root_path = '/root/autodl-tmp/fwin_experiment/dataset'
    input_filename = '32CanadianSolar.csv'
    input_path = os.path.join(root_path, input_filename)

    # 运行预处理
    df_processed, scaler = preprocess_data(input_path)
    print("预处理后的数据预览：")
    print(df_processed.head())