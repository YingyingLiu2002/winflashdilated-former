import sys
import os

# 将项目根目录添加到 Python 的模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='32CanadianSolar_processed.csv',
                 target='Active_Power', scale=True, inverse=False, timeenc=0, freq='5min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 12
            self.label_len = 12 * 12
            self.pred_len = 12 * 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols  # 这里的 self.cols 是初始化时的参数，不是特征列
        self.root_path = '/root/autodl-tmp/WinFlashDetailed_MultiScale/dataset'
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        all_cols = list(df_raw.columns)

        # 选择列
        physical_cols = ['Diffuse_Horizontal_Radiation', 'Global_Horizontal_Radiation',
                         'Radiation_Diffuse_Tilted', 'Radiation_Global_Tilted',
                         'Weather_Temperature_Celsius']
        cols = [col for col in physical_cols if col in all_cols]
        if self.target in cols:
            cols.remove(self.target)
        self.feature_cols = cols
        df_raw = df_raw[['timestamp'] + cols + [self.target]]
        # 数据划分
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 数据划分和缩放
        if self.scale:
            print("⚠ WARNING: Data is already scaled in preprocessing, skipping additional scaling.")
            data_x = df_raw[cols].values.astype(np.float32)  # 转换为 float32
            data_y = df_raw[[self.target]].values.astype(np.float32)  # 转换为 float32
        else:
            self.scaler.fit(df_raw[cols].values)
            data_x = self.scaler.transform(df_raw[cols].values).astype(np.float32)
            data_y = self.scaler.transform(df_raw[[self.target]].values).astype(np.float32)

        df_stamp = df_raw[['timestamp']][border1:border2]
        df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq).astype(np.float32)  # 转换为 float32

        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = data_stamp

        if data_stamp.shape[1] != 6:
            raise ValueError(f"Expected time features to have 6 channels, but got {data_stamp.shape[1]}")
        if np.isnan(self.data_x).any() or np.isinf(self.data_x).any():
            print(f"Warning: self.data_x contains NaN or Inf!")
        if np.isnan(self.data_y).any() or np.isinf(self.data_y).any():
            print(f"Warning: self.data_y contains NaN or Inf!")
        if np.isnan(self.data_stamp).any() or np.isinf(self.data_stamp).any():
            print(f"Warning: self.data_stamp contains NaN or Inf!")
        if len(self.data_x) != len(self.data_y) or len(self.data_x) != len(self.data_stamp):
            raise ValueError(
                f"Length mismatch: data_x ({len(self.data_x)}), data_y ({len(self.data_y)}), data_stamp ({len(self.data_stamp)})"
            )

    def __getitem__(self, index):
        if index >= len(self.data_x) - self.seq_len - self.pred_len + 1:
            raise IndexError(
                f"Index {index} out of range for sequence length {self.seq_len} and prediction length {self.pred_len}")

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len].astype(np.float32),
                 self.data_y[r_begin + self.label_len:r_end].astype(np.float32)], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 验证维度
        if seq_x.shape[1] != len(self.feature_cols):
            raise ValueError(f"Expected seq_x to have {len(self.feature_cols)} channels, but got {seq_x.shape[1]}")
        if seq_y.shape[1] != 1:
            raise ValueError(f"Expected seq_y to have 1 channel, but got {seq_y.shape[1]}")
        if seq_x_mark.shape[1] != 6 or seq_y_mark.shape[1] != 6:
            raise ValueError(
                f"Expected time marks to have 6 channels, but got {seq_x_mark.shape[1]} and {seq_y_mark.shape[1]}")

        if np.isnan(seq_x).any() or np.isinf(seq_x).any():
            print(f"Warning: seq_x at index {index} contains NaN or Inf!")
        if np.isnan(seq_y).any() or np.isinf(seq_y).any():
            print(f"Warning: seq_y at index {index} contains NaN or Inf!")
        if np.isnan(seq_x_mark).any() or np.isinf(seq_x_mark).any():
            print(f"Warning: seq_x_mark at index {index} contains NaN or Inf!")
        if np.isnan(seq_y_mark).any() or np.isinf(seq_y_mark).any():
            print(f"Warning: seq_y_mark at index {index} contains NaN or Inf!")

        # 转换为 torch.FloatTensor（float32）
        return (torch.FloatTensor(seq_x),
                torch.FloatTensor(seq_y),
                torch.FloatTensor(seq_x_mark),
                torch.FloatTensor(seq_y_mark))


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data).astype(np.float32)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='MS', data_path='32CanadianSolar_processed.csv',  # 更新为预处理后的数据集
                 target='Active_Power', scale=True, inverse=False, timeenc=0, freq='5min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 12  # 默认 24 小时 * 12 个 5 分钟间隔 = 288 个时间步
            self.label_len = 12 * 12  # 默认 12 小时 * 12 个 5 分钟间隔 = 144 个时间步
            self.pred_len = 12 * 12  # 默认预测 12 小时 = 144 个时间步
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ✅ 只选择输入特征，不包括 `timestamp` 和 `self.target`
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = [col for col in df_raw.columns if col not in ['timestamp', self.target]]

        df_raw = df_raw[['timestamp'] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        # ✅ 确保 `data_x` 只有 5 个通道，而 `time_features` 只用于 `data_stamp`
        df_data = df_raw[cols]  # ✅ 只选取 `self.cols` 指定的特征列

        if self.scale:
            self.scaler.fit(df_data.values)
            data = df_data.values
        else:
            data = df_data.values

        tmp_stamp = df_raw[['timestamp']][border1:border2]
        tmp_stamp['timestamp'] = pd.to_datetime(tmp_stamp.timestamp)
        pred_dates = pd.date_range(
            tmp_stamp.timestamp.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['timestamp'])
        df_stamp.timestamp = list(tmp_stamp.timestamp.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        print(f"Generated time features shape: {data_stamp.shape}")
        print(f"First few rows of generated time features:\n{data_stamp[:5]}")

        self.data_x = data[border1:border2]  # ✅ 现在 `data_x` 只有 `5` 个通道
        self.data_stamp = data_stamp  # ✅ `data_stamp` 只有时间特征

        if self.inverse:
            self.data_y = df_raw[[self.target]].values[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)