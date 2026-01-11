import torch 
import torch
from torch import nn, Tensor, optim
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections import namedtuple

import time
from datetime import timedelta
from tqdm import tqdm


class LSTMPredictor(nn.Module):
    def __init__(self, type_size, emb_size, hid_size, n_layers=2, is_reverse=False ):
        super().__init__()

        #self.features  = None # 用來暫存 hook 收集到的特徵 (Tensor)
        #self.feature_ch = sum([self.emb_size, self.hid_size])  # 代表融合後總共多少特徵通道 Layer1(128) + Layer2(256)
        #self.target_size = target_size #控制最後輸出的特徵圖大小。
        self.is_reverse = is_reverse

        self.embedding = nn.Embedding(
            num_embeddings = type_size, #有幾個類別要嵌入
            embedding_dim = emb_size, #每個詞向量(嵌入向量)的維度
            padding_idx = 31  # padding_idx 設定為 35，這是類別數量 + 1
        )

        self.lstm = nn.LSTM(
            input_size = emb_size,
            hidden_size = hid_size,
            num_layers = n_layers, #LSTM層數 第二層LSTM會接收第一層的輸出並計算最終結果
            batch_first = True
        )

        self.fc = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(inplace = True),
            nn.Linear(hid_size, type_size)
        )

        # 給自己這層的 lstm 註冊 hook
        #self.lstm.register_forward_hook(self.hook)

        # 載入預訓練
        # try:
        #     state_dict = torch.load("weight/mylstm.pt", weights_only=True)
        #     #state_dict = torch.load("weight/mylstm_backward.pt", weights_only=True)
        #     self.load_state_dict(state_dict)
        #     print("已載入預訓練權重")
        # except FileNotFoundError:
        #     print("找不到預訓練權重，請重新訓練")


    # def hook(self, model, inp, out):
    #     #取得這層的輸出 out。
    #     #存入 self.features。
    #     if not self.training:
    #         self.features = out  # 不改變形狀，單純存下來
    
    def forward(self, x : Tensor, ballrounds: Tensor,q_len: Tensor, hx = None): #x代表我一個batch的資料
        # idx : Tensor of shape (N,)  每筆樣本對應的時間步索引 (取 LSTM 的特徵)
        N, L = x.shape #N為batch L為sequence length
        

        #每個詞變成一個 d 維的向量
        x = self.embedding(x) #將feature_dim變成embedding要的維度d x = (N, L, d)
     

        #x: 所有時間步的隱藏狀態輸出（每個時間點都輸出一次）->(N,L,h) hx:最後時間步的隱藏狀態（可用於下次繼續 RNN）
        lstm_output, _ = self.lstm(x, hx) #這邊會把嵌入過的序列丟進 RNN 或 LSTM 處理 # x: (N, L, h) not contiguous here
        lstm_output : Tensor

        #攤平成一維以便用於全連接層 ，-1 是 PyTorch 中 reshape 的自動推論功能，表示「這個維度讓我自己算」。
        x = lstm_output.reshape(N * L, -1)
        x = self.fc(x)
        x = x.view(N, L, -1) #將維度還原，x: (N, L, n_tokens)

        # 確保 q_len 是 long 類型
        q_len_1_based_tensor = q_len.long() #真正的sequence長度
        # # 從 lstm_output 中擷取每個 sample 對應 ballround 的特徵
        d = lstm_output.size(2) #隱藏層的維度 d
        # ballrounds = ballrounds.view(N, 1, 1).expand(N, 1, d) #(N, 1, d)

        # if not self.is_reverse:
        #     # 正向序列
        #     index = (ballrounds - 1).long()  # 轉成 int64  
        #     #feature = torch.gather(lstm_output, dim=1, index=index)

        #     #index = ballrounds - 1  
        # else:
        #     # 反向序列：最後一個擊球在反轉後是開頭，所以用 (q_len - round)
        #     index = (L - ballrounds).long()
        
        
        
        
        # feature = torch.gather(lstm_output, dim=1, index=index)  # shape: (N, 1, d)

        with torch.no_grad():
            if ballrounds == None:
                if not self.is_reverse:
                    index_base = q_len_1_based_tensor - 1
                else:
                    index_base = q_len_1_based_tensor - q_len_1_based_tensor
            
            else:
                #先計算出正確的 0-based 索引 (N,)
                ballrounds_1_based = ballrounds.long() # 確保是 long 類型

                if not self.is_reverse:
                    # 正向序列：索引是 ballrounds - 1
                    # 例如 ballrounds=[5, 33] -> index_base=[4, 32] (N,)
                    index_base = ballrounds_1_based - 1
                else:
                    # 反向序列：索引是 L - ballrounds
                    # 例如 L=50, ballrounds=[5, 33] -> index_base=[45, 17] (N,))
                    index_base = q_len_1_based_tensor - ballrounds_1_based
            
            #然後將這個 (N,) 形狀的索引張量轉換為 torch.gather 所需的 (N, 1, d) 形狀
            # unsqueeze(1) -> (N, 1)
            # unsqueeze(2) -> (N, 1, 1)
            # expand(N, 1, d) -> (N, 1, d)
            index_for_gather = index_base.unsqueeze(1).unsqueeze(2).expand(N, 1, d)
            
            feature = torch.gather(lstm_output, dim=1, index=index_for_gather)  # shape: (N, 1, d)

            # # Eval模式才用hook或其他中間特徵邏輯
            # if not self.training:
            #     output = self.features
            #     self.features = None #清空 
            #     return x, output

       
        return x, feature

