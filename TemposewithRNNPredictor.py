import torch
from torch import nn, Tensor, optim
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections import namedtuple

import time
from datetime import timedelta
from tqdm import tqdm


# 定義球種與其對應的數值編碼 0~34  
class_ls = {
    "放小球": 1, "擋小球": 2, "殺球": 3, 
    "防守回挑": 4, "長球": 5, "平球": 6, "後場抽平球": 7, "切球": 8,
    "過渡切球": 9, "推挑": 10, "撲球": 11, "防守回抽": 12, "勾球": 13,
    "發短球": 14, "發長球": 15, "未知球種": 0
}
class_ls_inv = {v: k for k, v in class_ls.items()}
PADDING_TOKEN_ID = 31
Hyp = namedtuple('Hyp', [
    'serial_no', 'n_epochs', 'early_stop_n_epochs',
    'batch_size', 'lr', 'seq_len'
])
hyp = Hyp(
    serial_no = 1,
    n_epochs = 1200,
    early_stop_n_epochs= 50,
    batch_size =64,
    lr = 5e-4,
    seq_len = 30
)
def encode_type(row, class_ls):
    """
    根據球種和球員編碼。
    """
    base_code = class_ls.get(row['type'], class_ls["未知球種"])
    
    if row['player_pos'] == 'Bottom':
        base_code += 17
    return base_code

def get_stroke_types(data, class_ls):
    """
    將 DataFrame 中的 'type' 欄位轉換為數值編碼。
    """
    df_train = data[['rally','type', 'player_pos',]].copy()
    df_train['type_encode'] = df_train.apply(lambda row: encode_type(row, class_ls), axis=1)
    return df_train
# --- 從你的第一個檔案複製過來的類別定義 (假設這些類別存在並已正確導入) ---
# 由於這些類別 (MLP, TCN, TransformerEncoder, CrossTransformerLayer, MLP_Head, PositionalEncoding1D)
# 沒有在提供的片段中定義，為了讓程式碼可運行，我會假設它們是可用的。
# 如果你遇到這些類別未定義的錯誤，請確保它們被正確地定義在 'LSTM_main.py' 或其他導入的文件中。
class ShuttleSetData(Dataset):
    def __init__(self, folder: Path, set_id):
        super().__init__()

        self.src = []          # 原始 stroke sequence
        self.pre_stroke = []   # 前一拍
        self.next_stroke = []  # 後一拍
        self.q_len = []        # 每個 rally 的長度

        csv_path = folder / f'set{set_id}.csv'

        if not csv_path.exists():
            print(f"⚠️ Warning: File {csv_path} does not exist, skipping.")
            return

        df = pd.read_csv(str(csv_path))
        df_train = get_stroke_types(df, class_ls)  # 預設你有這兩個 function/變數

        rally_list = df_train['rally'].unique()

        for rally in rally_list:
            seq = df_train[df_train['rally'] == rally]['type_encode'].tolist()

            if len(seq) < 1:
                continue

            pre = [PADDING_TOKEN_ID] + seq[:-1]
            nxt = seq[1:] + [PADDING_TOKEN_ID]

            self.src.append(torch.tensor(seq, dtype=torch.long))
            self.pre_stroke.append(torch.tensor(pre, dtype=torch.long))
            self.next_stroke.append(torch.tensor(nxt, dtype=torch.long))
            self.q_len.append(len(seq))  # 原始長度

        print("src:", self.src)
        print("pre_stroke:", self.pre_stroke)
        print("next_stroke:", self.next_stroke)
        print("q_len:", self.q_len)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.pre_stroke[i], self.next_stroke[i], self.q_len[i]

# 假設這些類別的定義
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hd_dim, drop_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hd_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hd_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class TCN(nn.Module):
    def __init__(self, in_channels, num_channels, kernel_size, dropout=0.3):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1)*dilation_size, dropout=dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()



class Attention(nn.Module):
    def __init__(self, d_model, d_head, n_head, drop_p):
        super().__init__()
        self.d_head = d_head
        self.n_head = n_head
        self.scale = d_head**-0.5
        # 將 to_qkv 分開為 to_q, to_k, to_v 以處理不同的輸入
        self.to_q = nn.Linear(d_model, d_head * n_head, bias=False)
        self.to_k = nn.Linear(d_model, d_head * n_head, bias=False)
        self.to_v = nn.Linear(d_model, d_head * n_head, bias=False)
        
        self.to_out = nn.Linear(d_head * n_head, d_model)
        self.dropout = nn.Dropout(drop_p)

    # 這裡修改 forward 方法，接受 q, k, v，並提供預設值為 None
    def forward(self, x, k=None, v=None, mask=None):
        h = self.n_head
        
        # 如果 k 或 v 未提供，則使用 x 作為 k 和 v (Self-Attention)
        _k = x if k is None else k
        _v = x if v is None else v

        q = self.to_q(x).view(x.shape[0], x.shape[1], h, self.d_head).transpose(1, 2)
        k = self.to_k(_k).view(_k.shape[0], _k.shape[1], h, self.d_head).transpose(1, 2)
        v = self.to_v(_v).view(_v.shape[0], _v.shape[1], h, self.d_head).transpose(1, 2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            # 確保 mask 的維度與 dots 相匹配
            # dots shape: (batch_size, n_head, query_seq_len, key_seq_len)
            # mask shape: (batch_size, key_seq_len) from CrossTransformerLayer (mask[:, 1:])
            # mask 需要擴展為 (batch_size, 1, 1, key_seq_len)
            mask = mask.unsqueeze(1).unsqueeze(1) 
            dots = dots.masked_fill(~mask, float('-inf'))
            
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(out)
    
class FeedForward(nn.Module):
    '''Same as FeedForward in TemPose.'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)
        self.dropout = nn.Dropout(drop_p, inplace=True)

    def forward(self, x: Tensor):
        x = self.mlp(x)
        x = self.dropout(x)
        return x

class ConvFFN(nn.Module):
    '''ConvFFN替代原MLP_Block'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim , hd_dim, kernel_size=1)  # point-wise Conv
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_p, inplace=True)
        self.conv2 = nn.Conv2d(hd_dim, out_dim , kernel_size=1)  # point-wise Conv


    def forward(self, x: Tensor):
        x = x.unsqueeze(0)
        x = x.permute(0,3,2, 1)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.permute(0,3,2, 1)
        x = x.squeeze(0)
        return x

class ConvFFN_Head(nn.Module):
    '''ConvFFN_Head替代原MLP_Head'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.conv_ffn = ConvFFN(in_dim, out_dim, hd_dim, drop_p)

    def forward(self, x: Tensor):
        x = self.layer_norm(x)
        x = self.conv_ffn(x)
        return x

class ConvFFN_FeedForward(nn.Module):
    '''ConvFFN替代原FeedForward'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.conv_ffn = ConvFFN(in_dim, out_dim, hd_dim, drop_p)
        self.dropout = nn.Dropout(drop_p, inplace=True)

    def forward(self, x: Tensor):
        x = self.conv_ffn(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    '''Same as Attention in TemPose.'''
    def __init__(self, d_model, d_head, n_head, drop_p) -> None:
        super().__init__()
        d_cat = d_head * n_head

        self.h = n_head
        self.to_qkv = nn.Linear(d_model, d_cat * 3, bias=False)
        self.scale = d_head**-0.5

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(drop_p)  # This shouldn't be inplace.
        )
        
        self.tail = nn.Sequential(
            nn.Linear(d_cat, d_model),
            nn.Dropout(drop_p, inplace=True)
        ) if n_head != 1 or d_cat != d_model else nn.Identity()

    def forward(self, x: Tensor, mask: Tensor = None):
        # x: (b*n, t, d_model)
        qkv: Tensor = self.to_qkv(x)
        bn, t, _ = qkv.shape
        qkv = qkv.view(bn, t, self.h, 3, -1).transpose(1, 2)

        q = qkv[:, :, :, 0, :].contiguous()
        k_T = qkv[:, :, :, 1, :].transpose(-2, -1).contiguous()
        v = qkv[:, :, :, 2, :].contiguous()
        # q, k, v: (bn, h, t, d_head)

        dots: Tensor = (q @ k_T) * self.scale
        # dots: (bn, h, t, t)
        if mask is not None:
            # mask: (bn, t)
            mask = mask.view(bn, 1, 1, t)
            dots = dots.masked_fill(mask == 0.0, -torch.inf)
        
        coef = self.attend(dots)
        attension: Tensor = coef @ v
        # attension: (bn, h, t, d_head)
        
        out = attension.transpose(1, 2).reshape(bn, t, -1)
        # out: (bn, t, h*d_head)
        out = self.tail(out)
        return out  # (bn, t, d_model)


class DWConvAttention(nn.Module):
    '''用深度分離卷積替換原有的注意力機制'''
    def __init__(self, d_model, d_head, n_head, drop_p) -> None:
        super().__init__()
        self.h = n_head
        self.d_head = d_head
        self.scale = d_head ** -0.5
        
        # 第一層 Depthwise 卷積：處理空間維度
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False)
        
        # 第二層 Pointwise 卷積：融合不同通道的信息
        self.pointwise_conv = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        
        # 這裡還可以選擇添加 Dropout 和其他層
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x: Tensor, mask: Tensor = None):
        # x: (b*n, t, d_model)
        # bn, t, d_model = x.shape
        
        # Permute 輸入以適應卷積操作
        x = x.transpose(1, 2)  # (b*n, d_model, t)

        # Depthwise 卷積處理每個通道
        x = self.depthwise_conv(x)  # (b*n, d_model, t)
        
        # Pointwise 卷積融合不同通道的特徵
        x = self.pointwise_conv(x)  # (b*n, d_model, t)

        # 如果需要，加上 Dropout
        x = self.dropout(x)

        # 恢復維度
        x = x.transpose(1, 2)  # (b*n, t, d_model)
        
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, d_head, n_head, drop_p)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)

    def forward(self, x: Tensor, mask=None):
        z = self.layer_norm1(x)
        x = self.attn(z, mask) + x
        z = self.layer_norm2(x)
        x = self.ff(z) + x
        return x


class DWTransformerLayer(nn.Module):
    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p) -> None:
        super().__init__()
        # self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attn =MultiHeadAttention(d_model, d_head, n_head, drop_p)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = ConvFFN_FeedForward(d_model, d_model, hd_mlp, drop_p)

    def forward(self, x: Tensor, mask=None):
        z = self.layer_norm1(x)
        x = self.attn(z, mask) + x
        z = self.layer_norm2(x)
        x = self.ff(z) + x
        return x



class TransformerEncoder(nn.Module):
    '''Same as Transformer in TemPose.'''
    def __init__(self, d_model, d_head, n_head, depth, hd_mlp, drop_p) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, d_head, n_head, hd_mlp, drop_p)
             for _ in range(depth)]
        )

    def forward(self, x: Tensor, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DWTransformerEncoder(nn.Module):
    '''將 TransformerEncoder 中的 Attention 替換為 DWConvAttention'''
    def __init__(self, d_model, d_head, n_head, depth, hd_mlp, drop_p) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DWTransformerLayer(d_model, d_head, n_head, hd_mlp, drop_p)
             for _ in range(depth)]
        )

    def forward(self, x: Tensor, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class CrossTransformerLayer(nn.Module):
    def __init__(self, d_model, d_head, n_head, mlp_d_scale, drop_p):
        super().__init__()
        self.attn = PreNorm(d_model, Attention(d_model, d_head, n_head, drop_p))
        self.ff = PreNorm(d_model, FeedForward(d_model, mlp_d_scale, drop_p))

    def forward(self, x, context, mask=None):
        # x is query, context is key and value
        attended_x = self.attn(x, k=context, v=context, mask=mask) + x
        return self.ff(attended_x) + attended_x

class MLP_Head(nn.Module):
    def __init__(self, in_dim, out_dim, hd_dim, drop_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hd_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hd_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # pe shape: (1, max_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RNNPredictor(nn.Module):
    def __init__(self, type_size, emb_size, hid_size, n_layers=2, is_reverse=False ):
        super().__init__()

        #self.features  = None # 用來暫存 hook 收集到的特徵 (Tensor)
        #self.feature_ch = sum([self.emb_size, self.hid_size])  # 代表融合後總共多少特徵通道 Layer1(128) + Layer2(256)
        #self.target_size = target_size #控制最後輸出的特徵圖大小。
        self.is_reverse = is_reverse
        self.proj = nn.Linear(emb_size, hid_size) 

        self.embedding = nn.Embedding(
            num_embeddings = type_size, #有幾個類別要嵌入
            embedding_dim = emb_size, #每個詞向量(嵌入向量)的維度
            padding_idx = 31
        )

        self.rnn = nn.RNN(
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

        self.load_pretrained_weights() 
        for param in self.parameters():
            param.requires_grad = False
    
    def load_pretrained_weights(self):
        """
        根據模型的前向/後向屬性載入對應的權重。
        """
        if self.is_reverse:
            # 載入後向模型的權重
            try:
                self.load_state_dict(torch.load("weight/rnnpredictor_backward.pt", weights_only=True))
                print("成功載入後向 RNN 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到後向 RNN 模型權重檔案。")
        else:
            # 載入前向模型的權重
            try:
                self.load_state_dict(torch.load("weight/rnnpredictor.pt", weights_only=True))
                print("成功載入前向 RNN 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到前向 RNN 模型權重檔案。")

        
    def forward(self, x : Tensor, ballrounds: Tensor,q_len = None, hx = None): #x代表我一個batch的資料
        # idx : Tensor of shape (N,)  每筆樣本對應的時間步索引 (取 LSTM 的特徵)
        if self.is_reverse:
            x = torch.flip(x, dims=[1])  # 反轉序列

        pad = torch.full((x.size(0), 1), 31, device=x.device, dtype=x.dtype)
        x = torch.cat([pad, x[:, :-1]], dim=1)  # 在序列前面添加 padding
      
        N, L = x.shape #N為batch L為sequence length
        
        #每個詞變成一個 d 維的向量
        x = self.embedding(x) #將feature_dim變成embedding要的維度d x = (N, L, d)
     

        #x: 所有時間步的隱藏狀態輸出（每個時間點都輸出一次）->(N,L,h) hx:最後時間步的隱藏狀態（可用於下次繼續 RNN）
        rnn_output, _ = self.rnn(x, hx) #這邊會把嵌入過的序列丟進 RNN 或 LSTM 處理 # x: (N, L, h) not contiguous here
        rnn_output : Tensor
        print('rnn output',rnn_output.shape)
        #攤平成一維以便用於全連接層 ，-1 是 PyTorch 中 reshape 的自動推論功能，表示「這個維度讓我自己算」。
        x = rnn_output.reshape(N * L, -1)
        print('x',x.shape)
        x = self.fc(x)
        print('x1',x.shape)
        x = x.view(N, L, -1) #將維度還原，x: (N, L, n_tokens)
        print('x2',x.shape)
        # 確保 q_len 是 long 類型
        # q_len_tensor = q_len.long() #真正的sequence長度
        # # 從 lstm_output 中擷取每個 sample 對應 ballround 的特徵
        d = rnn_output.size(2) #隱藏層的維度 d
        # ballrounds = ballrounds.view(N, 1, 1).expand(N, 1, d) #(N, 1, d)

        
        
        # feature = torch.gather(lstm_output, dim=1, index=index)  # shape: (N, 1, d)

        with torch.no_grad():
            #先計算出正確的 0-based 索引 (N,)
            ballrounds_1_based = ballrounds.long() # 確保是 long 類型

            if not self.is_reverse:
                # 正向序列：索引是 ballrounds - 1
                # 例如 ballrounds=[5, 33] -> index_base=[4, 32] (N,)
                index_base = ballrounds_1_based - 1 
            else:
                # 反向序列：索引是 L - ballrounds
                # 例如 L=50, ballrounds=[5, 33] -> index_base=[45, 17] (N,))

                index_base = L - ballrounds_1_based  
            

            #然後將這個 (N,) 形狀的索引張量轉換為 torch.gather 所需的 (N, 1, d) 形狀
            # unsqueeze(1) -> (N, 1)
            # unsqueeze(2) -> (N, 1, 1)
            # expand(N, 1, d) -> (N, 1, d)
          
            index_for_gather = index_base.unsqueeze(2).expand(N, 1, d)


            # index_for_gather = index_base.unsqueeze(1).expand(N, L, 1)  # 改成 N, L, 1 形狀
            # index_for_gather = index_for_gather.expand(N, L, d)  # 擴展到 N, L, d
            feature = torch.gather(rnn_output, dim=1, index=index_for_gather)  # shape: (N, 1, d)
            print('feature',feature.shape)
            # # Eval模式才用hook或其他中間特徵邏輯
            # if not self.training:
            #     output = self.features
            #     self.features = None #清空 
            #     return x, output

       
        return _, feature

class RNNPredictor_1(nn.Module):
    def __init__(self, type_size, emb_size, hid_size, n_layers=2, is_reverse=False ):
        super().__init__()

        #self.features  = None # 用來暫存 hook 收集到的特徵 (Tensor)
        #self.feature_ch = sum([self.emb_size, self.hid_size])  # 代表融合後總共多少特徵通道 Layer1(128) + Layer2(256)
        #self.target_size = target_size #控制最後輸出的特徵圖大小。
        self.is_reverse = is_reverse

        self.embedding = nn.Embedding(
            num_embeddings = type_size, #有幾個類別要嵌入
            embedding_dim = emb_size, #每個詞向量(嵌入向量)的維度
            padding_idx = 31
        )

        self.rnn = nn.RNN(
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

        self.load_pretrained_weights() 
        for param in self.parameters():
            param.requires_grad = False
    
    def load_pretrained_weights(self):
        """
        根據模型的前向/後向屬性載入對應的權重。
        """
        if self.is_reverse:
            # 載入後向模型的權重
            try:
                self.load_state_dict(torch.load("weight/rnnpredictor_backward.pt", weights_only=True))
                print("成功載入後向 RNN 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到後向 RNN 模型權重檔案。")
        else:
            # 載入前向模型的權重
            try:
                self.load_state_dict(torch.load("weight/rnnpredictor.pt", weights_only=True))
                print("成功載入前向 RNN 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到前向 RNN 模型權重檔案。")

        
    def forward(self, x : Tensor, ballrounds: Tensor,q_len = None, hx = None): #x代表我一個batch的資料
        # idx : Tensor of shape (N,)  每筆樣本對應的時間步索引 (取 LSTM 的特徵)
        if self.is_reverse:
            x = torch.flip(x, dims=[1])  # 反轉序列

        pad = torch.full((x.size(0), 2), 31, device=x.device, dtype=x.dtype)
        x = torch.cat([pad, x[:, :-2]], dim=1)  # 在序列前面添加 padding
      
        N, L = x.shape #N為batch L為sequence length
        
        #每個詞變成一個 d 維的向量
        x = self.embedding(x) #將feature_dim變成embedding要的維度d x = (N, L, d)
     

        #x: 所有時間步的隱藏狀態輸出（每個時間點都輸出一次）->(N,L,h) hx:最後時間步的隱藏狀態（可用於下次繼續 RNN）
        rnn_output, _ = self.rnn(x, hx) #這邊會把嵌入過的序列丟進 RNN 或 LSTM 處理 # x: (N, L, h) not contiguous here
        rnn_output : Tensor

        #攤平成一維以便用於全連接層 ，-1 是 PyTorch 中 reshape 的自動推論功能，表示「這個維度讓我自己算」。
        x = rnn_output.reshape(N * L, -1)
        x = self.fc(x)
        x = x.view(N, L, -1) #將維度還原，x: (N, L, n_tokens)

        # 確保 q_len 是 long 類型
        # q_len_tensor = q_len.long() #真正的sequence長度
        # # 從 lstm_output 中擷取每個 sample 對應 ballround 的特徵
        d = rnn_output.size(2) #隱藏層的維度 d
        # ballrounds = ballrounds.view(N, 1, 1).expand(N, 1, d) #(N, 1, d)

        
        
        # feature = torch.gather(lstm_output, dim=1, index=index)  # shape: (N, 1, d)

        with torch.no_grad():
            #先計算出正確的 0-based 索引 (N,)
            ballrounds_1_based = ballrounds.long() # 確保是 long 類型

            if not self.is_reverse:
                # 正向序列：索引是 ballrounds - 1
                # 例如 ballrounds=[5, 33] -> index_base=[4, 32] (N,)
                index_base = ballrounds_1_based - 1 
            else:
                # 反向序列：索引是 L - ballrounds
                # 例如 L=50, ballrounds=[5, 33] -> index_base=[45, 17] (N,))

                index_base = L - ballrounds_1_based  
            

            #然後將這個 (N,) 形狀的索引張量轉換為 torch.gather 所需的 (N, 1, d) 形狀
            # unsqueeze(1) -> (N, 1)
            # unsqueeze(2) -> (N, 1, 1)
            # expand(N, 1, d) -> (N, 1, d)
          
            index_for_gather = index_base.unsqueeze(2).expand(N, 1, d)


            # index_for_gather = index_base.unsqueeze(1).expand(N, L, 1)  # 改成 N, L, 1 形狀
            # index_for_gather = index_for_gather.expand(N, L, d)  # 擴展到 N, L, d
            feature = torch.gather(rnn_output, dim=1, index=index_for_gather)  # shape: (N, 1, d)

            # # Eval模式才用hook或其他中間特徵邏輯
            # if not self.training:
            #     output = self.features
            #     self.features = None #清空 
            #     return x, output

       
        return _, feature


# 加上LSTM資訊
class Tempose_TFwithRNNPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)

        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output

# 加上LSTM資訊
class Tempose_TFwithtwostageRNNPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=32, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
        self.tempose = TemPose_TF_for_TwoStage(
                    in_dim=in_dim,
                    n_class=n_class,
                    seq_len=seq_len,
                    depth_inter=2
                )
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        # self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        if rally is None:
            return self.tempose(JnB, pos,shuttle, video_len)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        

        

        with torch.no_grad():
                _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
                _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        # pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output

# 加上LSTM資訊
class Tempose_GFwithRNNPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=hid_size , kernel_size=3, stride=1, padding=1)
        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(hid_size, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model] 
        self.tcn1_top = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_bottom = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_shuttle = TCN(2, tcn_channels, 3, drop_p)
        self.tcn2_top = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_bottom = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_shuttle = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn3_top = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_bottom = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_shuttle = TCN(d_model, tcn_channels, 7, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        JnB_b, JnB_n, JnB_t, JnB_d = JnB.shape
        JnB = JnB.view(JnB_b, JnB_n* JnB_t, JnB_d)  # 去掉 batch_size 維度，變成 [2, 30, 100]

        JnB = F.relu(self.conv1(JnB.permute(0, 2, 1)))
       
        JnB = JnB.permute(0, 2, 1).view(JnB_b,JnB_n,JnB_t,-1)
       
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn1_top(pos_top)
        pos_bottom = self.tcn1_bottom(pos_bottom)
        shuttle = self.tcn1_shuttle(shuttle)
      
        pos_top = self.tcn2_top(pos_top)
        pos_bottom = self.tcn2_bottom(pos_bottom)
        shuttle = self.tcn2_shuttle(shuttle)
   
        pos_top = self.tcn3_top(pos_top)
        pos_bottom = self.tcn3_bottom(pos_bottom)
        shuttle = self.tcn3_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output  


class SelfAttention(nn.Module):
    '''Same as Attention in TemPose.'''
    def __init__(self, d_model, d_head, h_head, drop_rnn) -> None:
        super().__init__()
        d_cat = d_head * h_head

        self.h = h_head
        self.to_qkv = nn.Linear(d_model, d_cat * 3, bias=False)
        self.scale = d_head**-0.5

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(drop_rnn)  # This shouldn't be inplace.
        )
        
        self.tail = nn.Sequential(
            nn.Linear(d_cat, d_model),
            nn.Dropout(drop_rnn, inplace=True)
        ) if h_head != 1 or d_cat != d_model else nn.Identity()

    def forward(self, x: Tensor, mask: Tensor = None):
        # x: (b*n, t, d_model)
        qkv: Tensor = self.to_qkv(x)
        bn, t, _ = qkv.shape
        qkv = qkv.view(bn, t, self.h, 3, -1).transpose(1, 2)

        q = qkv[:, :, :, 0, :].contiguous()
        k_T = qkv[:, :, :, 1, :].transpose(-2, -1).contiguous()
        v = qkv[:, :, :, 2, :].contiguous()
        # q, k, v: (bn, h, t, d_head)

        dots: Tensor = (q @ k_T) * self.scale
        # dots: (bn, h, t, t)
        if mask is not None:
            # mask: (bn, t)
            mask = mask.view(bn, 1, 1, t)
            dots = dots.masked_fill(mask == 0.0, -torch.inf)
        
        coef = self.attend(dots)
        attension: Tensor = coef @ v

        # attension: (bn, h, t, d_head)
        
        out = attension.transpose(1, 2).reshape(bn, t, -1)
        # out: (bn, t, h*d_head)
        out = self.tail(out)
        return out  # (bn, t, d_model)

class RNNPredictorSA(nn.Module):
    def __init__(self, type_size, emb_size, hid_size, n_layers=2, is_reverse=False, d_head=128, h_head=6, drop_rnn=0.1):
        super().__init__()

        self.is_reverse = is_reverse

        # 嵌入層
        self.embedding = nn.Embedding(
            num_embeddings = type_size,
            embedding_dim = emb_size,
            padding_idx = 31
        )

        # LSTM層
        self.rnn = nn.RNN(
            input_size = emb_size,
            hidden_size = hid_size,
            num_layers = n_layers,
            batch_first = True
        )

        # Attention層
        self.attention = SelfAttention(d_model=hid_size, d_head=d_head, h_head=h_head, drop_rnn=drop_rnn)

        # 全連接層
        self.fc = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(inplace=True),
            nn.Linear(hid_size, type_size)
        )

        self.load_pretrained_weights()
        # for param in self.parameters():
        #     param.requires_grad = False 
    
    def load_pretrained_weights(self):
        """
        根據模型的前向/後向屬性載入對應的權重。
        """
        if self.is_reverse:
            # 載入後向模型的權重
            try:
                self.load_state_dict(torch.load("weight/rnnpredictorsa_backward.pt", weights_only=True))
                print("成功載入後向 RNN 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到後向 RNN 模型權重檔案。")
        else:
            # 載入前向模型的權重
            try:
                self.load_state_dict(torch.load("weight/rnnpredictorsa.pt", weights_only=True))
                print("成功載入前向 RNN 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到前向 RNN 模型權重檔案。")
    
    def forward(self, x : Tensor, ballrounds: Tensor,q_len = None, hx = None): #x代表我一個batch的資料

        if self.is_reverse:
            x = torch.flip(x, dims=[1])  # 反轉序列

        pad = torch.full((x.size(0), 1), 31, device=x.device, dtype=x.dtype)
        x = torch.cat([pad, x[:, :-1]], dim=1)  # 在序列前面添加 padding
        # idx : Tensor of shape (N,)  每筆樣本對應的時間步索引 (取 LSTM 的特徵)
        N, L = x.shape #N為batch L為sequence length
        

        #每個詞變成一個 d 維的向量
        x = self.embedding(x) #將feature_dim變成embedding要的維度d x = (N, L, d)
     

        #x: 所有時間步的隱藏狀態輸出（每個時間點都輸出一次）->(N,L,h) hx:最後時間步的隱藏狀態（可用於下次繼續 RNN）
        rnn_output, _ = self.rnn(x, hx) #這邊會把嵌入過的序列丟進 RNN 或 LSTM 處理 # x: (N, L, h) not contiguous here
        rnn_output : Tensor

        # self-attention處理RNN的輸出
        attention_output = self.attention(rnn_output, mask=None)

        # 結果經過全連接層
        x = attention_output.view(N * L, -1)  # 展平
        x = self.fc(x)
        x = x.view(N, L, -1)  # 還原為(N, L, type_size)

        # #攤平成一維以便用於全連接層 ，-1 是 PyTorch 中 reshape 的自動推論功能，表示「這個維度讓我自己算」。
        # x = lstm_output.reshape(N * L, -1)
        # x = self.fc(x)
        # x = x.view(N, L, -1) #將維度還原，x: (N, L, n_tokens)

       
        # # 從 lstm_output 中擷取每個 sample 對應 ballround 的特徵
        d = rnn_output.size(2) #隱藏層的維度 d
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
            #先計算出正確的 0-based 索引 (N,)
            ballrounds_1_based = ballrounds.long() # 確保是 long 類型

            if not self.is_reverse:
                # 正向序列：索引是 ballrounds - 1
                # 例如 ballrounds=[5, 33] -> index_base=[4, 32] (N,)
                index_base = ballrounds_1_based - 1 
            else:
                # 反向序列：索引是 L - ballrounds
                # 例如 L=50, ballrounds=[5, 33] -> index_base=[45, 17] (N,))

                index_base = L - ballrounds_1_based  

            #然後將這個 (N,) 形狀的索引張量轉換為 torch.gather 所需的 (N, 1, d) 形狀
            # unsqueeze(1) -> (N, 1)
            # unsqueeze(2) -> (N, 1, 1)
            # expand(N, 1, d) -> (N, 1, d)
          
            index_for_gather = index_base.unsqueeze(2).expand(N, 1, d)
            feature = torch.gather(attention_output, dim=1, index=index_for_gather)  # shape: (N, 1, d)


            # # Eval模式才用hook或其他中間特徵邏輯
            # if not self.training:
            #     output = self.features
            #     self.features = None #清空 
            #     return x, output

       
        return x, feature
    

    
# 加上LSTM資訊 加上多頭自注意力機制 shuttle 跟 position加了attention 20.7G
class Tempose_TFwithRNNPredictorSA(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, h_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        self.attention = SelfAttention(d_model=d_model, d_head=d_head, h_head=h_head, drop_rnn=drop_rnn)
        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        # self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
        self.rnn_forward_feature = RNNPredictorSA(
                type_size, 
                emb_size, 
                hid_size, 
                n_layers=2, 
                is_reverse=False,
                d_head=128, 
                h_head=6, 
                drop_rnn=0.1
            )
        self.rnn_backward_feature = RNNPredictorSA(
                type_size, 
                emb_size, 
                hid_size, 
                n_layers=2, 
                is_reverse=True,
                d_head=128, 
                h_head=6, 
                drop_rnn=0.1
            )

        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
        # x =  x.view(-1, t, d)
        # x = self.attention(x, mask= None)

        # x = x.view(-1, 2, t, d)
       


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()
        
        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        
        # shuttle = self.attention(shuttle, mask = None)
        # pos_top = self.attention(pos_top, mask = None)
        # pos_bottom = self.attention(pos_bottom, mask = None)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
      
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)

        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        # alpha = torch.sigmoid(nn.Parameter(torch.ones(1))).to(x.device)
        # rnn_forward_feature = rnn_forward_feature * alpha * 2
        # rnn_backward_feature = rnn_backward_feature * (1 - alpha) * 2
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output   


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
            padding_idx = 31
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

        self.load_pretrained_weights() 
        # for param in self.parameters():
        #     param.requires_grad = False
    
    def load_pretrained_weights(self):
        """
        根據模型的前向/後向屬性載入對應的權重。
        """
        if self.is_reverse:
            # 載入後向模型的權重
            try:
                self.load_state_dict(torch.load("weight/lstmpredictor_backward.pt", weights_only=True))
                print("成功載入後向 LSTM 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到後向 LSTM 模型權重檔案。")
        else:
            # 載入前向模型的權重
            try:
                self.load_state_dict(torch.load("weight/lstmpredictor.pt", weights_only=True))
                print("成功載入前向 LSTM 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到前向 LSTM 模型權重檔案。")

        
    def forward(self, x : Tensor, ballrounds: Tensor,q_len = None, hx = None): #x代表我一個batch的資料
        # idx : Tensor of shape (N,)  每筆樣本對應的時間步索引 (取 LSTM 的特徵)
        # print("LSTMPredictor x shape:", x.tolist())
        # print("LSTMPredictor x shape:", x.shape)
        # print("LSTMPredictor ballrounds shape:", ballrounds.tolist())
        # print("LSTMPredictor ballrounds shape:", ballrounds.shape)

        if self.is_reverse:
            x = torch.flip(x, dims=[1])  # 反轉序列

        pad = torch.full((x.size(0), 1), 31, device=x.device, dtype=x.dtype)
       
        x = torch.cat([pad, x[:, :-1]], dim=1)  # 在序列前面添加 padding
       
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
        # q_len_tensor = q_len.long() #真正的sequence長度
        # # 從 lstm_output 中擷取每個 sample 對應 ballround 的特徵
        d = lstm_output.size(2) #隱藏層的維度 d
        # ballrounds = ballrounds.view(N, 1, 1).expand(N, 1, d) #(N, 1, d)

        
        
        # feature = torch.gather(lstm_output, dim=1, index=index)  # shape: (N, 1, d)

        with torch.no_grad():
            #先計算出正確的 0-based 索引 (N,)
            ballrounds_1_based = ballrounds.long() # 確保是 long 類型

            if not self.is_reverse:
                # 正向序列：索引是 ballrounds - 1
                # 例如 ballrounds=[5, 33] -> index_base=[4, 32] (N,)
                index_base = ballrounds_1_based - 1 
            else:
                # 反向序列：索引是 L - ballrounds
                # 例如 L=50, ballrounds=[5, 33] -> index_base=[45, 17] (N,))

                index_base = L - ballrounds_1_based  
            
            # print(f"🔍 rally.shape = {x.shape}")
            # print(f"🔍 ballrounds.shape = {ballrounds.shape}")
            # if ballrounds.shape[0] == 17:
            #     print(f"🔍 ballrounds = {ballrounds}")
            # print(f"🔍 index_base.shape = {index_base.shape}")
            # print(f"🔍 N = {N}, d = {d}")

            #然後將這個 (N,) 形狀的索引張量轉換為 torch.gather 所需的 (N, 1, d) 形狀
            # unsqueeze(1) -> (N, 1)
            # unsqueeze(2) -> (N, 1, 1)
            # expand(N, 1, d) -> (N, 1, d)
          
            index_for_gather = index_base.unsqueeze(2).expand(N, 1, d)


            # index_for_gather = index_base.unsqueeze(1).expand(N, L, 1)  # 改成 N, L, 1 形狀
            # index_for_gather = index_for_gather.expand(N, L, d)  # 擴展到 N, L, d
            feature = torch.gather(lstm_output, dim=1, index=index_for_gather)  # shape: (N, 1, d)

            # # Eval模式才用hook或其他中間特徵邏輯
            # if not self.training:
            #     output = self.features
            #     self.features = None #清空 
            #     return x, output

       
        return _, feature



# 加上LSTM資訊
class Tempose_TFwithLSTMPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output  
    
class Stroke_Embedding(nn.Module):
    def __init__(self, type_size, emb_size, hid_size):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=type_size,
            embedding_dim=emb_size,
            padding_idx=0
        )
        self.fusion = nn.Sequential(
            nn.Linear(emb_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size)
        )
        # 添加這一行，將 hid_size 儲存為實例屬性
        self.hid_size = hid_size 

    def forward(self, pre_stroke: Tensor):
        """
        輸入:
            pre_stroke: (N, 1)，每列是 [前一拍類別]，N 是所有序列的 stroke 總數
        輸出:
            x_final: (N, hid_size) (每拍一個特徵向量)
        """
        # # 拆成兩個 tensor
        # pre = pre_next_stroke[:, 0]  # (N,)
        # nxt = pre_next_stroke[:, 1]  # (N,)

        # 做 embedding： (N, D)
        pre_emb = self.embedding(pre_stroke)


        # 融合： (N, hid_size)
        x_final = self.fusion(pre_emb)
        return x_final

class Stroke_Embedding_RNN(nn.Module):
    def __init__(self, type_size, emb_size, hid_size):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=type_size,
            embedding_dim=emb_size,
            padding_idx=31
        )
        self.fusion = nn.Sequential(
            nn.Linear(emb_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size)
        )
        # 添加這一行，將 hid_size 儲存為實例屬性
        self.hid_size = hid_size 

    def forward(self, pre_stroke: Tensor):
        """
        輸入:
            pre_stroke: (N, 1)，每列是 [前一拍類別]，N 是所有序列的 stroke 總數
        輸出:
            x_final: (N, hid_size) (每拍一個特徵向量)
        """
        # # 拆成兩個 tensor
        # pre = pre_next_stroke[:, 0]  # (N,)
        # nxt = pre_next_stroke[:, 1]  # (N,)

        # 做 embedding： (N, D)
        pre_emb = self.embedding(pre_stroke)


        # 融合： (N, hid_size)
        x_final = self.fusion(pre_emb)
        return x_final
    
# 加上LSTM資訊
class Tempose_TFwithOneStrokeLSTMPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        # self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
        self.tempose = TemPose_TF_for_TwoStage(
                    in_dim=in_dim,
                    n_class=n_class,
                    seq_len=seq_len,
                    depth_inter=2
                )
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=n_class, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        if rally is None:
            return self.tempose(JnB, pos,shuttle, video_len)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        # print('pre stroke shape', pre_stroke.shape)
        # print('ballrounds shape', ballrounds.shape)
        # print('rally shape', rally.shape)
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            # _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)
        # print('rnn shape: ', rnn_forward_feature.shape)
        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))

        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # print('rnn shape 1: ', rnn_forward_feature.shape)
        pre_feature = pre_feature.view(b, -1) 
        # print('pre feature shape: ', pre_feature.shape)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        # rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
        # print('x main', x_main_features.shape)
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        # rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = torch.cat((rnn_forward_feature , pre_feature), dim=1) #(batch_size, hid_size * 3)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 3)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        # print('output',x_output.shape)
        return x_output  
    
# 加上LSTM資訊
class ConvTempose_TFwithOneStrokeLSTMPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(hid_size, d_model)

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=hid_size , kernel_size=3, stride=1, padding=1)


        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        JnB_b, JnB_n, JnB_t, JnB_d = JnB.shape
        JnB = JnB.view(JnB_b, JnB_n* JnB_t, JnB_d)  # 去掉 batch_size 維度，變成 [2, 30, 100]

        JnB = F.relu(self.conv1(JnB.permute(0, 2, 1)))
       
        JnB = JnB.permute(0, 2, 1).view(JnB_b,JnB_n,JnB_t,-1)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output  

# 加上LSTM資訊
class Tempose_TFwithStrokeFusionLSTMPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        # self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
        self.tempose = TemPose_TF_for_TwoStage(
                    in_dim=in_dim,
                    n_class=n_class,
                    seq_len=seq_len,
                    depth_inter=2
                )
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_fusion(type_size=n_class, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_next_stroke: Tensor

    ):
        if rally is None:
            return self.tempose(JnB, pos,shuttle, video_len)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        # print('pre stroke shape', pre_stroke.shape)
        # print('ballrounds shape', ballrounds.shape)
        # print('rally shape', rally.shape)
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            # _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)
        # print('rnn shape: ', rnn_forward_feature.shape)
        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))

        pre_next_stroke = self.stroke_emb(pre_next_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # print('rnn shape 1: ', rnn_forward_feature.shape)
        pre_next_stroke = pre_next_stroke.view(b, -1) 
        # print('pre feature shape: ', pre_feature.shape)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        # rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
        # print('x main', x_main_features.shape)
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        # rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = torch.cat((rnn_forward_feature , pre_next_stroke), dim=1) #(batch_size, hid_size * 3)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 3)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        # print('output',x_output.shape)
        return x_output 

class TemPose_TF_for_TwoStage(nn.Module):
    '''Similar to TemPose_TF in TemPose.'''
    def __init__(
        self, in_dim, seq_len, n_class=32, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.project = nn.Linear(in_dim, d_model)

        # TCNs
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+3, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        pos: Tensor,  # pos: (b, t, n, 2)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)
        # shuttle: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)
        shuttle: Tensor = self.tcn_shuttle(shuttle)
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)
        # shuttle: (b, d, t)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)

        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
        # x_additional: (b, 3, t, d)

        # Temporal Fusion (TF)
        x = torch.cat((x, x_additional), dim=1)
        n += 3

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        x_main_features = self.mlp_head(x_main_features)
        return x_main_features
    
class TemPose_TF_for_RNNTwoStage(nn.Module):
    '''Similar to TemPose_TF in TemPose.'''
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.project = nn.Linear(in_dim, d_model)

        # TCNs
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+3, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model, n_class, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model
        self.init_weights()
        self.load_pretrained_weights()


    def load_pretrained_weights(self):
        """
        根據模型的前向/後向屬性載入對應的權重。
        """
        
            # 載入後向模型的權重
        try:
            self.load_state_dict(torch.load("weight/tempose_TF_JnB_bone_rally_data_npy_collated_11.pt", weights_only=True))
            print("成功載入Tempose 模型權重。")
        except FileNotFoundError:
            print("錯誤：找不到Tempose 模型權重檔案。")

    @torch.no_grad()
    def init_weights(self):
        # Positional encodings are different from TemPose.
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        # Same as TemPose here.
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        # Same as TemPose
        if isinstance(m, nn.Linear):
            # following official JAX ViT xavier.uniform is used:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)
    

    def forward(
        self,
        JnB: Tensor,  # JnB: (b, t, n, input_dim)
        pos: Tensor,  # pos: (b, t, n, 2)
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)
        # shuttle: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)
        shuttle: Tensor = self.tcn_shuttle(shuttle)
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)
        # shuttle: (b, d, t)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)

        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
        # x_additional: (b, 3, t, d)

        # Temporal Fusion (TF)
        x = torch.cat((x, x_additional), dim=1)
        n += 3

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        x_main_features = self.mlp_head(x_main_features)
        return x_main_features

# 加上LSTM資訊
class Tempose_TFwithtwostageBiLSTMPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.3,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
        self.tempose = TemPose_TF_for_TwoStage(
                    in_dim=in_dim,
                    n_class=type_size,
                    seq_len=seq_len,
                    depth_inter=2
                )
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        if rally is None:
            return self.tempose(JnB, pos,shuttle, video_len)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        

        

        with torch.no_grad():
                _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
                _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        # pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output


# 加上LSTM資訊
class Tempose_TFwithtwostageLSTMPredictorSA(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=32, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # Attention層
        self.attention = SelfAttention(d_model=hid_size, d_head=d_head, h_head=n_head, drop_rnn=drop_rnn)
        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
        self.tempose = TemPose_TF_for_TwoStage(
                    in_dim=in_dim,
                    n_class=n_class,
                    seq_len=seq_len,
                    depth_inter=2
                )
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        if rally is None:
            return self.tempose(JnB, pos,shuttle, video_len)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        

        

        with torch.no_grad():
                _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
                _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        # pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = self.attention(rnn_forward_feature)
        rnn_backward_feature = self.attention(rnn_backward_feature)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        

        # pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output


# 加上LSTM資訊
class Tempose_TFwithtwostageSingleLSTMPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.3,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size , type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        # self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
        self.tempose = TemPose_TF_for_TwoStage(
                    in_dim=in_dim,
                    n_class=type_size,
                    seq_len=seq_len,
                    depth_inter=2
                )
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        if rally is None:
            return self.tempose(JnB, pos,shuttle, video_len)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        

        

        with torch.no_grad():
                _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
                # _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        # pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        # rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        # rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_forward_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output

# 加上LSTM資訊
class Tempose_SFwithOneStrokeLSTMPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]

        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 1, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 1, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
      
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data

        shuttle = self.tcn_shuttle(shuttle)

    
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = shuttle.unsqueeze(1)
        # x_additional: (b, 1, t, d)

        # Shuttlecock Fusion (SF)
        x = torch.cat((x, x_additional), dim=1)
        n += 1

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output

 # 加上LSTM資訊
class TCNTempose_TFwithOneStrokeLSTMPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn1_top = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_bottom = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_shuttle = TCN(2, tcn_channels, 3, drop_p)
        self.tcn2_top = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_bottom = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_shuttle = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn3_top = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_bottom = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_shuttle = TCN(d_model, tcn_channels, 7, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn1_top(pos_top)
        pos_bottom = self.tcn1_bottom(pos_bottom)
        shuttle = self.tcn1_shuttle(shuttle)
      
        pos_top = self.tcn2_top(pos_top)
        pos_bottom = self.tcn2_bottom(pos_bottom)
        shuttle = self.tcn2_shuttle(shuttle)
   
        pos_top = self.tcn3_top(pos_top)
        pos_bottom = self.tcn3_bottom(pos_bottom)
        shuttle = self.tcn3_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output    


 # 加上LSTM資訊
class Tempose_GFwithOneStrokeLSTMPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(hid_size, d_model)

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=hid_size , kernel_size=3, stride=1, padding=1)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn1_top = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_bottom = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_shuttle = TCN(2, tcn_channels, 3, drop_p)
        self.tcn2_top = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_bottom = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_shuttle = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn3_top = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_bottom = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_shuttle = TCN(d_model, tcn_channels, 7, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        JnB_b, JnB_n, JnB_t, JnB_d = JnB.shape
        JnB = JnB.view(JnB_b, JnB_n* JnB_t, JnB_d)  # 去掉 batch_size 維度，變成 [2, 30, 100]

        JnB = F.relu(self.conv1(JnB.permute(0, 2, 1)))
       
        JnB = JnB.permute(0, 2, 1).view(JnB_b,JnB_n,JnB_t,-1)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn1_top(pos_top)
        pos_bottom = self.tcn1_bottom(pos_bottom)
        shuttle = self.tcn1_shuttle(shuttle)
      
        pos_top = self.tcn2_top(pos_top)
        pos_bottom = self.tcn2_bottom(pos_bottom)
        shuttle = self.tcn2_shuttle(shuttle)
   
        pos_top = self.tcn3_top(pos_top)
        pos_bottom = self.tcn3_bottom(pos_bottom)
        shuttle = self.tcn3_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output    


# 加上LSTM資訊
class Tempose_TFwithOneStrokeGRUPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output  
 # 加上LSTM資訊
class ConvTempose_TFwithOneStrokeGRUPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(hid_size, d_model)

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=hid_size , kernel_size=3, stride=1, padding=1)


        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        JnB_b, JnB_n, JnB_t, JnB_d = JnB.shape
        JnB = JnB.view(JnB_b, JnB_n* JnB_t, JnB_d)  # 去掉 batch_size 維度，變成 [2, 30, 100]

        JnB = F.relu(self.conv1(JnB.permute(0, 2, 1)))
       
        JnB = JnB.permute(0, 2, 1).view(JnB_b,JnB_n,JnB_t,-1)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output  


# 加上LSTM資訊
class Tempose_SFwithOneStrokeGRUPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]

        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 1, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 1, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
      
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data

        shuttle = self.tcn_shuttle(shuttle)

    
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = shuttle.unsqueeze(1)
        # x_additional: (b, 1, t, d)

        # Shuttlecock Fusion (SF)
        x = torch.cat((x, x_additional), dim=1)
        n += 1

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output

  # 加上LSTM資訊
class Tempose_GFwithOneStrokeGRUPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(hid_size, d_model)

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=hid_size , kernel_size=3, stride=1, padding=1)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn1_top = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_bottom = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_shuttle = TCN(2, tcn_channels, 3, drop_p)
        self.tcn2_top = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_bottom = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_shuttle = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn3_top = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_bottom = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_shuttle = TCN(d_model, tcn_channels, 7, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        JnB_b, JnB_n, JnB_t, JnB_d = JnB.shape
        JnB = JnB.view(JnB_b, JnB_n* JnB_t, JnB_d)  # 去掉 batch_size 維度，變成 [2, 30, 100]

        JnB = F.relu(self.conv1(JnB.permute(0, 2, 1)))
       
        JnB = JnB.permute(0, 2, 1).view(JnB_b,JnB_n,JnB_t,-1)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn1_top(pos_top)
        pos_bottom = self.tcn1_bottom(pos_bottom)
        shuttle = self.tcn1_shuttle(shuttle)
      
        pos_top = self.tcn2_top(pos_top)
        pos_bottom = self.tcn2_bottom(pos_bottom)
        shuttle = self.tcn2_shuttle(shuttle)
   
        pos_top = self.tcn3_top(pos_top)
        pos_bottom = self.tcn3_bottom(pos_bottom)
        shuttle = self.tcn3_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output    
      

# 加上LSTM資訊
class TCNTempose_TFwithOneStrokeGRUPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn1_top = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_bottom = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_shuttle = TCN(2, tcn_channels, 3, drop_p)
        self.tcn2_top = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_bottom = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_shuttle = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn3_top = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_bottom = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_shuttle = TCN(d_model, tcn_channels, 7, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn1_top(pos_top)
        pos_bottom = self.tcn1_bottom(pos_bottom)
        shuttle = self.tcn1_shuttle(shuttle)
      
        pos_top = self.tcn2_top(pos_top)
        pos_bottom = self.tcn2_bottom(pos_bottom)
        shuttle = self.tcn2_shuttle(shuttle)
   
        pos_top = self.tcn3_top(pos_top)
        pos_bottom = self.tcn3_bottom(pos_bottom)
        shuttle = self.tcn3_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output    


# 加上LSTM資訊
class Tempose_TFwithOneStrokeRNNPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output 


# 加上LSTM資訊
class TCNTempose_TFwithOneStrokeRNNPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn1_top = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_bottom = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_shuttle = TCN(2, tcn_channels, 3, drop_p)
        self.tcn2_top = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_bottom = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_shuttle = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn3_top = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_bottom = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_shuttle = TCN(d_model, tcn_channels, 7, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn1_top(pos_top)
        pos_bottom = self.tcn1_bottom(pos_bottom)
        shuttle = self.tcn1_shuttle(shuttle)
      
        pos_top = self.tcn2_top(pos_top)
        pos_bottom = self.tcn2_bottom(pos_bottom)
        shuttle = self.tcn2_shuttle(shuttle)
   
        pos_top = self.tcn3_top(pos_top)
        pos_bottom = self.tcn3_bottom(pos_bottom)
        shuttle = self.tcn3_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output    


# 加上LSTM資訊
class ConvTempose_TFwithOneStrokeRNNPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(hid_size, d_model)

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=hid_size , kernel_size=3, stride=1, padding=1)


        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        JnB_b, JnB_n, JnB_t, JnB_d = JnB.shape
        JnB = JnB.view(JnB_b, JnB_n* JnB_t, JnB_d)  # 去掉 batch_size 維度，變成 [2, 30, 100]

        JnB = F.relu(self.conv1(JnB.permute(0, 2, 1)))
       
        JnB = JnB.permute(0, 2, 1).view(JnB_b,JnB_n,JnB_t,-1)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output  


# 加上LSTM資訊
class Tempose_SFwithOneStrokeRNNPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]

        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 1, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 1, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
      
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data

        shuttle = self.tcn_shuttle(shuttle)

    
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = shuttle.unsqueeze(1)
        # x_additional: (b, 1, t, d)

        # Shuttlecock Fusion (SF)
        x = torch.cat((x, x_additional), dim=1)
        n += 1

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output


  # 加上LSTM資訊
class Tempose_GFwithOneStrokeRNNPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(hid_size, d_model)

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=hid_size , kernel_size=3, stride=1, padding=1)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn1_top = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_bottom = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_shuttle = TCN(2, tcn_channels, 3, drop_p)
        self.tcn2_top = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_bottom = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_shuttle = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn3_top = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_bottom = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_shuttle = TCN(d_model, tcn_channels, 7, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        # self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_stroke: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        JnB_b, JnB_n, JnB_t, JnB_d = JnB.shape
        JnB = JnB.view(JnB_b, JnB_n* JnB_t, JnB_d)  # 去掉 batch_size 維度，變成 [2, 30, 100]

        JnB = F.relu(self.conv1(JnB.permute(0, 2, 1)))
       
        JnB = JnB.permute(0, 2, 1).view(JnB_b,JnB_n,JnB_t,-1)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            # _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn1_top(pos_top)
        pos_bottom = self.tcn1_bottom(pos_bottom)
        shuttle = self.tcn1_shuttle(shuttle)
      
        pos_top = self.tcn2_top(pos_top)
        pos_bottom = self.tcn2_bottom(pos_bottom)
        shuttle = self.tcn2_shuttle(shuttle)
   
        pos_top = self.tcn3_top(pos_top)
        pos_bottom = self.tcn3_bottom(pos_bottom)
        shuttle = self.tcn3_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        # rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_backward_feature, pre_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output        

# 加上LSTM資訊
class Tempose_GFwithLSTMPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=hid_size , kernel_size=3, stride=1, padding=1)

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(hid_size, d_model)

        # TCN layers for top, bottom, shuttle
        # tcn_channels = [d_model // 2, d_model]
        # self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        # self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        # self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        tcn_channels = [d_model // 2, d_model] 
        self.tcn1_top = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_bottom = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_shuttle = TCN(2, tcn_channels, 3, drop_p)
        self.tcn2_top = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_bottom = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_shuttle = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn3_top = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_bottom = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_shuttle = TCN(d_model, tcn_channels, 7, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        JnB_b, JnB_n, JnB_t, JnB_d = JnB.shape
        JnB = JnB.view(JnB_b, JnB_n* JnB_t, JnB_d)  # 去掉 batch_size 維度，變成 [2, 30, 100]

        JnB = F.relu(self.conv1(JnB.permute(0, 2, 1)))
       
        JnB = JnB.permute(0, 2, 1).view(JnB_b,JnB_n,JnB_t,-1)
       
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn1_top(pos_top)
        pos_bottom = self.tcn1_bottom(pos_bottom)
        shuttle = self.tcn1_shuttle(shuttle)
      
        pos_top = self.tcn2_top(pos_top)
        pos_bottom = self.tcn2_bottom(pos_bottom)
        shuttle = self.tcn2_shuttle(shuttle)
   
        pos_top = self.tcn3_top(pos_top)
        pos_bottom = self.tcn3_bottom(pos_bottom)
        shuttle = self.tcn3_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output  


class LSTMPredictorSA(nn.Module):
    def __init__(self, type_size, emb_size, hid_size, n_layers=2, is_reverse=False, d_head=128, h_head=6, drop_rnn=0.1):
        super().__init__()

        self.is_reverse = is_reverse

        # 嵌入層
        self.embedding = nn.Embedding(
            num_embeddings = type_size,
            embedding_dim = emb_size,
            padding_idx = 31
        )

        # LSTM層
        self.lstm = nn.LSTM(
            input_size = emb_size,
            hidden_size = hid_size,
            num_layers = n_layers,
            batch_first = True
        )

        # Attention層
        self.attention = SelfAttention(d_model=hid_size, d_head=d_head, h_head=h_head, drop_rnn=drop_rnn)

        # 全連接層
        self.fc = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(inplace=True),
            nn.Linear(hid_size, type_size)
        )

        self.load_pretrained_weights()
        # for param in self.parameters():
        #     param.requires_grad = False 
    
    def load_pretrained_weights(self):
        """
        根據模型的前向/後向屬性載入對應的權重。
        """
        if self.is_reverse:
            # 載入後向模型的權重
            try:
                self.load_state_dict(torch.load("weight/lstmpredictorsa_backward.pt", weights_only=True))
                print("成功載入後向 LSTM 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到後向 LSTM 模型權重檔案。")
        else:
            # 載入前向模型的權重
            try:
                self.load_state_dict(torch.load("weight/lstmpredictorsa.pt", weights_only=True))
                print("成功載入前向 LSTM 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到前向 LSTM 模型權重檔案。")

    def forward(self, x : Tensor, ballrounds: Tensor,q_len = None, hx = None): #x代表我一個batch的資料

        if self.is_reverse:
            x = torch.flip(x, dims=[1])  # 反轉序列

        pad = torch.full((x.size(0), 1), 31, device=x.device, dtype=x.dtype)
        x = torch.cat([pad, x[:, :-1]], dim=1)  # 在序列前面添加 padding
        # idx : Tensor of shape (N,)  每筆樣本對應的時間步索引 (取 LSTM 的特徵)
        N, L = x.shape #N為batch L為sequence length
        

        #每個詞變成一個 d 維的向量
        x = self.embedding(x) #將feature_dim變成embedding要的維度d x = (N, L, d)
     

        #x: 所有時間步的隱藏狀態輸出（每個時間點都輸出一次）->(N,L,h) hx:最後時間步的隱藏狀態（可用於下次繼續 RNN）
        lstm_output, _ = self.lstm(x, hx) #這邊會把嵌入過的序列丟進 RNN 或 LSTM 處理 # x: (N, L, h) not contiguous here
        lstm_output : Tensor

        # self-attention處理LSTM的輸出
        attention_output = self.attention(lstm_output, mask=None)

        # 結果經過全連接層
        x = attention_output.view(N * L, -1)  # 展平
        x = self.fc(x)
        x = x.view(N, L, -1)  # 還原為(N, L, type_size)

        # #攤平成一維以便用於全連接層 ，-1 是 PyTorch 中 reshape 的自動推論功能，表示「這個維度讓我自己算」。
        # x = lstm_output.reshape(N * L, -1)
        # x = self.fc(x)
        # x = x.view(N, L, -1) #將維度還原，x: (N, L, n_tokens)

       
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
            #先計算出正確的 0-based 索引 (N,)
            ballrounds_1_based = ballrounds.long() # 確保是 long 類型

            if not self.is_reverse:
                # 正向序列：索引是 ballrounds - 1
                # 例如 ballrounds=[5, 33] -> index_base=[4, 32] (N,)
                index_base = ballrounds_1_based - 1 
            else:
                # 反向序列：索引是 L - ballrounds
                # 例如 L=50, ballrounds=[5, 33] -> index_base=[45, 17] (N,))

                index_base = L - ballrounds_1_based  

            #然後將這個 (N,) 形狀的索引張量轉換為 torch.gather 所需的 (N, 1, d) 形狀
            # unsqueeze(1) -> (N, 1)
            # unsqueeze(2) -> (N, 1, 1)
            # expand(N, 1, d) -> (N, 1, d)
          
            index_for_gather = index_base.unsqueeze(2).expand(N, 1, d)
            feature = torch.gather(attention_output, dim=1, index=index_for_gather)  # shape: (N, 1, d)


            # # Eval模式才用hook或其他中間特徵邏輯
            # if not self.training:
            #     output = self.features
            #     self.features = None #清空 
            #     return x, output

       
        return x, feature
    

    
# 加上LSTM資訊 加上多頭自注意力機制 shuttle 跟 position加了attention 20.7G
class Tempose_TFwithLSTMPredictorSA(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6,h_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        self.attention = SelfAttention(d_model=d_model, d_head=d_head, h_head=h_head, drop_rnn=drop_rnn)
        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = LSTMPredictorSA(
                type_size, 
                emb_size, 
                hid_size, 
                n_layers=2, 
                is_reverse=False,
                d_head=128, 
                h_head=6, 
                drop_rnn=0.1
            )
        self.rnn_backward_feature = LSTMPredictorSA(
                type_size, 
                emb_size, 
                hid_size, 
                n_layers=2, 
                is_reverse=True,
                d_head=128, 
                h_head=6, 
                drop_rnn=0.1
            )

        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
        # x =  x.view(-1, t, d)
        # x = self.attention(x, mask= None)

        # x = x.view(-1, 2, t, d)
       


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()
        
        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        
        # shuttle = self.attention(shuttle, mask = None)
        # pos_top = self.attention(pos_top, mask = None)
        # pos_bottom = self.attention(pos_bottom, mask = None)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        # alpha = torch.sigmoid(nn.Parameter(torch.ones(1))).to(x.device)
        # rnn_forward_feature = rnn_forward_feature * alpha * 2
        # rnn_backward_feature = rnn_backward_feature * (1 - alpha) * 2
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output

class TemPose_VwithRNNPredictor(nn.Module):
    '''Similar to TemPose_V in TemPose.'''
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        ## Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        
        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        # MLP Head
        self.mlp_head = MLP_Head(d_model+ hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)

        self.d_model = d_model
       
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)
        

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)

        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()

        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output
    
class Stroke_fusion(nn.Module):
    def __init__(self, type_size, emb_size, hid_size):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=type_size,
            embedding_dim=emb_size,
            padding_idx=0
        )
        self.fusion = nn.Sequential(
            nn.Linear(emb_size * 2, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size)
        )
        # 添加這一行，將 hid_size 儲存為實例屬性
        self.hid_size = hid_size 

    def forward(self, pre_next_stroke: Tensor):
        """
        輸入:
            pre_next_stroke: (N, 2)，每列是 [前一拍類別, 後一拍類別]，N 是所有序列的 stroke 總數
        輸出:
            x_final: (N, hid_size)，融合前後拍的特徵 (每拍一個特徵向量)
        """
        # 拆成兩個 tensor
        pre = pre_next_stroke[:, 0]  # (N,)
        nxt = pre_next_stroke[:, 1]  # (N,)

        # 做 embedding： (N, D)
        pre_emb = self.embedding(pre)
        next_emb = self.embedding(nxt)

        # 合併： (N, 2D)
        feat_cat = torch.cat([pre_emb, next_emb], dim=1)

        # 融合： (N, hid_size)
        x_final = self.fusion(feat_cat)
        return x_final


#  加上LSTM資訊
class Tempose_TFwithStrokeFusionandRNNPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.3,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 3, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)

        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)

        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_next_stroke: Tensor
    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim

        pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        pre_next_feature = pre_next_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_next_feature = self.stroke_dropout(pre_next_feature)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        # rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        x_combined = torch.cat((x_combined, pre_next_feature), dim=1)  # (b, d_model + hid_size * 3)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output
    
class Tempose_SFwithRNNPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=32, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5, drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

       # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+1, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)


        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = RNNPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)

        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        shuttle = self.tcn_shuttle(shuttle)

        shuttle = shuttle.transpose(1, 2).contiguous()
        x_additional = shuttle.unsqueeze(1)
        # x_additional: (b, 1, t, d)

        # Shuttlecock Fusion (SF)
        x = torch.cat((x, x_additional), dim=1)
        n += 1

       # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        # rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output   


#  加上LSTM資訊
class Tempose_TFwithStrokeFusionandLSTMPredictorSA(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        self.attention = SelfAttention(d_model=d_model, d_head=d_head, n_head=n_head, drop_p=drop_p)
        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 3, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)

        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = LSTMPredictorSA(
                type_size, 
                emb_size, 
                hid_size, 
                n_layers=2, 
                is_reverse=False,
                d_head=128, 
                n_head=6, 
                drop_p=0.1
            )
        self.rnn_backward_feature = LSTMPredictorSA(
                type_size, 
                emb_size, 
                hid_size, 
                n_layers=2, 
                is_reverse=True,
                d_head=128, 
                n_head=6, 
                drop_p=0.1
            )
        
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor,
        pre_next_stroke: Tensor
    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim

        pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
        
        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        pre_next_feature = pre_next_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_next_feature = self.stroke_dropout(pre_next_feature)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        x_combined = torch.cat((x_combined, pre_next_feature), dim=1)  # (b, d_model + hid_size * 3)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output


class Tempose_SFwithLSTMPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=32, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5, drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

       # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+1, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)


        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = LSTMPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)

        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        shuttle = self.tcn_shuttle(shuttle)

        shuttle = shuttle.transpose(1, 2).contiguous()
        x_additional = shuttle.unsqueeze(1)
        # x_additional: (b, 1, t, d)

        # Shuttlecock Fusion (SF)
        x = torch.cat((x, x_additional), dim=1)
        n += 1

       # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        # rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output 

class GRUPredictor(nn.Module):
    def __init__(self, type_size, emb_size, hid_size, n_layers=2, is_reverse=False ):
        super().__init__()

        #self.features  = None # 用來暫存 hook 收集到的特徵 (Tensor)
        #self.feature_ch = sum([self.emb_size, self.hid_size])  # 代表融合後總共多少特徵通道 Layer1(128) + Layer2(256)
        #self.target_size = target_size #控制最後輸出的特徵圖大小。
        self.is_reverse = is_reverse

        self.embedding = nn.Embedding(
            num_embeddings = type_size, #有幾個類別要嵌入
            embedding_dim = emb_size, #每個詞向量(嵌入向量)的維度
            padding_idx = 31
        )

        self.gru = nn.GRU(
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

        self.load_pretrained_weights() 
        for param in self.parameters():
            param.requires_grad = False
    
    def load_pretrained_weights(self):
        """
        根據模型的前向/後向屬性載入對應的權重。
        """
        if self.is_reverse:
            # 載入後向模型的權重
            try:
                self.load_state_dict(torch.load("weight/grupredictor_backward.pt", weights_only=True))
                print("成功載入後向 GRU 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到後向 GRU 模型權重檔案。")
        else:
            # 載入前向模型的權重
            try:
                self.load_state_dict(torch.load("weight/grupredictor.pt", weights_only=True))
                print("成功載入前向 GRU 模型權重。")
            except FileNotFoundError:
                print("錯誤：找不到前向 GRU 模型權重檔案。")

        
    def forward(self, x : Tensor, ballrounds: Tensor,q_len = None, hx = None): #x代表我一個batch的資料
        # idx : Tensor of shape (N,)  每筆樣本對應的時間步索引 (取 LSTM 的特徵)
        if self.is_reverse:
            x = torch.flip(x, dims=[1])  # 反轉序列

        pad = torch.full((x.size(0), 1), 31, device=x.device, dtype=x.dtype)
        x = torch.cat([pad, x[:, :-1]], dim=1)  # 在序列前面添加 padding
      
        N, L = x.shape #N為batch L為sequence length
        
        #每個詞變成一個 d 維的向量
        x = self.embedding(x) #將feature_dim變成embedding要的維度d x = (N, L, d)
     

        #x: 所有時間步的隱藏狀態輸出（每個時間點都輸出一次）->(N,L,h) hx:最後時間步的隱藏狀態（可用於下次繼續 RNN）
        gru_output, _ = self.gru(x, hx) #這邊會把嵌入過的序列丟進 RNN 或 LSTM 處理 # x: (N, L, h) not contiguous here
        gru_output : Tensor

        #攤平成一維以便用於全連接層 ，-1 是 PyTorch 中 reshape 的自動推論功能，表示「這個維度讓我自己算」。
        x = gru_output.reshape(N * L, -1)
        x = self.fc(x)
        x = x.view(N, L, -1) #將維度還原，x: (N, L, n_tokens)

        # 確保 q_len 是 long 類型
        # q_len_tensor = q_len.long() #真正的sequence長度
        # # 從 lstm_output 中擷取每個 sample 對應 ballround 的特徵
        d = gru_output.size(2) #隱藏層的維度 d
        # ballrounds = ballrounds.view(N, 1, 1).expand(N, 1, d) #(N, 1, d)

        
        
        # feature = torch.gather(lstm_output, dim=1, index=index)  # shape: (N, 1, d)

        with torch.no_grad():
            #先計算出正確的 0-based 索引 (N,)
            ballrounds_1_based = ballrounds.long() # 確保是 long 類型

            if not self.is_reverse:
                # 正向序列：索引是 ballrounds - 1
                # 例如 ballrounds=[5, 33] -> index_base=[4, 32] (N,)
                index_base = ballrounds_1_based - 1 
            else:
                # 反向序列：索引是 L - ballrounds
                # 例如 L=50, ballrounds=[5, 33] -> index_base=[45, 17] (N,))

                index_base = L - ballrounds_1_based  
            

            #然後將這個 (N,) 形狀的索引張量轉換為 torch.gather 所需的 (N, 1, d) 形狀
            # unsqueeze(1) -> (N, 1)
            # unsqueeze(2) -> (N, 1, 1)
            # expand(N, 1, d) -> (N, 1, d)
          
            index_for_gather = index_base.unsqueeze(2).expand(N, 1, d)


            # index_for_gather = index_base.unsqueeze(1).expand(N, L, 1)  # 改成 N, L, 1 形狀
            # index_for_gather = index_for_gather.expand(N, L, d)  # 擴展到 N, L, d
            feature = torch.gather(gru_output, dim=1, index=index_for_gather)  # shape: (N, 1, d)

            # # Eval模式才用hook或其他中間特徵邏輯
            # if not self.training:
            #     output = self.features
            #     self.features = None #清空 
            #     return x, output

       
        return _, feature



# 加上LSTM資訊
class Tempose_TFwithGRUPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError


        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)


        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2 , type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        
       
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)
      
        

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output

# 加上LSTM資訊
class Tempose_TFwithtwostageGRUPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=32, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
        self.tempose = TemPose_TF_for_TwoStage(
                    in_dim=in_dim,
                    n_class=n_class,
                    seq_len=seq_len,
                    depth_inter=2
                )
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        # Stroke emb model
        # self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        if rally is None:
            return self.tempose(JnB, pos,shuttle, video_len)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        

        

        with torch.no_grad():
                _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
                _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        
        # pre_feature = self.stroke_emb(pre_stroke)

       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # pre_feature = pre_feature.view(b, -1) 
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output

# 加上LSTM資訊
class Tempose_GFwithGRUPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=hid_size , kernel_size=3, stride=1, padding=1)

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(hid_size, d_model)

        # TCN layers for top, bottom, shuttle
        # tcn_channels = [d_model // 2, d_model]
        # self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        # self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        # self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        tcn_channels = [d_model // 2, d_model] 
        self.tcn1_top = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_bottom = TCN(2, tcn_channels, 3, drop_p)
        self.tcn1_shuttle = TCN(2, tcn_channels, 3, drop_p)
        self.tcn2_top = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_bottom = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn2_shuttle = TCN(d_model, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn3_top = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_bottom = TCN(d_model, tcn_channels, 7, drop_p)
        self.tcn3_shuttle = TCN(d_model, tcn_channels, 7, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, type_size, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2 , type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)
       
        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        pos: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        JnB_b, JnB_n, JnB_t, JnB_d = JnB.shape
        JnB = JnB.view(JnB_b, JnB_n* JnB_t, JnB_d)  # 去掉 batch_size 維度，變成 [2, 30, 100]

        JnB = F.relu(self.conv1(JnB.permute(0, 2, 1)))
       
        JnB = JnB.permute(0, 2, 1).view(JnB_b,JnB_n,JnB_t,-1)
       
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn1_top(pos_top)
        pos_bottom = self.tcn1_bottom(pos_bottom)
        shuttle = self.tcn1_shuttle(shuttle)
      
        pos_top = self.tcn2_top(pos_top)
        pos_bottom = self.tcn2_bottom(pos_bottom)
        shuttle = self.tcn2_shuttle(shuttle)
   
        pos_top = self.tcn3_top(pos_top)
        pos_bottom = self.tcn3_bottom(pos_bottom)
        shuttle = self.tcn3_shuttle(shuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)

        # Step 5: Concatenate additional features with original TCN features
        x = torch.cat((x, x_additional), dim=1)
        n += 3  # Update number of people with the additional features (top, bottom, shuttle)

        # Step 6: Temporal Transformer Layer
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1  # Add 1 for class token

        x = x + self.embedding_tem
        x = self.pre_dropout(x)
        x = x.view(b * n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        mask = mask.repeat_interleave(n, dim=0)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Step 7: Interactional Transformer Layer
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1  # Add 1 for interaction class token

        x = x + self.embedding_inter
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
       
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output

class Tempose_SFwithGRUPredictor(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=32, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5, drop_rnn=0.1,
        type_size=32, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError

        # === 原本 Tempose 架構 ===
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

       # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+1, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)


        # 這裡 mlp_head 的輸入維度需要增加
        # 原本是 d_model * 3 (p1_conc + p2_conc + shuttle_cls)
        # 現在加上 rnn_feature_forward 和 rnn_feature_backward，每個都是 hid_size
        # 所以新的輸入維度是 d_model * 3 + hid_size * 2
        #self.mlp_head = MLP_Head(d_model * 3 + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p) # 修改這裡
        # self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        self.mlp_head = MLP_Head(d_model  + hid_size * 2, n_class, d_model * mlp_d_scale, drop_p)
        # self.mlp_head = MLP_Head(d_model  + hid_size, type_size, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        # === 新增lstm forward and backward的dropout 層 ===
        self.rnn_dropout = nn.Dropout(drop_rnn)

        # === 內建 LSTM ===
        # 這些參數現在可以直接從 BST_8 的 __init__ 接收
        self.rnn_forward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=False)
        self.rnn_backward_feature = GRUPredictor(type_size, emb_size, hid_size, n_layers=2, is_reverse=True)

        self.init_weights()

        # # 載入權重並加上 weights_only=True
        # # 修正權重加載：確保使用 self.rnn_feature
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor.pt", weights_only=True))
        # self.rnn_feature.load_state_dict(torch.load("weight/rnnwithpredictor_backward.pt", weights_only=True))

        
        

    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(
        self,
        JnB: Tensor,
        shuttle: Tensor,
        video_len: Tensor,
        ballrounds : Tensor,
        rally: Tensor

    ):
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim


        # 從 RNNPredictor 獲取特徵，確保在 no_grad 模式下運行
        with torch.no_grad():
            _, rnn_forward_feature = self.rnn_forward_feature(rally, ballrounds)
            _, rnn_backward_feature = self.rnn_backward_feature(rally, ballrounds)

        # 對 LSTM 特徵應用 dropout
        # lstm_feature_forward 原始形狀是 (N, 1, hid_size)，需要 squeeze(1) 變成 (N, hid_size)
        #pre_next_feature = self.lstm_dropout_forward(lstm_feature_forward.squeeze(1))
        


       # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        shuttle = shuttle.transpose(1, 2).contiguous()

        # Apply TCNs on positional and shuttle data
        shuttle = self.tcn_shuttle(shuttle)

        shuttle = shuttle.transpose(1, 2).contiguous()
        x_additional = shuttle.unsqueeze(1)
        # x_additional: (b, 1, t, d)

        # Shuttlecock Fusion (SF)
        x = torch.cat((x, x_additional), dim=1)
        n += 1

       # Concat cls token (temporal)
        class_token_tem = self.learned_token_tem.view(1, 1, 1, -1).expand(b, n, -1, -1)
        x = torch.cat((class_token_tem, x), dim=2)
        t += 1

        # Temporal embedding
        x = x + self.embedding_tem
        x: Tensor = self.pre_dropout(x)

        # Temporal TransformerLayers
        x = x.view(b*n, t, d)

        range_t = torch.arange(0, t, device=x.device).unsqueeze(0).expand(b, -1)
        video_len = video_len.unsqueeze(-1)
        mask = range_t < (1 + video_len)
        # mask: (b, t)
        mask = mask.repeat_interleave(n, dim=0)
        # mask: (b*n, t)
        
        x = self.encoder_tem(x, mask)
        x = x[:, 0].view(b, n, d)

        # Concat cls token (interactional)
        class_token_inter = self.learned_token_inter.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat((class_token_inter, x), dim=1)
        n += 1

        # Interactional embedding
        x = x + self.embedding_inter

        # Interactional TransformerLayers
        x = self.encoder_inter(x)
        x = x[:, 0].contiguous()
        
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        rnn_forward_feature = rnn_forward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_forward_feature = self.rnn_dropout(rnn_forward_feature)
        rnn_backward_feature = rnn_backward_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # rnn_backward_feature = self.rnn_dropout(rnn_backward_feature)
        # 使用 torch.cat 串接
        # dim=1 表示在特徵維度上進行串接
        rnn_feature = torch.cat((rnn_forward_feature, rnn_backward_feature), dim=1) #(batch_size, hid_size * 2)
        # rnn_feature = self.rnn_dropout(rnn_feature)
        # 將所有最終特徵拼接起來
        # 這裡 concat 的順序很重要，確保與 mlp_head 的輸入維度定義一致
        x_combined = torch.cat((x_main_features, rnn_feature), dim=1)  # (b, d_model + hid_size * 2)
        # x_combined = torch.cat((x_main_features, rnn_forward_feature), dim=1)  
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output 

# --- 測試程式碼 ---
if __name__ == "__main__":
    # 模型參數
    b, t, n, in_dim = 1, 5, 2, 10
    d_model = 100
    n_class = 31
    type_size = 32 # 0~30 類別 + padding_idx=31
    emb_size = 128
    hid_size = 256

    # 創建模型實例
    model = Tempose_TFwithOneStrokeLSTMPredictor(
        in_dim=in_dim,
        seq_len=t,
        n_class=n_class,
        d_model=d_model,
        type_size=type_size,
        emb_size=emb_size,
        hid_size=hid_size
    )

    # 創建測試資料
    # 使用你的程式碼片段來生成大部分的資料
    JnB = torch.randn(b, t, n, in_dim)
    shuttle = torch.randn(b, t, 2)
    pos = torch.randn(b, t, n, 2)
    videos_len = torch.tensor([t])
    # 模擬 ballrounds：假設預測這個 rally 的第 2 拍（index=1）
    ballrounds =  torch.randint(1, 6, (1, 1), dtype=torch.long)
    # q_len = torch.tensor([t, t], dtype=torch.long)
    # 模擬 src (rally) 序列，用來給 RNNPredictor
    # 這裡確保序列長度是 t，並使用 padding_idx=31
    src = torch.randint(0, 31, (1, 5), dtype=torch.long)
    pre_stroke = torch.randint(0, 31, (1, 1), dtype=torch.long)  # (B, 1)
    input_data=[JnB,shuttle, pos, videos_len, ballrounds, src, pre_stroke]
   
        # summary(model, input_data=[JnB, shuttle, pos, video_len, ballrounds, src])
    summary(model, input_data=input_data, depth=4, device='cpu')
   

 