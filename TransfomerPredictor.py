import torch
import torch.nn as nn
import math

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
        x = x + self.pe[:, :x.size(1), :]
        # x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    def __init__(self, type_size, emb_size, hid_size, n_heads=4, n_layers=2, is_reverse=False):
        super().__init__()
        self.is_reverse = is_reverse
        self.proj = nn.Linear(emb_size, hid_size) 
        # 類別嵌入
        self.embedding = nn.Embedding(
            num_embeddings=type_size,
            embedding_dim=emb_size,
            padding_idx=31
        )

        # 位置編碼
        self.pos_encoder = PositionalEncoding1D(emb_size)

        # Transformer 編碼器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=hid_size,
            batch_first=True,  # PyTorch transformer 預設 (L, N, E)
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

         # TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=hid_size,
            dropout=0.1,
            batch_first=True
        )

        # 多層 Decoder
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 輸出分類層
        self.fc = nn.Sequential(
            nn.Linear(emb_size, hid_size),
            nn.ReLU(inplace=True),
            nn.Linear(hid_size, type_size)
        )
    
    # def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
    #     """
    #     產生上三角遮罩矩陣（future mask），
    #     讓位置 i 只能看到 <= i 的位置。
    #     """
    #     mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    #     mask = mask.masked_fill(mask == 1, float('-inf'))
    #     return mask

    

    def forward(self, x, ballrounds, q_len, hx=None):
        # x: (N, L)
        N, L = x.shape
        # print('n',N)
        # 1. 先嵌入 + 加上位置編碼
        # tgt_mask = self.generate_square_subsequent_mask(N).to(x.device)
        tgt_mask = torch.triu(torch.ones(L, L, device=x.device)*float('-inf'), diagonal=1)
        # print('tgt mask', tgt_mask)  
        x = self.embedding(x)          # (N, L, emb)
        # x = x.transpose(0, 1)          # 轉成 (L, N, emb)      
        x = self.pos_encoder(x)        # 加入位置資訊
       

        # memory = self.transformer_encoder(x)  # [B, L, D]

        # # === 🔒 生成防洩漏 Mask ===
        # tgt_in = x[:, :-1, :]      # 去掉最後一個 (不能看未來)
        # tgt_mask = self.generate_square_subsequent_mask(tgt_in.size(1)).to(x.device)
        
        # 2. 準備 causal mask，防止看到未來拍
        # tgt_mask = self.generate_square_subsequent_mask(L).to(x.device)
        memory = torch.zeros_like(x)
        transformer_output = self.transformer_decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)  # (L, N, emb)
        # print('transformer output', transformer_output)
        # transformer_output = transformer_output.transpose(0, 1)  # 回到 (N, L, emb)
        # transformer_output = self.proj(transformer_output)
        # transformer_output = transformer_output.reshape(N * L, -1)
        # print('transformer output 1', transformer_output.shape)
        # 3. 分類層輸出
        out = self.fc(transformer_output)  # (N, L, type_size)
        # print('out', out.shape)
        # out = out.view(N, L, -1) #將維度還原，x: (N, L, n_tokens)

        # print('out', out.shape)
        # 4. 擷取指定 ballrounds 特徵
        d = out.size(2) #隱藏層的維度 d
        q_len_1_based_tensor = q_len.long()

        with torch.no_grad():
            if ballrounds is None:
                if not self.is_reverse:
                    index_base = q_len_1_based_tensor - 1
                else:
                    index_base = q_len_1_based_tensor - q_len_1_based_tensor
            else:
                ballrounds_1_based = ballrounds.long()
                if not self.is_reverse:
                    index_base = ballrounds_1_based - 1
                else:
                    index_base = q_len_1_based_tensor - ballrounds_1_based

            index_for_gather = index_base.unsqueeze(1).unsqueeze(2).expand(N, 1, d)
   

            feature = torch.gather(transformer_output, dim=1, index=index_for_gather)
            # print('feature', feature.shape)
        return out, feature
