# TemPose: a new skeleton-based transformer model designed for fine-grained motion recognition in badminton
# (2023/08) https://ieeexplore.ieee.org/document/10208321
# Authors: Magnus Ibh, Stella Grasshof, Dan Witzner, Pascal Madeleine
# Modified by Jing-Yuan Chang
# Modified by Tzu-Yu Liu

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D
from torchinfo import summary
from stgcn import ST_GCN_Block, ST_GCN_10

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, drop_p) -> None:
        super().__init__()
        d_cat = d_head * n_head

        self.h = n_head
        self.to_q = nn.Linear(d_model, d_cat, bias=False)
        self.to_kv = nn.Linear(d_model, d_cat * 2, bias=False)
        self.scale = d_head**-0.5

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(drop_p)  # This shouldn't be inplace.
        )
        
        self.tail = nn.Sequential(
            nn.Linear(d_cat, d_model),
            nn.Dropout(drop_p, inplace=True)
        ) if n_head != 1 or d_cat != d_model else nn.Identity()
    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor = None):

        # x1: (B, Q, d_model)
        # x2: (B, K, d_model)

        q = self.to_q(x1)
        kv = self.to_kv(x2)

        B, Q, _ = q.shape
        _, K, _ = kv.shape

        # -------------------------
        # Query
        # -------------------------
        q = q.view(B, Q, self.h, -1)
        q = q.transpose(1, 2).contiguous()
        # (B, h, Q, d_head)

        # -------------------------
        # Key / Value
        # -------------------------
        kv = kv.view(B, K, self.h, 2, -1)
        kv = kv.transpose(1, 2).contiguous()
        # (B, h, K, 2, d_head)

        k = kv[:, :, :, 0, :]
        v = kv[:, :, :, 1, :]

        k_T = k.transpose(-1, -2).contiguous()
        # (B, h, d_head, K)

        # -------------------------
        # Attention
        # -------------------------
        dots = (q @ k_T) * self.scale
        # (B, h, Q, K)

        # -------------------------
        # Mask
        # -------------------------
        if mask is not None:
            # mask: (B, K)
            mask = mask.view(B, 1, 1, K)
            dots = dots.masked_fill(mask == 0, -torch.inf)

        coef = self.attend(dots)

        attention = coef @ v
        # (B, h, Q, d_head)

        # -------------------------
        # Merge heads
        # -------------------------
        out = attention.transpose(1, 2).reshape(B, Q, -1)

        out = self.tail(out)

        return out


class CrossTransformerLayer(nn.Module):
    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p) -> None:
        super().__init__()
        self.layer_norm1_x1 = nn.LayerNorm(d_model)
        self.layer_norm1_x2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, d_head, n_head, drop_p)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)

    def forward(self, x1, x2, mask=None):

        z1 = self.layer_norm1_x1(x1)
        z2 = self.layer_norm1_x2(x2)

        attn = self.cross_attn(z1, z2, mask)
       

        x = attn + x1

        z = self.layer_norm2(x)

        x = self.ff(z) + x

        return x
    
class MLP(nn.Module):
    '''Same as MLP_Block in TemPose paper.'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hd_dim),
            nn.GELU(),
            nn.Dropout(drop_p, inplace=True),
            nn.Linear(hd_dim, out_dim)
        )

    def forward(self, x: Tensor):
        return self.mlp(x)


class MLP_Head(nn.Module):
    '''Same as MLP_Head in TemPose.'''
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, out_dim, hd_dim, drop_p)

    def forward(self, x: Tensor):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    
class ResidualMLP_Head(nn.Module):
    def __init__(self, in_dim, out_dim, hd_dim, drop_p=0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hd_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop_p)
        self.fc2 = nn.Linear(hd_dim, in_dim)   # 回到 in_dim，方便做殘差
        self.classifier = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor):
        residual = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + residual     # 殘差連接 (shape 對齊)
        return self.classifier(x)


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
    
class TemporalSelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super(TemporalSelfAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)  # Convert to (seq_len, batch_size, d_model) for attention
        x, _ = self.attn(x, x, x)  # Self-attention over the sequence
        x = x.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, d_model)
        return x


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

class TCN(nn.Module):
    '''Same as TCN in TemPose. There is a bit different from the original TCN.'''
    def __init__(self, in_channel, channels: list[int], kernel_size=5, drop_p=0.3) -> None:
        '''`kernel_size` should be an odd number, so the output sequence length can remain the same as input.'''
        super().__init__()
        layers = []
        for i in range(len(channels)):
            in_ch = in_channel if i == 0 else channels[i-1]
            out_ch = channels[i]
            
            dilation = i * 2 + 1
            padding = (kernel_size - 1) * dilation // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(drop_p, inplace=True)
            ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: Tensor):
        return self.net(x)

class MultiScaleTCN(nn.Module):
    def __init__(self, in_channel, channels: list[int], kernel_sizes=[3,5,7], drop_p=0.3):
        super().__init__()
        self.branches = nn.ModuleList([
            TCN(in_channel, channels, kernel_size=k, drop_p=drop_p)
            for k in kernel_sizes
        ])

    def forward(self, x: Tensor):
        # x: (batch, in_channel, seq_len)
        out = [branch(x) for branch in self.branches]  # list of (b, out_ch, t)
        return torch.cat(out, dim=1)  # concat channel


class TemPose_V(nn.Module):
    '''Similar to TemPose_TF in TemPose.'''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4
    ):
        super().__init__()

        self.project = nn.Linear(in_dim, d_model)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people, d_model))
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
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

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

        x = self.mlp_head(x)
        return x


class TemPose_PF(nn.Module):
    '''For ablation studies.

    Equal to TemPose_TF without the shuttlecock trajectory
    or TemPose_V with the player positions.
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
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

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people+2, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people+2, d_model))
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
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        # pos_top: (b, 2, t)
        # pos_bottom: (b, 2, t)

        # TCNs
        pos_top: Tensor = self.tcn_top(pos_top)
        pos_bottom: Tensor = self.tcn_bottom(pos_bottom)
        # pos_top: (b, d, t)
        # pos_bottom: (b, d, t)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        x_additional = torch.stack((pos_top, pos_bottom), dim=1)
        # x_additional: (b, 2, t, d)

        # Positions Fusion (PF)
        x = torch.cat((x, x_additional), dim=1)
        n += 2

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

        x = self.mlp_head(x)
        return x

class TemPose_VwithStrokeFusion(nn.Module):
    '''Similar to TemPose_TF in TemPose.'''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()

        self.project = nn.Linear(in_dim, d_model)

        # Temporal TransformerLayers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional TransformerLayers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+n_people, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head
        self.mlp_head = MLP_Head(d_model+hid_size, n_class, d_model * mlp_d_scale, drop_p)
        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)

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
        video_len: Tensor,  # video_len: (b)
        pre_next_stroke: Tensor
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

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
        pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
        x_main_features = x.view(b, -1)  # Ensure the shape is (batch_size, d_model)
        pre_next_feature = pre_next_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        
        pre_next_feature = self.stroke_dropout(pre_next_feature)
        x_combined = torch.cat((x_main_features, pre_next_feature), dim=1)  # (b, d_model + hid_size)
        x_output = self.mlp_head(x_combined)
        return x_output

class TemPose_SF(nn.Module):
    '''For ablation studies.

    Equal to TemPose_TF without the player positions
    or TemPose_V with the shuttlecock trajectory.
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()

        self.project = nn.Linear(in_dim, d_model)

        # TCNs
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
        shuttle: Tensor,  # shuttle: (b, t, 2)
        video_len: Tensor  # video_len: (b)
    ):
        JnB = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB)
        b, n, t, d = x.shape

        shuttle = shuttle.transpose(1, 2).contiguous()
        # shuttle: (b, 2, t)

        # TCN
        shuttle: Tensor = self.tcn_shuttle(shuttle)
        # shuttle: (b, d, t)

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

        x = self.mlp_head(x)
        return x


class TemPose_TF(nn.Module):
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

        x = self.mlp_head(x)
        return x


class TemPose2_TF(nn.Module):
    '''Comparing to TemPose_TF, replace poses projection by TCN.'''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)

        # Temporal Fusion Preprocessing
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
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        x: Tensor = self.tcn_pose(JnB)
        x = x.view(b, n, -1, t).transpose(-2, -1)
        _, _, _, d = x.shape

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

        x = self.mlp_head(x)
        return x



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

class TemPose_TFwithStrokeFusion(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model] 
        
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

      
       
        # tcn_out_ch_per_branch = d_model // (2 * 3)  # 每個 branch 輸出通道數，3 個 kernel
        # kernel_sizes = [3,7,5]
        # channels_per_branch = [d_model // (2*len(kernel_sizes)), d_model // (2*len(kernel_sizes))]  # list

        # self.tcn_top = MultiScaleTCN(2,channels_per_branch, kernel_sizes, drop_p)
        # self.tcn_bottom = MultiScaleTCN(2, channels_per_branch, kernel_sizes, drop_p)
        # self.tcn_shuttle = MultiScaleTCN(2, channels_per_branch, kernel_sizes, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head for output  
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head = ResidualMLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)

        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, JnB: Tensor, pos: Tensor, shuttle: Tensor, video_len: Tensor, pre_next_stroke: Tensor):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_next_stroke.shape)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim

        
        pre_next_feature = self.stroke_fusion_model(pre_next_stroke)

        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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
        #print(f"x shape: {x.shape}")
        
        #print(f"pre_next_feature shape: {pre_next_feature.shape}")
        # Step 8: Extract the main features from interactional transformer output
        
        #print(f"x_main_features shape: {x_main_features.shape}")
        # Step 9: Combine main features with stroke fusion features
        pre_next_feature = self.stroke_dropout(pre_next_feature)
        x_combined = torch.cat((x_main_features, pre_next_feature), dim=1)  # (b, d_model + hid_size)
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


class TemPose_TFwithtwoStageStrokeFusion(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model] 
        
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

      
       
        # tcn_out_ch_per_branch = d_model // (2 * 3)  # 每個 branch 輸出通道數，3 個 kernel
        # kernel_sizes = [3,7,5]
        # channels_per_branch = [d_model // (2*len(kernel_sizes)), d_model // (2*len(kernel_sizes))]  # list

        # self.tcn_top = MultiScaleTCN(2,channels_per_branch, kernel_sizes, drop_p)
        # self.tcn_bottom = MultiScaleTCN(2, channels_per_branch, kernel_sizes, drop_p)
        # self.tcn_shuttle = MultiScaleTCN(2, channels_per_branch, kernel_sizes, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head for output  
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head = ResidualMLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
       
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)

        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, JnB: Tensor, pos: Tensor, shuttle: Tensor, video_len: Tensor, pre_next_stroke: Tensor):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_next_stroke.shape)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim

        
        pre_next_feature = self.stroke_fusion_model(pre_next_stroke.long())

        

        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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
        #print(f"x shape: {x.shape}")
        
        #print(f"pre_next_feature shape: {pre_next_feature.shape}")
        # Step 8: Extract the main features from interactional transformer output
        
        #print(f"x_main_features shape: {x_main_features.shape}")
        # Step 9: Combine main features with stroke fusion features
        pre_next_feature = self.stroke_dropout(pre_next_feature)
        x_combined = torch.cat((x_main_features, pre_next_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output


class TemPose_TFwithOneStroke(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head_tem = MLP_Head(d_model+ hid_size, n_class, d_model * mlp_d_scale, drop_p) 

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        
        


        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, 
                JnB: Tensor, 
                pos: Tensor, 
                shuttle: Tensor,
                video_len: Tensor, 
                pre_stroke: Tensor
            ):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_stroke)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
     
        x = self.project(JnB)  # Project to the desired model dimension
      
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
       
        pre_feature = self.stroke_emb(pre_stroke.long())
        
        
        
        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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

        # if pre_stroke is None:
        #    x_output = self.mlp_head_tem (x_main_features)
        #    return x_output
        
       
        pre_feature = pre_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
       
   
        pre_feature = self.stroke_dropout(pre_feature)
        x_combined = torch.cat((x_main_features, pre_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
      
        
        return x_output  



class TemPose_TFwithNextStroke(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head_tem = MLP_Head(d_model+ hid_size, n_class, d_model * mlp_d_scale, drop_p) 

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        
        


        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, 
                JnB: Tensor, 
                pos: Tensor, 
                shuttle: Tensor,
                video_len: Tensor, 
                next_stroke: Tensor
            ):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_stroke)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
     
        x = self.project(JnB)  # Project to the desired model dimension
      
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
       
        next_feature = self.stroke_emb(next_stroke.long())
        
        
        
        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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

        # if pre_stroke is None:
        #    x_output = self.mlp_head_tem (x_main_features)
        #    return x_output
        
       
        next_feature = next_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
       
   
        next_feature = self.stroke_dropout(next_feature)
        x_combined = torch.cat((x_main_features, next_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
      
        
        return x_output


#add gcn block
class TemPose_GFwithStrokeFusion(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        

        #GCN block
        # GCN-based model for poses
        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=hid_size , kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=hid_size , kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=hid_size*2, out_channels=hid_size , kernel_size=3, stride=1, padding=1)
        # self.dropout_pose = nn.Dropout(drop_p)
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
        self.encoder_tem = DWTransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head = ResidualMLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)

        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, JnB: Tensor, pos: Tensor, shuttle: Tensor, video_len: Tensor, pre_next_stroke: Tensor):

        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_next_stroke.shape)
      
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        JnB_b, JnB_n, JnB_t, JnB_d = JnB.shape
        JnB = JnB.view(JnB_b, JnB_n* JnB_t, JnB_d)  # 去掉 batch_size 維度，變成 [2, 30, 100]
        JnB = F.relu(self.conv1(JnB.permute(0, 2, 1)))
        # JnB = self.dropout_pose(JnB)
        # JnB = F.relu(self.conv2(JnB))
        JnB = JnB.permute(0, 2, 1).view(JnB_b,JnB_n,JnB_t,-1)
       

        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
          # 去掉 batch_size 維度，變成 [2, 30, 100]


        pre_next_feature = self.stroke_fusion_model(pre_next_stroke)

        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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
        pre_next_feature = pre_next_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        #print(f"x shape: {x.shape}")
        
        #print(f"pre_next_feature shape: {pre_next_feature.shape}")
        # Step 8: Extract the main features from interactional transformer output
        
        #print(f"x_main_features shape: {x_main_features.shape}")
        # Step 9: Combine main features with stroke fusion features
        pre_next_feature = self.stroke_dropout(pre_next_feature)
        x_combined = torch.cat((x_main_features, pre_next_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output


class TemPose_TFwithTheStroke(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head_tem = MLP_Head(d_model+ hid_size, n_class, d_model * mlp_d_scale, drop_p) 

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        
        


        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, 
                JnB: Tensor, 
                pos: Tensor, 
                shuttle: Tensor,
                video_len: Tensor, 
                the_stroke: Tensor
            ):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_stroke)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
     
        x = self.project(JnB)  # Project to the desired model dimension
      
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
       
        stroke_feature = self.stroke_emb(the_stroke.long())
        
        
        
        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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

        # if pre_stroke is None:
        #    x_output = self.mlp_head_tem (x_main_features)
        #    return x_output
        
       
        stroke_feature = stroke_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
       
   
        stroke_feature = self.stroke_dropout(stroke_feature)
        x_combined = torch.cat((x_main_features, stroke_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
      
        
        return x_output  


class TemPose_TF_for_TwoStage(nn.Module):
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
            self.load_state_dict(torch.load("weight/tempose_TF_JnB_bone_dataset_npy_collated_11.pt", weights_only=True))
            print("成功載入TemPose模型權重。")
        except FileNotFoundError:
            print("錯誤：找不到TemPose模型權重檔案。")
        
                
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
       
        x = self.mlp_head(x)
        # 使用 torch.argmax 來找到每個樣本預測的類別
        predicted_class = torch.argmax(x, dim=1)  # 在第二維度上選擇最大值所在的索引
       
        predicted_class = predicted_class.clone().detach().view(-1, 1)


        # 如果你有 batch_size > 1，這將會是一個包含所有樣本預測類別的張量
        
                
        return predicted_class
    
class TemPose_TFwithTwoStroke(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size*2 , n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head_tem = MLP_Head(d_model+ hid_size, n_class, d_model * mlp_d_scale, drop_p) 

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        
        


        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, 
                JnB: Tensor, 
                pos: Tensor, 
                shuttle: Tensor,
                video_len: Tensor, 
                pre_stroke: Tensor,
                the_stroke: Tensor
            ):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_stroke)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
     
        x = self.project(JnB)  # Project to the desired model dimension
      
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
       

        pre_feature = self.stroke_emb(pre_stroke.long())
        stroke_feature = self.stroke_emb(the_stroke.long())
        
        
        
        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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

        # if pre_stroke is None:
        #    x_output = self.mlp_head_tem (x_main_features)
        #    return x_output
        
        pre_feature = pre_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        stroke_feature = stroke_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
       
        pre_feature = self.stroke_dropout(pre_feature)
        stroke_feature = self.stroke_dropout(stroke_feature)
        stroke_feature_combine =  torch.cat((pre_feature, stroke_feature), dim=1)
        x_combined = torch.cat((x_main_features, stroke_feature_combine), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
      
        
        return x_output  
    
class TemPose_TFIIwithStrokeFusion(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=35, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError
        
        # 是否為第一階段
        self.first_phase = True

        #original tempose
        self.tempose_front = TemPose_TF_for_TwoStage(
                    in_dim=in_dim,
                    n_class=n_class,
                    seq_len=seq_len,
                    depth_inter=2
                )

        
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)
        self.hid_dim = 256

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

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)

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

    def set_train(self, first_phase: bool):
        self.first_phase = first_phase
        if first_phase:  # 第一階段
            self.train()
        else:  # 第二階段
            for name, m in self.named_children():
                if name == 'stroke_fusion_model' or name == 'mlp_head':
                    m.train()
                else:
                    m.eval()

    def set_eval(self, first_phase: bool):
        self.first_phase = first_phase
        self.eval()
    
    def forward(
            self, 
            JnB: Tensor, 
            pos: Tensor, 
            shuttle: Tensor, 
            video_len: Tensor, 
            pre_next_stroke: Tensor
        ):

        JnB_new = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB_new)
        b, _, _, _ = x.shape
   
        if self.first_phase:
            pre_next_feature = torch.zeros(b, self.hid_dim, device=x.device)
            x = self.tempose_front(JnB, pos, shuttle, video_len)
        else:
            pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
            with torch.no_grad():
                x = self.tempose_front(JnB, pos, shuttle, video_len)

        pre_next_feature = pre_next_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_next_feature = self.stroke_dropout(pre_next_feature)

        x_combined = torch.cat((x, pre_next_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)

        return x_output

class TemPose_TFwithThreeStroke(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size*2 , n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head_tem = MLP_Head(d_model+ hid_size, n_class, d_model * mlp_d_scale, drop_p) 

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        


        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, 
                JnB: Tensor, 
                pos: Tensor, 
                shuttle: Tensor,
                video_len: Tensor, 
                pre_next_stroke: Tensor,
                the_stroke: Tensor
            ):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_stroke)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
     
        x = self.project(JnB)  # Project to the desired model dimension
      
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
       

        pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
        stroke_feature = self.stroke_emb(the_stroke.long())
        
        
        
        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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

        # if pre_stroke is None:
        #    x_output = self.mlp_head_tem (x_main_features)
        #    return x_output
        
        pre_next_feature = pre_next_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        stroke_feature = stroke_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
       
        stroke_feature_combine =  torch.cat((pre_next_feature, stroke_feature), dim=1)
        stroke_feature_combine = self.stroke_dropout(stroke_feature_combine)
        x_combined = torch.cat((x_main_features, stroke_feature_combine), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
      
        
        return x_output  
    


class TemPose2_TFwithStrokeFusion(nn.Module):
    '''For ablation studies.

    Equal to TemPose_TF without the shuttlecock trajectory
    or TemPose_V with the player positions.
    '''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=35, emb_size=128, hid_size=256 
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.project = nn.Linear(in_dim, d_model)
        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)

        # Temporal Fusion Preprocessing
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


        
        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 

        self.d_model = d_model
        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)

        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False


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
            JnB: Tensor, 
            pos: Tensor, 
            shuttle: Tensor, 
            video_len: Tensor, 
            pre_next_stroke: Tensor
        ):

        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        x: Tensor = self.tcn_pose(JnB)
        x = x.view(b, n, -1, t).transpose(-2, -1)
        _, _, _, d = x.shape

   
        pre_next_feature = self.stroke_fusion_model(pre_next_stroke)



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
        pre_next_feature = pre_next_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        
        pre_next_feature = self.stroke_dropout(pre_next_feature)
        x_combined = torch.cat((x_main_features, pre_next_feature), dim=1)  # (b, d_model + hid_size)



        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        return x_output

class TemPose2_TF_for_TwoStage(nn.Module):
    '''Comparing to TemPose_TF, replace poses projection by TCN.'''
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
    ):
        '''`d_model` should be an even number.'''
        super().__init__()
        if n_people > 2:
            raise NotImplementedError

        self.project = nn.Linear(in_dim, d_model)
        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)

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
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        x: Tensor = self.tcn_pose(JnB)
        x = x.view(b, n, -1, t).transpose(-2, -1)
        _, _, _, d = x.shape

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
        # x = self.mlp_head(x)
        return x_main_features

class TemPose2_TFIIwithStrokeFusion(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=35, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=35, emb_size=128, hid_size=256 
    ):
        super().__init__()

        if n_people > 2:
            raise NotImplementedError
        
        # 是否為第一階段
        self.first_phase = True

        #original tempose2
        self.tempose_front = TemPose2_TF_for_TwoStage(
                    in_dim=in_dim,
                    n_class=n_class,
                    seq_len=seq_len,
                    depth_inter=2
                )

        # TCN layers
        self.project = nn.Linear(in_dim, d_model)
        self.hid_dim = 256

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
    
    
    def set_train(self, first_phase: bool):
        self.first_phase = first_phase
        if first_phase:  # 第一階段
            self.train()
        else:  # 第二階段
            for name, m in self.named_children():
                if name == 'stroke_fusion' or name == 'mlp_head':
                    m.train()
                else:
                    m.eval()

    def set_eval(self, first_phase: bool):
        self.first_phase = first_phase
        self.eval()
    
    def forward(
            self, 
            JnB: Tensor, 
            pos: Tensor, 
            shuttle: Tensor, 
            video_len: Tensor, 
            pre_next_stroke: Tensor
        ):

        JnB_new = JnB.transpose(1, 2).contiguous()
        # JnB: (b, n, t, input_dim)
        
        x = self.project(JnB_new)
        b, _, _, _ = x.shape

   
        if self.first_phase:
            pre_next_feature = torch.zeros(b, self.hid_dim, device=x.device)
            x = self.tempose_front(JnB, pos, shuttle, video_len)
        else:
            pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
            with torch.no_grad():
                x = self.tempose_front(JnB, pos, shuttle, video_len)

        pre_next_feature = pre_next_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        pre_next_feature = self.stroke_dropout(pre_next_feature)

        x_combined = torch.cat((x, pre_next_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)

        return x_output

class TemPose3D_TFwithStrokeFusion(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(3, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 3, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 

        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)

        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, JnB: Tensor, pos: Tensor, shuttle: Tensor, video_len: Tensor, pre_next_stroke: Tensor):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_next_stroke.shape)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim

        
        pre_next_feature = self.stroke_fusion_model(pre_next_stroke)

        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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
        #print(f"x shape: {x.shape}")
        
        #print(f"pre_next_feature shape: {pre_next_feature.shape}")
        # Step 8: Extract the main features from interactional transformer output
        
        #print(f"x_main_features shape: {x_main_features.shape}")
        # Step 9: Combine main features with stroke fusion features
        pre_next_feature = self.stroke_dropout(pre_next_feature)
        x_combined = torch.cat((x_main_features, pre_next_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output


class ConvTemPose_TFwithStrokeFusion(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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
        self.encoder_tem = DWTransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 3, d_model))
        self.encoder_inter = DWTransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 

        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)

        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, JnB: Tensor, pos: Tensor, shuttle: Tensor, video_len: Tensor, pre_next_stroke: Tensor):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_next_stroke.shape)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim

        
        pre_next_feature = self.stroke_fusion_model(pre_next_stroke)

        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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
        #print(f"x shape: {x.shape}")
        
        #print(f"pre_next_feature shape: {pre_next_feature.shape}")
        # Step 8: Extract the main features from interactional transformer output
        
        #print(f"x_main_features shape: {x_main_features.shape}")
        # Step 9: Combine main features with stroke fusion features
        pre_next_feature = self.stroke_dropout(pre_next_feature)
        x_combined = torch.cat((x_main_features, pre_next_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output      
    

class TemPose_SFwithOneStroke(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
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

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        
        


        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, 
                JnB: Tensor, 
                shuttle: Tensor,
                video_len: Tensor, 
                pre_stroke: Tensor
            ):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_next_stroke.shape)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
     
        x = self.project(JnB)  # Project to the desired model dimension
      
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
       
       
        
        pre_feature = self.stroke_emb(pre_stroke)
        
        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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
        pre_feature = pre_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        #print(f"x shape: {x.shape}")
        
        #print(f"pre_next_feature shape: {pre_next_feature.shape}")
        # Step 8: Extract the main features from interactional transformer output
        
        #print(f"x_main_features shape: {x_main_features.shape}")
        # Step 9: Combine main features with stroke fusion features
        pre_feature = self.stroke_dropout(pre_feature)
        x_combined = torch.cat((x_main_features, pre_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output 


class TemPose_TFwithPreShuttle(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
        # TCN layers
        self.project = nn.Linear(in_dim, d_model)

        # TCN layers for top, bottom, shuttle
        tcn_channels = [d_model // 2, d_model]
        self.tcn_top = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_bottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_shuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_preshuttle = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_pretop = TCN(2, tcn_channels, tcn_kernel_size, drop_p)
        self.tcn_prebottom = TCN(2, tcn_channels, tcn_kernel_size, drop_p)

        # Temporal Transformer Layers
        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, n_people + 6, 1 + seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)

        # Interactional Transformer Layers
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1 + n_people + 6, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model , n_class, d_model * mlp_d_scale, drop_p) 

        # Stroke emb model
        # self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        # self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        
        


        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, 
                JnB: Tensor, 
                pos: Tensor, 
                shuttle: Tensor,
                video_len: Tensor, 
                prepos: Tensor, 
                preshuttle: Tensor
            ):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_next_stroke.shape)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        
     
        x = self.project(JnB)  # Project to the desired model dimension
      
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
       
       
        
        # pre_feature = self.stroke_emb(pre_stroke)
        
        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


        # Step 4: Process position and shuttle data (position features for top/bottom and shuttle features)
        pos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        pos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        shuttle = shuttle.transpose(1, 2).contiguous()
        prepos_top = pos[:, :, 0, :].transpose(1, 2).contiguous()
        prepos_bottom = pos[:, :, 1, :].transpose(1, 2).contiguous()
        preshuttle = preshuttle.transpose(1, 2).contiguous()


        # Apply TCNs on positional and shuttle data
        pos_top = self.tcn_top(pos_top)
        pos_bottom = self.tcn_bottom(pos_bottom)
        shuttle = self.tcn_shuttle(shuttle)
        prepos_top = self.tcn_pretop(prepos_top)
        prepos_bottom = self.tcn_prebottom(prepos_bottom)
        preshuttle = self.tcn_preshuttle(preshuttle)

        pos_top = pos_top.transpose(1, 2)
        pos_bottom = pos_bottom.transpose(1, 2)
        shuttle = shuttle.transpose(1, 2)
        prepos_top = prepos_top.transpose(1, 2)
        prepos_bottom = prepos_bottom.transpose(1, 2)
        preshuttle = preshuttle.transpose(1, 2)
        
        x_additional = torch.stack((pos_top, pos_bottom, shuttle,prepos_top, prepos_bottom, preshuttle), dim=1)
        
        # Step 5: Concatenate additional features with original TCN features
        
        x = torch.cat((x, x_additional), dim=1)
        
        n += 6  # Update number of people with the additional features (top, bottom, shuttle, preshuttle)
        
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
        # pre_feature = pre_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        # #print(f"x shape: {x.shape}")
        
        #print(f"pre_next_feature shape: {pre_next_feature.shape}")
        # Step 8: Extract the main features from interactional transformer output
        
        #print(f"x_main_features shape: {x_main_features.shape}")
        # Step 9: Combine main features with stroke fusion features
        # pre_feature = self.stroke_dropout(pre_feature)
        # x_combined = torch.cat((x_main_features, pre_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_main_features)
        
        return x_output


class TemPose_TF2withOneStroke(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 

        #original tempose
        self.tempose = TemPose_TF_for_TwoStage(
                    in_dim=in_dim,
                    n_class=n_class,
                    seq_len=seq_len,
                    depth_inter=2
                )

        # Stroke emb model
        self.stroke_emb = Stroke_Embedding(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        
        


        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, 
                JnB: Tensor, 
                pos: Tensor, 
                shuttle: Tensor,
                preJnB: Tensor, 
                prepos: Tensor, 
                preshuttle: Tensor,
                video_len: Tensor
            ):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_next_stroke.shape)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
     
        x = self.project(JnB)  # Project to the desired model dimension
      
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
       
       
        with torch.no_grad():
            pre_feature = self.tempose(preJnB,prepos,preshuttle,video_len)
        pre_feature = self.stroke_emb(pre_feature)
        
        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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
        pre_feature = pre_feature.view(b, -1)  # Ensure it is (batch_size, hid_size)
        #print(f"x shape: {x.shape}")
        
        #print(f"pre_next_feature shape: {pre_next_feature.shape}")
        # Step 8: Extract the main features from interactional transformer output
        
        #print(f"x_main_features shape: {x_main_features.shape}")
        # Step 9: Combine main features with stroke fusion features
        pre_feature = self.stroke_dropout(pre_feature)
        x_combined = torch.cat((x_main_features, pre_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output
    

class TemPose_TF2withStrokeFusion(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output
        self.mlp_head = MLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p) 

        # Stroke fusion model
        self.stroke_fusion_model = Stroke_fusion(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        
        self.tempose = TemPose_TF_for_TwoStage(
                    in_dim, seq_len, n_class=31, n_people=2,
                    d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
                    drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5
                )


        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
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

    def forward(self, 
                JnB: Tensor, 
                pos: Tensor, 
                shuttle: Tensor,
                nextJnB: Tensor,
                nextpos: Tensor, 
                nextshuttle: Tensor,
                preJnB: Tensor,
                prepos: Tensor, 
                preshuttle: Tensor, 
                video_len: Tensor
            ):
        # print("JnB shape:", JnB.shape)
        # print("pos shape:", pos.shape)
        # print("shuttle shape:", shuttle.shape)
        # print("video_len shape:", video_len.shape)
        # print("pre_next_stroke shape:", pre_next_stroke.shape)
        
        # Step 1: Process JnB -> (b, n, t, input_dim)
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
     
        x = self.project(JnB)  # Project to the desired model dimension
      
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
       
        with torch.no_grad():
            pre_feature = self.tempose(preJnB,prepos,preshuttle,video_len)
            next_feature = self.tempose(nextJnB,nextpos,nextshuttle,video_len)
        pre_next_stroke = torch.cat((pre_feature, next_feature), dim=1)
        pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
        
        #pre_next_feature = pre_next_feature_per_stroke  # 直接使用原始特徵


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
        #print(f"x shape: {x.shape}")
        
        #print(f"pre_next_feature shape: {pre_next_feature.shape}")
        # Step 8: Extract the main features from interactional transformer output
        
        #print(f"x_main_features shape: {x_main_features.shape}")
        # Step 9: Combine main features with stroke fusion features
        pre_next_feature = self.stroke_dropout(pre_next_feature)
        x_combined = torch.cat((x_main_features, pre_next_feature), dim=1)  # (b, d_model + hid_size)
        # Step 10: Pass the combined features through the MLP Head for final classification output
        x_output = self.mlp_head(x_combined)
        
        return x_output

class Stroke_EmbeddingTA(nn.Module):
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
        if pre_stroke.dim() > 1:
            pre_stroke = pre_stroke[:, 0]
        pre_emb = self.embedding(pre_stroke)


        # 融合： (N, hid_size)
        x_final = self.fusion(pre_emb)
        return x_final

class Stroke_TA(nn.Module): 

    def __init__(self, type_size, emb_size, hid_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=type_size,
            embedding_dim=emb_size,
            padding_idx=0
        )

        self.fusion = nn.Sequential(
            nn.Linear(emb_size , hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size)
        )

    def forward(self, pre_next_stroke):

        pre = pre_next_stroke[:,0]
        nxt = pre_next_stroke[:,1]

        pre_emb = self.embedding(pre)
        next_emb = self.embedding(nxt)
        pre_emb = self.fusion(pre_emb)
        next_emb = self.fusion(next_emb)

        return pre_emb, next_emb

class TemPose_TFwithtwostageStrokeTA(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output  
        self.mlp_head = MLP_Head(d_model * 2, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head = ResidualMLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.Jnb_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )

        self.shuttle_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )
        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        # # Bidirectional Cross Attention
        # self.cross_shuttle_to_semantic = CrossTransformerLayer(
        #     d_model, d_head, n_head,
        #     d_model * mlp_d_scale,
        #     drop_p
        # )

        # self.cross_semantic_to_shuttle = CrossTransformerLayer(
        #     d_model, d_head, n_head,
        #     d_model * mlp_d_scale,
        #     drop_p
        # )
        self.stroke_ca = Stroke_TA(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        self.semantic_proj = nn.Linear(hid_size, d_model)
        self.semantic_pool = nn.Linear(d_model, 1) 
        self.semantic_type_embedding = nn.Parameter(torch.zeros(1, 2, d_model)) #0527
        self.semantic_norm = nn.LayerNorm(d_model) #0527
        self.temporal_semantic_pool = nn.Linear(d_model, 1) #0527
        self.semantic_global_gate = nn.Sequential(  #0527
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)
        nn.init.normal_(self.semantic_type_embedding, std=0.02) #0527

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, JnB: Tensor, pos: Tensor, shuttle: Tensor, video_len: Tensor, pre_next_stroke: Tensor):
       
        

        # Semantic Embedding
        # pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
        pre_emb, next_emb = self.stroke_ca(pre_next_stroke)
        pre_emb = self.stroke_dropout(self.semantic_proj(pre_emb))
        next_emb = self.stroke_dropout(self.semantic_proj(next_emb))


        #shuttle attends semantic
        semantic_tokens = torch.stack(
            [pre_emb, next_emb],
            dim=1
        )
        semantic_tokens = self.semantic_norm(
            semantic_tokens + self.semantic_type_embedding
        )
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
        
        # Cross Attention
        T_shuttle = shuttle.shape[1]

        range_t = torch.arange(
            T_shuttle,
            device=shuttle.device
        ).unsqueeze(0)

      


        # shuttle attends semantic
        semantic_mask = None

        shuttle_attn = self.cross_trans(
            shuttle,            # Q
            semantic_tokens,    # KV
            semantic_mask
        )

      
        shuttle_alpha = self.shuttle_gate(
            torch.cat([shuttle, shuttle_attn], dim=-1)
        )
        shuttle = shuttle + shuttle_alpha * shuttle_attn
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
        JnB_top = x[:, 0]       # (B,T,D)
        JnB_bottom = x[:, 1]    # (B,T,D)

        semantic_context_top = self.cross_trans(
            JnB_top,          # Q
            semantic_tokens,  # KV
            None
        )

        semantic_context_bottom = self.cross_trans(
            JnB_bottom,
            semantic_tokens,
            None
        )
        semantic_feature = shuttle_attn

        
        #shuttle attends semantic
        gate_input_top = torch.cat(
        [JnB_top, semantic_context_top],
        dim=-1
        )

        gate_input_bottom = torch.cat(
            [JnB_bottom, semantic_context_bottom],
            dim=-1
        )
        
      

        alpha = self.Jnb_gate(gate_input_top)      # (B,T,1)
        beta  = self.Jnb_gate(gate_input_bottom)   # (B,T,1)
        # gamma = gate_weight[..., 2].unsqueeze(-1)

        # x = (
        #     alpha * semantic_context
        #     +
        #     (1 - alpha) * x
        # )
        JnB_top = JnB_top + alpha * semantic_context_top

        JnB_bottom = JnB_bottom + beta * semantic_context_bottom
        x = torch.stack(
            [JnB_top, JnB_bottom],
            dim=1
        )
        # Step 5: Concatenate additional features with original TCN features
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
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
        video_len_tem = video_len.unsqueeze(-1)

        mask = range_t < (
            1 + video_len_tem
        )

        mask = mask.repeat_interleave(
            n,
            dim=0
        )
        # video_len = video_len.unsqueeze(-1)
        # mask = range_t < (1 + video_len)
        # mask = mask.repeat_interleave(n, dim=0)
        
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



        #semantic feature的方法
        # semantic_global = semantic_tokens.mean(dim=1)  # (B,T,1)
        semantic_score = self.semantic_pool(semantic_tokens)
        semantic_weight = F.softmax(semantic_score, dim=1)
        # semantic_global = (semantic_weight * semantic_tokens).sum(dim=1) #0526
        token_global = (semantic_weight * semantic_tokens).sum(dim=1)

        temporal_semantic_score = self.temporal_semantic_pool(
            semantic_feature
        )
        temporal_mask = torch.arange(
            semantic_feature.shape[1],
            device=semantic_feature.device
        ).unsqueeze(0) < video_len.unsqueeze(1)
        temporal_semantic_score = temporal_semantic_score.masked_fill(
            ~temporal_mask.unsqueeze(-1),
            torch.finfo(temporal_semantic_score.dtype).min
        )
        temporal_semantic_weight = F.softmax(
            temporal_semantic_score,
            dim=1
        )
        temporal_global = (
            temporal_semantic_weight * semantic_feature
        ).sum(dim=1)

        semantic_global_alpha = self.semantic_global_gate(
            torch.cat([token_global, temporal_global], dim=-1)
        )
        semantic_global = (
            semantic_global_alpha * token_global
            +
            (1 - semantic_global_alpha) * temporal_global
        )
    
         ##semantic attends shuttle
        x_combined = torch.cat([
            x_main_features,
            semantic_global
        ], dim=1)
    
        x_output = self.mlp_head(x_combined)
        
        return x_output

class TemPose_TFwithtwostageOneStrokeTA(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output  
        self.mlp_head = MLP_Head(d_model * 2, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head = ResidualMLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.Jnb_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )

        self.shuttle_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )
        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

    
        self.stroke_emb = Stroke_EmbeddingTA(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        self.semantic_proj = nn.Linear(hid_size, d_model)
        self.semantic_pool = nn.Linear(d_model, 1) 
        self.semantic_type_embedding = nn.Parameter(torch.zeros(1, 1, d_model)) #0527
        self.semantic_norm = nn.LayerNorm(d_model) #0527
        self.temporal_semantic_pool = nn.Linear(d_model, 1) #0527
        self.semantic_global_gate = nn.Sequential(  #0527
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)
        nn.init.normal_(self.semantic_type_embedding, std=0.02) #0527

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, JnB: Tensor, pos: Tensor, shuttle: Tensor, video_len: Tensor, pre_next_stroke: Tensor):
       
        

        # Semantic Embedding
        # pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
        pre_emb = self.stroke_emb(pre_next_stroke)
        pre_emb = self.stroke_dropout(self.semantic_proj(pre_emb))
        # next_emb = self.stroke_dropout(self.semantic_proj(next_emb))


        #shuttle attends semantic
        semantic_tokens = pre_emb.unsqueeze(1)
        semantic_tokens = self.semantic_norm(
            semantic_tokens + self.semantic_type_embedding
        )
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
        
        # Cross Attention
        T_shuttle = shuttle.shape[1]

        range_t = torch.arange(
            T_shuttle,
            device=shuttle.device
        ).unsqueeze(0)

      


        # shuttle attends semantic
        semantic_mask = None

        shuttle_attn = self.cross_trans(
            shuttle,            # Q
            semantic_tokens,    # KV
            semantic_mask
        )

      
        shuttle_alpha = self.shuttle_gate(
            torch.cat([shuttle, shuttle_attn], dim=-1)
        )
        shuttle = shuttle + shuttle_alpha * shuttle_attn
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
        JnB_top = x[:, 0]       # (B,T,D)
        JnB_bottom = x[:, 1]    # (B,T,D)

        semantic_context_top = self.cross_trans(
            JnB_top,          # Q
            semantic_tokens,  # KV
            None
        )

        semantic_context_bottom = self.cross_trans(
            JnB_bottom,
            semantic_tokens,
            None
        )
        semantic_feature = shuttle_attn

        
        #shuttle attends semantic
        gate_input_top = torch.cat(
        [JnB_top, semantic_context_top],
        dim=-1
        )

        gate_input_bottom = torch.cat(
            [JnB_bottom, semantic_context_bottom],
            dim=-1
        )
        
      

        alpha = self.Jnb_gate(gate_input_top)      # (B,T,1)
        beta  = self.Jnb_gate(gate_input_bottom)   # (B,T,1)
        # gamma = gate_weight[..., 2].unsqueeze(-1)

        # x = (
        #     alpha * semantic_context
        #     +
        #     (1 - alpha) * x
        # )
        JnB_top = JnB_top + alpha * semantic_context_top

        JnB_bottom = JnB_bottom + beta * semantic_context_bottom
        x = torch.stack(
            [JnB_top, JnB_bottom],
            dim=1
        )
        # Step 5: Concatenate additional features with original TCN features
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
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
        video_len_tem = video_len.unsqueeze(-1)

        mask = range_t < (
            1 + video_len_tem
        )

        mask = mask.repeat_interleave(
            n,
            dim=0
        )
        # video_len = video_len.unsqueeze(-1)
        # mask = range_t < (1 + video_len)
        # mask = mask.repeat_interleave(n, dim=0)
        
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



        #semantic feature的方法
        # semantic_global = semantic_tokens.mean(dim=1)  # (B,T,1)
        semantic_score = self.semantic_pool(semantic_tokens)
        semantic_weight = F.softmax(semantic_score, dim=1)
        # semantic_global = (semantic_weight * semantic_tokens).sum(dim=1) #0526
        token_global = (semantic_weight * semantic_tokens).sum(dim=1)

        temporal_semantic_score = self.temporal_semantic_pool(
            semantic_feature
        )
        temporal_mask = torch.arange(
            semantic_feature.shape[1],
            device=semantic_feature.device
        ).unsqueeze(0) < video_len.unsqueeze(1)
        temporal_semantic_score = temporal_semantic_score.masked_fill(
            ~temporal_mask.unsqueeze(-1),
            torch.finfo(temporal_semantic_score.dtype).min
        )
        temporal_semantic_weight = F.softmax(
            temporal_semantic_score,
            dim=1
        )
        temporal_global = (
            temporal_semantic_weight * semantic_feature
        ).sum(dim=1)

        semantic_global_alpha = self.semantic_global_gate(
            torch.cat([token_global, temporal_global], dim=-1)
        )
        semantic_global = (
            semantic_global_alpha * token_global
            +
            (1 - semantic_global_alpha) * temporal_global
        )
    
         ##semantic attends shuttle
        x_combined = torch.cat([
            x_main_features,
            semantic_global
        ], dim=1)
    
        x_output = self.mlp_head(x_combined)
        
        return x_output

class TemPose_TFwithtwostageTheStrokeTA(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output  
        self.mlp_head = MLP_Head(d_model * 2, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head = ResidualMLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.Jnb_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )

        self.shuttle_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )
        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

    
        self.stroke_emb = Stroke_EmbeddingTA(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        self.semantic_proj = nn.Linear(hid_size, d_model)
        self.semantic_pool = nn.Linear(d_model, 1) 
        self.semantic_type_embedding = nn.Parameter(torch.zeros(1, 1, d_model)) #0527
        self.semantic_norm = nn.LayerNorm(d_model) #0527
        self.temporal_semantic_pool = nn.Linear(d_model, 1) #0527
        self.semantic_global_gate = nn.Sequential(  #0527
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)
        nn.init.normal_(self.semantic_type_embedding, std=0.02) #0527

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, JnB: Tensor, pos: Tensor, shuttle: Tensor, video_len: Tensor, the_stroke: Tensor):
       
        

        # Semantic Embedding
        # pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
        the_emb = self.stroke_emb(the_stroke)
        the_emb = self.stroke_dropout(self.semantic_proj(the_emb))
        # next_emb = self.stroke_dropout(self.semantic_proj(next_emb))


        #shuttle attends semantic
        semantic_tokens = the_emb.unsqueeze(1)
        semantic_tokens = self.semantic_norm(
            semantic_tokens + self.semantic_type_embedding
        )
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
        
        # Cross Attention
        T_shuttle = shuttle.shape[1]

        range_t = torch.arange(
            T_shuttle,
            device=shuttle.device
        ).unsqueeze(0)

      


        # shuttle attends semantic
        semantic_mask = None

        shuttle_attn = self.cross_trans(
            shuttle,            # Q
            semantic_tokens,    # KV
            semantic_mask
        )

      
        shuttle_alpha = self.shuttle_gate(
            torch.cat([shuttle, shuttle_attn], dim=-1)
        )
        shuttle = shuttle + shuttle_alpha * shuttle_attn
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
        JnB_top = x[:, 0]       # (B,T,D)
        JnB_bottom = x[:, 1]    # (B,T,D)

        semantic_context_top = self.cross_trans(
            JnB_top,          # Q
            semantic_tokens,  # KV
            None
        )

        semantic_context_bottom = self.cross_trans(
            JnB_bottom,
            semantic_tokens,
            None
        )
        semantic_feature = shuttle_attn

        
        #shuttle attends semantic
        gate_input_top = torch.cat(
        [JnB_top, semantic_context_top],
        dim=-1
        )

        gate_input_bottom = torch.cat(
            [JnB_bottom, semantic_context_bottom],
            dim=-1
        )
        
      

        alpha = self.Jnb_gate(gate_input_top)      # (B,T,1)
        beta  = self.Jnb_gate(gate_input_bottom)   # (B,T,1)
        # gamma = gate_weight[..., 2].unsqueeze(-1)

        # x = (
        #     alpha * semantic_context
        #     +
        #     (1 - alpha) * x
        # )
        JnB_top = JnB_top + alpha * semantic_context_top

        JnB_bottom = JnB_bottom + beta * semantic_context_bottom
        x = torch.stack(
            [JnB_top, JnB_bottom],
            dim=1
        )
        # Step 5: Concatenate additional features with original TCN features
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
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
        video_len_tem = video_len.unsqueeze(-1)

        mask = range_t < (
            1 + video_len_tem
        )

        mask = mask.repeat_interleave(
            n,
            dim=0
        )
        # video_len = video_len.unsqueeze(-1)
        # mask = range_t < (1 + video_len)
        # mask = mask.repeat_interleave(n, dim=0)
        
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



        #semantic feature的方法
        # semantic_global = semantic_tokens.mean(dim=1)  # (B,T,1)
        semantic_score = self.semantic_pool(semantic_tokens)
        semantic_weight = F.softmax(semantic_score, dim=1)
        # semantic_global = (semantic_weight * semantic_tokens).sum(dim=1) #0526
        token_global = (semantic_weight * semantic_tokens).sum(dim=1)

        temporal_semantic_score = self.temporal_semantic_pool(
            semantic_feature
        )
        temporal_mask = torch.arange(
            semantic_feature.shape[1],
            device=semantic_feature.device
        ).unsqueeze(0) < video_len.unsqueeze(1)
        temporal_semantic_score = temporal_semantic_score.masked_fill(
            ~temporal_mask.unsqueeze(-1),
            torch.finfo(temporal_semantic_score.dtype).min
        )
        temporal_semantic_weight = F.softmax(
            temporal_semantic_score,
            dim=1
        )
        temporal_global = (
            temporal_semantic_weight * semantic_feature
        ).sum(dim=1)

        semantic_global_alpha = self.semantic_global_gate(
            torch.cat([token_global, temporal_global], dim=-1)
        )
        semantic_global = (
            semantic_global_alpha * token_global
            +
            (1 - semantic_global_alpha) * temporal_global
        )
    
         ##semantic attends shuttle
        x_combined = torch.cat([
            x_main_features,
            semantic_global
        ], dim=1)
    
        x_output = self.mlp_head(x_combined)
        
        return x_output

class TemPose_TFwithtwostageTwoStrokeTA(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output  
        self.mlp_head = MLP_Head(d_model * 2, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head = ResidualMLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.Jnb_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )

        self.shuttle_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )
        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        self.stroke_emb = Stroke_EmbeddingTA(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        self.semantic_proj = nn.Linear(hid_size, d_model)
        self.semantic_pool = nn.Linear(d_model, 1) 
        self.semantic_type_embedding = nn.Parameter(torch.zeros(1, 2, d_model)) #0527
        self.semantic_norm = nn.LayerNorm(d_model) #0527
        self.temporal_semantic_pool = nn.Linear(d_model, 1) #0527
        self.semantic_global_gate = nn.Sequential(  #0527
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)
        nn.init.normal_(self.semantic_type_embedding, std=0.02) #0527

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, JnB: Tensor, pos: Tensor, shuttle: Tensor, video_len: Tensor, pre_stroke: Tensor, the_stroke: Tensor):
       

        # Semantic Embedding
        # pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
        pre_emb = self.stroke_emb(pre_stroke)
        the_emb = self.stroke_emb(the_stroke)
        pre_emb = self.stroke_dropout(self.semantic_proj(pre_emb))
        the_emb = self.stroke_dropout(self.semantic_proj(the_emb))


        #shuttle attends semantic
        semantic_tokens = torch.stack(
            [pre_emb, the_emb],
            dim=1
        )
        semantic_tokens = self.semantic_norm(
            semantic_tokens + self.semantic_type_embedding
        )
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
        
        # Cross Attention
        T_shuttle = shuttle.shape[1]

        range_t = torch.arange(
            T_shuttle,
            device=shuttle.device
        ).unsqueeze(0)

      


        # shuttle attends semantic
        semantic_mask = None

        shuttle_attn = self.cross_trans(
            shuttle,            # Q
            semantic_tokens,    # KV
            semantic_mask
        )

      
        shuttle_alpha = self.shuttle_gate(
            torch.cat([shuttle, shuttle_attn], dim=-1)
        )
        shuttle = shuttle + shuttle_alpha * shuttle_attn
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
        JnB_top = x[:, 0]       # (B,T,D)
        JnB_bottom = x[:, 1]    # (B,T,D)

        semantic_context_top = self.cross_trans(
            JnB_top,          # Q
            semantic_tokens,  # KV
            None
        )

        semantic_context_bottom = self.cross_trans(
            JnB_bottom,
            semantic_tokens,
            None
        )
        semantic_feature = shuttle_attn

        
        #shuttle attends semantic
        gate_input_top = torch.cat(
        [JnB_top, semantic_context_top],
        dim=-1
        )

        gate_input_bottom = torch.cat(
            [JnB_bottom, semantic_context_bottom],
            dim=-1
        )
        
      

        alpha = self.Jnb_gate(gate_input_top)      # (B,T,1)
        beta  = self.Jnb_gate(gate_input_bottom)   # (B,T,1)
        # gamma = gate_weight[..., 2].unsqueeze(-1)

        # x = (
        #     alpha * semantic_context
        #     +
        #     (1 - alpha) * x
        # )
        JnB_top = JnB_top + alpha * semantic_context_top

        JnB_bottom = JnB_bottom + beta * semantic_context_bottom
        x = torch.stack(
            [JnB_top, JnB_bottom],
            dim=1
        )
        # Step 5: Concatenate additional features with original TCN features
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
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
        video_len_tem = video_len.unsqueeze(-1)

        mask = range_t < (
            1 + video_len_tem
        )

        mask = mask.repeat_interleave(
            n,
            dim=0
        )
        # video_len = video_len.unsqueeze(-1)
        # mask = range_t < (1 + video_len)
        # mask = mask.repeat_interleave(n, dim=0)
        
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



        #semantic feature的方法
        # semantic_global = semantic_tokens.mean(dim=1)  # (B,T,1)
        semantic_score = self.semantic_pool(semantic_tokens)
        semantic_weight = F.softmax(semantic_score, dim=1)
        # semantic_global = (semantic_weight * semantic_tokens).sum(dim=1) #0526
        token_global = (semantic_weight * semantic_tokens).sum(dim=1)

        temporal_semantic_score = self.temporal_semantic_pool(
            semantic_feature
        )
        temporal_mask = torch.arange(
            semantic_feature.shape[1],
            device=semantic_feature.device
        ).unsqueeze(0) < video_len.unsqueeze(1)
        temporal_semantic_score = temporal_semantic_score.masked_fill(
            ~temporal_mask.unsqueeze(-1),
            torch.finfo(temporal_semantic_score.dtype).min
        )
        temporal_semantic_weight = F.softmax(
            temporal_semantic_score,
            dim=1
        )
        temporal_global = (
            temporal_semantic_weight * semantic_feature
        ).sum(dim=1)

        semantic_global_alpha = self.semantic_global_gate(
            torch.cat([token_global, temporal_global], dim=-1)
        )
        semantic_global = (
            semantic_global_alpha * token_global
            +
            (1 - semantic_global_alpha) * temporal_global
        )
    
         ##semantic attends shuttle
        x_combined = torch.cat([
            x_main_features,
            semantic_global
        ], dim=1)
    
        x_output = self.mlp_head(x_combined)
        
        return x_output

class TemPose_TFwithtwostageNextStrokeTA(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output  
        self.mlp_head = MLP_Head(d_model * 2, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head = ResidualMLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.Jnb_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )

        self.shuttle_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )
        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

    
        self.stroke_emb = Stroke_EmbeddingTA(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        self.semantic_proj = nn.Linear(hid_size, d_model)
        self.semantic_pool = nn.Linear(d_model, 1) 
        self.semantic_type_embedding = nn.Parameter(torch.zeros(1, 1, d_model)) #0527
        self.semantic_norm = nn.LayerNorm(d_model) #0527
        self.temporal_semantic_pool = nn.Linear(d_model, 1) #0527
        self.semantic_global_gate = nn.Sequential(  #0527
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)
        nn.init.normal_(self.semantic_type_embedding, std=0.02) #0527

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, JnB: Tensor, pos: Tensor, shuttle: Tensor, video_len: Tensor, next_stroke: Tensor):
       
        

        # Semantic Embedding
        # pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
        next_emb = self.stroke_emb(next_stroke)
        next_emb = self.stroke_dropout(self.semantic_proj(next_emb))
        # next_emb = self.stroke_dropout(self.semantic_proj(next_emb))


        #shuttle attends semantic
        semantic_tokens = next_emb.unsqueeze(1)
        semantic_tokens = self.semantic_norm(
            semantic_tokens + self.semantic_type_embedding
        )
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
        
        # Cross Attention
        T_shuttle = shuttle.shape[1]

        range_t = torch.arange(
            T_shuttle,
            device=shuttle.device
        ).unsqueeze(0)

      


        # shuttle attends semantic
        semantic_mask = None

        shuttle_attn = self.cross_trans(
            shuttle,            # Q
            semantic_tokens,    # KV
            semantic_mask
        )

      
        shuttle_alpha = self.shuttle_gate(
            torch.cat([shuttle, shuttle_attn], dim=-1)
        )
        shuttle = shuttle + shuttle_alpha * shuttle_attn
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
        JnB_top = x[:, 0]       # (B,T,D)
        JnB_bottom = x[:, 1]    # (B,T,D)

        semantic_context_top = self.cross_trans(
            JnB_top,          # Q
            semantic_tokens,  # KV
            None
        )

        semantic_context_bottom = self.cross_trans(
            JnB_bottom,
            semantic_tokens,
            None
        )
        semantic_feature = shuttle_attn

        
        #shuttle attends semantic
        gate_input_top = torch.cat(
        [JnB_top, semantic_context_top],
        dim=-1
        )

        gate_input_bottom = torch.cat(
            [JnB_bottom, semantic_context_bottom],
            dim=-1
        )
        
      

        alpha = self.Jnb_gate(gate_input_top)      # (B,T,1)
        beta  = self.Jnb_gate(gate_input_bottom)   # (B,T,1)
        # gamma = gate_weight[..., 2].unsqueeze(-1)

        # x = (
        #     alpha * semantic_context
        #     +
        #     (1 - alpha) * x
        # )
        JnB_top = JnB_top + alpha * semantic_context_top

        JnB_bottom = JnB_bottom + beta * semantic_context_bottom
        x = torch.stack(
            [JnB_top, JnB_bottom],
            dim=1
        )
        # Step 5: Concatenate additional features with original TCN features
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
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
        video_len_tem = video_len.unsqueeze(-1)

        mask = range_t < (
            1 + video_len_tem
        )

        mask = mask.repeat_interleave(
            n,
            dim=0
        )
        # video_len = video_len.unsqueeze(-1)
        # mask = range_t < (1 + video_len)
        # mask = mask.repeat_interleave(n, dim=0)
        
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



        #semantic feature的方法
        # semantic_global = semantic_tokens.mean(dim=1)  # (B,T,1)
        semantic_score = self.semantic_pool(semantic_tokens)
        semantic_weight = F.softmax(semantic_score, dim=1)
        # semantic_global = (semantic_weight * semantic_tokens).sum(dim=1) #0526
        token_global = (semantic_weight * semantic_tokens).sum(dim=1)

        temporal_semantic_score = self.temporal_semantic_pool(
            semantic_feature
        )
        temporal_mask = torch.arange(
            semantic_feature.shape[1],
            device=semantic_feature.device
        ).unsqueeze(0) < video_len.unsqueeze(1)
        temporal_semantic_score = temporal_semantic_score.masked_fill(
            ~temporal_mask.unsqueeze(-1),
            torch.finfo(temporal_semantic_score.dtype).min
        )
        temporal_semantic_weight = F.softmax(
            temporal_semantic_score,
            dim=1
        )
        temporal_global = (
            temporal_semantic_weight * semantic_feature
        ).sum(dim=1)

        semantic_global_alpha = self.semantic_global_gate(
            torch.cat([token_global, temporal_global], dim=-1)
        )
        semantic_global = (
            semantic_global_alpha * token_global
            +
            (1 - semantic_global_alpha) * temporal_global
        )
    
         ##semantic attends shuttle
        x_combined = torch.cat([
            x_main_features,
            semantic_global
        ], dim=1)
    
        x_output = self.mlp_head(x_combined)
        
        return x_output

class TemPose_TFwithtwostageThreeStrokeTA(nn.Module):
    def __init__(
        self, in_dim, seq_len, n_class=31, n_people=2,
        d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=2,
        drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5,
        type_size=31, emb_size=128, hid_size=256 
    ):
        super().__init__()
        
        if n_people > 2:
            raise NotImplementedError
        
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

        # MLP Head for output  
        self.mlp_head = MLP_Head(d_model * 2, n_class, d_model * mlp_d_scale, drop_p) 
        # self.mlp_head = ResidualMLP_Head(d_model + hid_size, n_class, d_model * mlp_d_scale, drop_p)

        # Stroke fusion model
        self.Jnb_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )

        self.shuttle_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model , 1),
            nn.Sigmoid()
        )
        # CrossTransformerLayer
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)

        self.stroke_emb = Stroke_EmbeddingTA(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_ca = Stroke_TA(type_size=type_size, emb_size=emb_size, hid_size=hid_size)
        self.stroke_dropout = nn.Dropout(drop_p, inplace=True)
        self.semantic_proj = nn.Linear(hid_size, d_model)
        self.semantic_pool = nn.Linear(d_model, 1) 
        self.semantic_type_embedding = nn.Parameter(torch.zeros(1, 3, d_model)) #0527
        self.semantic_norm = nn.LayerNorm(d_model) #0527
        self.temporal_semantic_pool = nn.Linear(d_model, 1) #0527
        self.semantic_global_gate = nn.Sequential(  #0527
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.d_model = d_model
        self.init_weights()
        # for param in self.stroke_fusion_model.parameters():
        #     param.requires_grad = False
            
    @torch.no_grad()
    def init_weights(self):
        p_enc_1d_model = PositionalEncoding1D(self.d_model)
        
        pos_encoding: Tensor = p_enc_1d_model(self.embedding_tem.squeeze(0))
        self.embedding_tem.copy_(pos_encoding.unsqueeze(0))

        pos_encoding: Tensor = p_enc_1d_model(self.embedding_inter)
        self.embedding_inter.copy_(pos_encoding)

        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)
        nn.init.normal_(self.semantic_type_embedding, std=0.02) #0527

        self.apply(self.init_weights_recursive)

    def init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)

    def forward(self, JnB: Tensor, pos: Tensor, shuttle: Tensor, video_len: Tensor, pre_next_stroke: Tensor, the_stroke: Tensor):
       

        # Semantic Embedding
        # pre_next_feature = self.stroke_fusion_model(pre_next_stroke)
        pre_emb, next_emb = self.stroke_ca(pre_next_stroke)
        the_emb = self.stroke_emb(the_stroke)
        pre_emb = self.stroke_dropout(self.semantic_proj(pre_emb))
        next_emb = self.stroke_dropout(self.semantic_proj(next_emb))
        the_emb = self.stroke_dropout(self.semantic_proj(the_emb))


        #shuttle attends semantic
        semantic_tokens = torch.stack(
            [pre_emb, the_emb, next_emb],
            dim=1
        )
        semantic_tokens = self.semantic_norm(
            semantic_tokens + self.semantic_type_embedding
        )
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
        
        # Cross Attention
        T_shuttle = shuttle.shape[1]

        range_t = torch.arange(
            T_shuttle,
            device=shuttle.device
        ).unsqueeze(0)

      


        # shuttle attends semantic
        semantic_mask = None

        shuttle_attn = self.cross_trans(
            shuttle,            # Q
            semantic_tokens,    # KV
            semantic_mask
        )

      
        shuttle_alpha = self.shuttle_gate(
            torch.cat([shuttle, shuttle_attn], dim=-1)
        )
        shuttle = shuttle + shuttle_alpha * shuttle_attn
        JnB = JnB.transpose(1, 2).contiguous()  # (b, n, t, input_dim)
        x = self.project(JnB)  # Project to the desired model dimension
        b, n, t, d = x.shape  # b: batch_size, n: num_people, t: sequence_length, d: feature_dim
        JnB_top = x[:, 0]       # (B,T,D)
        JnB_bottom = x[:, 1]    # (B,T,D)

        semantic_context_top = self.cross_trans(
            JnB_top,          # Q
            semantic_tokens,  # KV
            None
        )

        semantic_context_bottom = self.cross_trans(
            JnB_bottom,
            semantic_tokens,
            None
        )
        semantic_feature = shuttle_attn

        
        #shuttle attends semantic
        gate_input_top = torch.cat(
        [JnB_top, semantic_context_top],
        dim=-1
        )

        gate_input_bottom = torch.cat(
            [JnB_bottom, semantic_context_bottom],
            dim=-1
        )
        
      

        alpha = self.Jnb_gate(gate_input_top)      # (B,T,1)
        beta  = self.Jnb_gate(gate_input_bottom)   # (B,T,1)
        # gamma = gate_weight[..., 2].unsqueeze(-1)

        # x = (
        #     alpha * semantic_context
        #     +
        #     (1 - alpha) * x
        # )
        JnB_top = JnB_top + alpha * semantic_context_top

        JnB_bottom = JnB_bottom + beta * semantic_context_bottom
        x = torch.stack(
            [JnB_top, JnB_bottom],
            dim=1
        )
        # Step 5: Concatenate additional features with original TCN features
        x_additional = torch.stack((pos_top, pos_bottom, shuttle), dim=1)
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
        video_len_tem = video_len.unsqueeze(-1)

        mask = range_t < (
            1 + video_len_tem
        )

        mask = mask.repeat_interleave(
            n,
            dim=0
        )
        # video_len = video_len.unsqueeze(-1)
        # mask = range_t < (1 + video_len)
        # mask = mask.repeat_interleave(n, dim=0)
        
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



        #semantic feature的方法
        # semantic_global = semantic_tokens.mean(dim=1)  # (B,T,1)
        semantic_score = self.semantic_pool(semantic_tokens)
        semantic_weight = F.softmax(semantic_score, dim=1)
        # semantic_global = (semantic_weight * semantic_tokens).sum(dim=1) #0526
        token_global = (semantic_weight * semantic_tokens).sum(dim=1)

        temporal_semantic_score = self.temporal_semantic_pool(
            semantic_feature
        )
        temporal_mask = torch.arange(
            semantic_feature.shape[1],
            device=semantic_feature.device
        ).unsqueeze(0) < video_len.unsqueeze(1)
        temporal_semantic_score = temporal_semantic_score.masked_fill(
            ~temporal_mask.unsqueeze(-1),
            torch.finfo(temporal_semantic_score.dtype).min
        )
        temporal_semantic_weight = F.softmax(
            temporal_semantic_score,
            dim=1
        )
        temporal_global = (
            temporal_semantic_weight * semantic_feature
        ).sum(dim=1)

        semantic_global_alpha = self.semantic_global_gate(
            torch.cat([token_global, temporal_global], dim=-1)
        )
        semantic_global = (
            semantic_global_alpha * token_global
            +
            (1 - semantic_global_alpha) * temporal_global
        )
    
         ##semantic attends shuttle
        x_combined = torch.cat([
            x_main_features,
            semantic_global
        ], dim=1)
    
        x_output = self.mlp_head(x_combined)
        
        return x_output


if __name__ == '__main__':
    n_features = (17 + 19 * 2) * 2
    pose = torch.randn((1, 30, 2, n_features), dtype=torch.float)
    pos = torch.randn((1, 30, 2, 2), dtype=torch.float)
    shuttle = torch.randn((1, 30, 2), dtype=torch.float)
    videos_len = torch.tensor([30])
    label = torch.tensor([1])
    # pre_next_stroke: 每列是前一拍和后一拍的类别，形状为 (B, 2)
    pre_next_stroke = torch.randint(0, 31, (1, 2), dtype=torch.long)  # (B, 2)
    pre_stroke = torch.randint(0, 31, (1, 1), dtype=torch.long)  # (B, 1)
    the_stroke = torch.randint(0, 31, (1, 1), dtype=torch.long)  # (B, 1)
    next_stroke = torch.randint(0, 31, (1, 1), dtype=torch.long)  # (B, 1)
    seq_len = 30
    # input_data = [pose, pos, shuttle, videos_len]
    # model = TemPose_TF(
    #     in_dim=n_features,
    #     seq_len=30,
    #     n_class=31,
    #     d_model=100
    # )
    # input_data = [pose, videos_len]
    # model = TemPose_V(
    #     in_dim=n_features,
    #     seq_len=30,
    #     n_class=31,
    #     d_model=100
    # )
    # input_data = [pose, videos_len, pre_next_stroke]

    
    # model = TemPose_VwithStrokeFusion(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )


    # input_data = [pose, pos, shuttle, videos_len, pre_next_stroke]
    # model = TemPose_TFwithtwoStageStrokeFusion(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )

    # input_data = [pose, pos, shuttle, videos_len, pre_next_stroke]
    # model = TemPose_TFwithtwostageStrokeTA(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )
    

    # input_data = [pose, pos, shuttle, videos_len, pre_stroke]
    # model = TemPose_TFwithOneStroke(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )

    # input_data = [pose, pos, shuttle, videos_len, pre_stroke]
    # model = TemPose_TFwithtwostageOneStrokeTA(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )
    
    # input_data = [pose, pos, shuttle, videos_len, the_stroke]
    # model = TemPose_TFwithTheStroke(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )

    # input_data = [pose, pos, shuttle, videos_len, the_stroke]
    # model = TemPose_TFwithtwostageTheStrokeTA(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )

    # input_data = [pose, pos, shuttle, videos_len, next_stroke]
    # model = TemPose_TFwithNextStroke(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )

    # input_data = [pose, pos, shuttle, videos_len, next_stroke]
    # model = TemPose_TFwithtwostageNextStrokeTA(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )

    # input_data = [pose, pos, shuttle, videos_len, pre_stroke, the_stroke]
    # model = TemPose_TFwithTwoStroke(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )

    # input_data = [pose, pos, shuttle, videos_len, pre_stroke, the_stroke]
    # model = TemPose_TFwithtwostageTwoStrokeTA(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )

    # input_data = [pose, pos, shuttle, videos_len, pre_next_stroke, the_stroke]
    # model = TemPose_TFwithThreeStroke(
    #     in_dim=n_features,
    #     seq_len=seq_len,
    #     n_class=31,
    #     d_model=100
    # )

    input_data = [pose, pos, shuttle, videos_len, pre_next_stroke, the_stroke]
    model = TemPose_TFwithtwostageThreeStrokeTA(
        in_dim=n_features,
        seq_len=seq_len,
        n_class=31,
        d_model=100
    )

   


    summary(model, input_data=input_data, depth=4, device='cpu')
   
    # output = model(*input_data)  
    # print("Output shape:", output.shape)
    # print("Output:", output)

    # predicted_class = torch.argmax(output, dim=1)  
    # print("Predicted Class:", predicted_class)
