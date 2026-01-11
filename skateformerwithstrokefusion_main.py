import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections import namedtuple
import time
from datetime import timedelta

from dataset import prepare_npy_collated_loaders_for_skateformer, RandomTranslation_batch, \
                    Dataset_npy
from skateformer import SkateFormer, SkateFormerwithStrokeFusion
from utils import show_f1_results, plot_confusion_matrix


Hyp = namedtuple('Hyp', [
    'n_epochs', 'batch_size', 'lr',
    'n_classes', 'early_stop_n_epochs',
    'T', 'V', 'N', 'L'
])
hyp = Hyp(
    n_epochs=1600,
    early_stop_n_epochs=300,
    batch_size=128,
    lr=1e-3,
    n_classes=31,
    T=32,
    N=4,
    V=25,
    L=5
)


def reset_joints_order_before_partitions(x: Tensor):
    # x: (..., V, C)
    head = np.arange(5)
    left_hand = np.array([9, 7, 5, 11, 6])
    right_hand = np.array([5, 6, 12, 8, 10])
    left_leg = np.array([5, 11, 12, 13, 15])
    right_leg = np.array([11, 6, 12, 14, 16])

    new_idx = torch.from_numpy(np.concatenate(
        [head, left_hand, right_hand, left_leg, right_leg]
    )).to(x.device)
    new_x = torch.index_select(x, dim=-2, index=new_idx)
    return new_x  # (..., V', C)


def pad_to_T(x: Tensor):
    # x: (B, S, m, V, C)
    S = x.shape[1]; T = hyp.T
    assert S <= T, 'T is too short.'
    pad_len = T - S
    new_x = F.pad(x, [0, 0, 0, 0, 0, 0, 0, pad_len])
    return new_x


def train_one_epoch(
    model: nn.Module,
    loader,
    random_shift_fn,
    loss_fn,
    optimizer: optim.Optimizer,
    device
):
    model.train()
    total_loss = 0.0
    special_classes = torch.tensor([0, 14, 15, 29, 30], device=device)


    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        labels: Tensor = labels.to(device)
        pre_next_stroke: Tensor = pre_next_stroke.to(device)
        human_pose = random_shift_fn(human_pose)

        human_pose = reset_joints_order_before_partitions(human_pose)
        human_pose = pad_to_T(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)

        pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
        logits_stage1 = model(human_pose, pre_next_stroke_zero, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
        pred_stage1 = torch.argmax(logits_stage1, dim=1)

        # ---------- 建立 pre_next_stroke_pair ----------
        first_prev = pre_next_stroke[0, 0].unsqueeze(0).detach().clone()
        last_next = pre_next_stroke[-1, 1].unsqueeze(0).detach().clone()

        if batch_idx == 0:
            pre_next_stroke_pair = torch.cat([
                torch.tensor([0], device=device),
                pred_stage1,
                last_next
            ])
        else:
            pre_next_stroke_pair = torch.cat([
                first_prev,
                pred_stage1,
                last_next
            ])

        pre_next_stroke_pair = torch.stack(
            [pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1
        )

        # ---------- 修正特殊類別 ----------
        mask_current = torch.isin(pred_stage1, special_classes)
        pre_next_stroke_pair[mask_current, 0] = 0

        mask_next = torch.isin(pre_next_stroke_pair[:, 1], special_classes)
        pre_next_stroke_pair[mask_next, 1] = 0

      

        # ---------- Stage 2 ----------
        optimizer.zero_grad()
        logits_stage2 = model(human_pose, pre_next_stroke_pair, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
        loss = loss_fn(logits_stage2, labels)
        # logits = model(human_pose, pre_next_stroke, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
        # loss: Tensor = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    train_loss = total_loss / len(loader)
    return train_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loss_fn,
    loader,
    device
):
    model.eval()
    total_loss = 0.0
    cum_tp = torch.zeros(hyp.n_classes)
    cum_tn = torch.zeros(hyp.n_classes)
    cum_fp = torch.zeros(hyp.n_classes)
    cum_fn = torch.zeros(hyp.n_classes)
    all_preds_stage1 = []  # 用來儲存所有 batch 的預測結果

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        labels: Tensor = labels.to(device)
        pre_next_stroke: Tensor = pre_next_stroke.to(device)
        human_pose = reset_joints_order_before_partitions(human_pose)
        human_pose = pad_to_T(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)

        pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
        logits_stage1 = model(human_pose,pre_next_stroke_zero, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
        pred_stage1 = torch.argmax(logits_stage1, dim=1)

        # 儲存每個 batch 的預測結果
        all_preds_stage1.append(pred_stage1.detach().clone())


    # 第二階段：使用所有第一階段的預測結果來組裝 pre_next_stroke_pair
    prev_last = torch.tensor([0], device=device)  # 用於記錄前一個 batch 最後的預測
    next_last = torch.tensor([all_preds_stage1[1][0].item()], device=device)  # 用於記錄下一個 batch 最後的預測
    total_loss = 0.0

    # 現在將所有的預測結果組合為 pre_next_stroke_pair 並進行第二階段推理
    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        pos: Tensor = pos.to(device)
        shuttle: Tensor = shuttle.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        pre_next_stroke: Tensor = pre_next_stroke.to(device)
        human_pose = reset_joints_order_before_partitions(human_pose)
        human_pose = pad_to_T(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)
        
        # ---------- 第二階段 ----------
        # 合併前一拍和下一拍的預測結果
        special_classes = torch.tensor([0, 14, 15, 29, 30], device=pred_stage1.device)

        # 使用第一階段的預測結果來創建 pre_next_stroke_pair
        if batch_idx == 0:
            # 第一個 batch 用 0 當前一拍和下一拍
            pre_next_stroke_pair = torch.cat([torch.tensor([0], device=device), all_preds_stage1[batch_idx], next_last.detach().clone()])
            pre_next_stroke_pair = torch.stack([pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1)

            # print(f"pre_next_stroke_pair : {pre_next_stroke_pair}")  # 第一個 batch
            # print(f"pre_next_stroke_pair shape: {pre_next_stroke_pair.shape}")  # 第一個 batch
        else:
            # 其他 batch 若有特殊類別 → 前一拍設為 0，否則用上一批的最後一個
            pre_next_stroke_pair = torch.cat([prev_last.detach().clone(), all_preds_stage1[batch_idx], next_last.detach().clone()])
            pre_next_stroke_pair = torch.stack([pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1)
        
        # 此 batch 的最後一拍的下一拍應該是下一個 batch 的第一拍
        if batch_idx + 1 < len(all_preds_stage1):
            next_last = all_preds_stage1[batch_idx + 1][0].detach().clone()

        # 修正特殊類別：如果當前拍是特殊類別，將前一拍設為 0
        pred_current = torch.isin(all_preds_stage1[batch_idx], special_classes)
        pre_next_stroke_pair[pred_current, 0] = 0  # 若當前拍是特殊類別，將前一拍設為 0

        mask_current = torch.isin(pre_next_stroke_pair[:, 1], special_classes)
        pre_next_stroke_pair[mask_current, 1] = 0

        # 修正下一拍：如果下一拍是特殊類別，將當前拍的下一拍設為 0
        
    

        # 更新 prev_last 和 next_last（記錄此 batch 最後一個預測）
        prev_last = all_preds_stage1[batch_idx][-1:].detach().clone()

        # 將 next_last 轉換為一維張量，以便能與其他張量拼接
        next_last = torch.tensor([next_last.item()], device=device)  # 確保它是可以拼接的

        # 將 pre_next_stroke_pair 重塑為 [batch_size, 2]，包含前一拍和下一拍
        pre_next_stroke_pair = pre_next_stroke_pair.view(-1, 2)
        # print(f"pre_next_stroke_pair 3 : {pre_next_stroke_pair.shape}") 

        # 模型推理
        logits_stage2 = model(human_pose, pre_next_stroke_pair, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))

        pred_stage2 = torch.argmax(logits_stage2, dim=1)

        # loss = loss_fn(logits_stage2, labels)
        # logits = model(human_pose, pre_next_stroke, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
        loss: Tensor = loss_fn(logits_stage2, labels)
        total_loss += loss.item()

        pred = F.one_hot(torch.argmax(logits_stage2, dim=1), hyp.n_classes).bool()
        labels_onehot = F.one_hot(labels, hyp.n_classes).bool()

        tp = torch.sum(pred & labels_onehot, dim=0)
        tn = torch.sum(~pred & ~labels_onehot, dim=0)

        fp = torch.sum(pred & ~labels_onehot, dim=0)
        fn = torch.sum(~pred & labels_onehot, dim=0)

        cum_tp += tp.cpu()
        cum_tn += tn.cpu()
        cum_fp += fp.cpu()
        cum_fn += fn.cpu()

    val_loss = total_loss / len(loader)

    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / (cum_tp + cum_fn)

    f1_score = 2 * precision * recall / (precision + recall)
    f1_score[f1_score.isnan()] = 0

    f1_score_avg = f1_score.mean()
    f1_score_min = f1_score.min()
    return val_loss, f1_score_avg, f1_score_min


@torch.no_grad()
def test(
    model: nn.Module,
    loader,
    device
):
    model.eval()
    pred_ls = []
    labels_ls = []
    misclassified_reports = []
    all_preds_stage1 = []  # 收集第一階段所有 batch 的預測結果

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        pre_next_stroke: Tensor = pre_next_stroke.to(device)
        human_pose = reset_joints_order_before_partitions(human_pose)
        human_pose = pad_to_T(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)
        # logits = model(human_pose, pre_next_stroke, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
        # pred = torch.argmax(logits, dim=1).cpu()
        
        # pred_ls.append(pred)
        # 第一階段輸入 pre_next_stroke 全為 0
        pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
        logits_stage1 = model(human_pose, pre_next_stroke_zero, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
        pred_stage1 = torch.argmax(logits_stage1, dim=1)

        all_preds_stage1.append(pred_stage1.detach().clone())

    # ========== 第二階段：跨 batch 使用預測結果 ==========
    prev_last = torch.tensor([0], device=device)  # 記錄上一批最後一拍
    next_last = torch.tensor([all_preds_stage1[1][0].item()] if len(all_preds_stage1) > 1 else [0], device=device)
    special_classes = torch.tensor([0, 14, 15, 29, 30], device=device)

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        pos: Tensor = pos.to(device)
        shuttle: Tensor = shuttle.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        human_pose = reset_joints_order_before_partitions(human_pose)
        human_pose = pad_to_T(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)

        # 構建 pre_next_stroke_pair
        if batch_idx == 0:
            pre_next_stroke_pair = torch.cat([
                torch.tensor([0], device=device),
                all_preds_stage1[batch_idx],
                next_last.detach().clone()
            ])
            pre_next_stroke_pair = torch.stack([pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1)
        else:
            pre_next_stroke_pair = torch.cat([
                prev_last.detach().clone(),
                all_preds_stage1[batch_idx],
                next_last.detach().clone()
            ])
            pre_next_stroke_pair = torch.stack([pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1)

        # 下一批的第一拍
        if batch_idx + 1 < len(all_preds_stage1):
            next_last = all_preds_stage1[batch_idx + 1][0].detach().clone()

        # 修正特殊類別：
        # 1️⃣ 當前拍是特殊類別 → 前一拍 = 0
        pred_current = torch.isin(all_preds_stage1[batch_idx], special_classes)
        pre_next_stroke_pair[pred_current, 0] = 0

        # 2️⃣ 下一拍是特殊類別 → 下一拍 = 0
        mask_next = torch.isin(pre_next_stroke_pair[:, 1], special_classes)
        pre_next_stroke_pair[mask_next, 1] = 0

        # 更新 prev_last / next_last
        prev_last = all_preds_stage1[batch_idx][-1:].detach().clone()
        next_last = torch.tensor([next_last.item()], device=device)

        # 模型第二階段推理
        logits_stage2 = model(human_pose, pre_next_stroke_pair, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
        pred_stage2 = torch.argmax(logits_stage2, dim=1).cpu()

        # ========= 錯誤紀錄 =========
        is_misclassified = (pred_stage2 != labels.cpu())
        if is_misclassified.any():
            misclassified_pred = pred_stage2[is_misclassified].tolist()
            misclassified_labels = labels.cpu()[is_misclassified].tolist()
            misclassified_pairs = list(zip(misclassified_labels, misclassified_pred))
            misclassified_reports.extend(misclassified_pairs)

        pred_ls.append(pred_stage2)
        labels_ls.append(labels.cpu())

    return torch.cat(pred_ls), torch.cat(labels_ls)


@torch.no_grad()
def test_topk(
    model: nn.Module,
    loader,
    device,
    k=2
):
    model.eval()
    pred_ls = []
    labels_ls = []
    all_preds_stage1 = []

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        pre_next_stroke: Tensor = pre_next_stroke.to(device)
        human_pose = reset_joints_order_before_partitions(human_pose)
        human_pose = pad_to_T(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)
        # logits = model(human_pose, pre_next_stroke, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))

        # _, pred = torch.topk(logits, k=k, dim=1)
        
        # pred_ls.append(pred.cpu())
        # labels_ls.append(labels)
        pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
        logits_stage1 = model(human_pose, pre_next_stroke_zero, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))
        pred_stage1 = torch.argmax(logits_stage1, dim=1)

        all_preds_stage1.append(pred_stage1.detach().clone())

    # ========== 第二階段：跨 batch 推理 ==========
    prev_last = torch.tensor([0], device=device)
    next_last = torch.tensor([all_preds_stage1[1][0].item()] if len(all_preds_stage1) > 1 else [0], device=device)
    special_classes = torch.tensor([0, 14, 15, 29, 30], device=device)

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        pos: Tensor = pos.to(device)
        shuttle: Tensor = shuttle.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        human_pose = reset_joints_order_before_partitions(human_pose)
        human_pose = pad_to_T(human_pose)

        # human_pose: (n, t, 2, v, c)
        human_pose = human_pose.permute(0, 4, 1, 3, 2).contiguous()
        # human_pose: (n, c, t, v, m=2)

        # ----------- 組 pre_next_stroke_pair -----------
        if batch_idx == 0:
            pre_next_stroke_pair = torch.cat([
                torch.tensor([0], device=device),
                all_preds_stage1[batch_idx],
                next_last.detach().clone()
            ])
            pre_next_stroke_pair = torch.stack([pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1)
        else:
            pre_next_stroke_pair = torch.cat([
                prev_last.detach().clone(),
                all_preds_stage1[batch_idx],
                next_last.detach().clone()
            ])
            pre_next_stroke_pair = torch.stack([pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1)

        # 下一批第一拍
        if batch_idx + 1 < len(all_preds_stage1):
            next_last = all_preds_stage1[batch_idx + 1][0].detach().clone()

        # ----------- 修正特殊類別 -----------
        # 當前拍是特殊類別 → pre = 0
        pred_current = torch.isin(all_preds_stage1[batch_idx], special_classes)
        pre_next_stroke_pair[pred_current, 0] = 0

        # 下一拍是特殊類別 → next = 0
        mask_next = torch.isin(pre_next_stroke_pair[:, 1], special_classes)
        pre_next_stroke_pair[mask_next, 1] = 0

        # 更新 prev_last / next_last
        prev_last = all_preds_stage1[batch_idx][-1:].detach().clone()
        next_last = torch.tensor([next_last.item()], device=device)

        # ----------- 第二階段推理 -----------
        logits_stage2 = model(human_pose, pre_next_stroke_pair, torch.arange(1, human_pose.shape[2]+1, device=human_pose.device))

        # top-k 預測（例如 top2）
        _, pred_topk = torch.topk(logits_stage2, k=k, dim=1)

        pred_ls.append(pred_topk.cpu())
        labels_ls.append(labels.cpu())

    return torch.cat(pred_ls), torch.cat(labels_ls)


def train_network(
    model: SkateFormer,
    train_loader,
    val_loader,
    device,
    save_path: Path,
):
    random_shift_fn = RandomTranslation_batch()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    para_group_dic = [
        {
            'params': [v for k, v in model.named_parameters() if k not in model.no_weight_decay()],
            'weight_decay': 5e-4  # original: 5e-4
        },
        {
            'params': [v for k, v in model.named_parameters() if k in model.no_weight_decay()],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(para_group_dic, lr=hyp.lr)

    best_value = 0.0
    early_stop_count = 0

    for epoch in range(1, hyp.n_epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            random_shift_fn=random_shift_fn,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        val_loss, f1_score_avg, f1_score_min = validate(
            model=model,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device
        )
        t1 = time.time()
        print(f'Epoch({epoch}/{hyp.n_epochs}): train_loss={train_loss:.3f}, '\
              f'val_loss={val_loss:.3f}, macro_f1={f1_score_avg:.3f}, min_f1={f1_score_min:.3f} '\
              f'- {t1 - t0:.2f} s')

        early_stop_count += 1
        if best_value < f1_score_avg:
            best_value = f1_score_avg
            best_state = deepcopy(model.state_dict())
            print(f'Picked! => Best value {f1_score_avg:.3f}')
            early_stop_count = 0

        if early_stop_count == hyp.early_stop_n_epochs:
            print(f'Early stop with best value {best_value:.3f}')
            break
    
    torch.save(best_state, str(save_path))
    model.load_state_dict(best_state)
    return model


class Task:
    def __init__(self) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'

    def prepare_dataloaders(
        self,
        root_dir: Path,
        pose_style='J_only',
        train_partial=1.0
    ):
        self.train_loader, \
        self.val_loader, \
        self.test_loader \
            = prepare_npy_collated_loaders_for_skateformer(
                root_dir=root_dir,
                pose_style=pose_style,
                batch_size=hyp.batch_size,
                use_cuda=self.use_cuda,
                num_workers=(0, 0, 0),
                train_partial=train_partial
            )
        self.pose_style = pose_style

    def get_network_architecture(self, model_name):
        '''
        `model_name`
        - 'SkateFormer_2' (2.38M)
        - 'SkateFormer_3' (3.52M)
        - 'SkateFormerwithStrokeFusion' (2.71M)
        '''
        T = hyp.T; V = hyp.V  # (T frames, V joints) in total
        N = hyp.N; L = hyp.L  # (N frames, L joints) in first partition

        match model_name:
            # use skateformer2
            case 'SkateFormerwithStrokeFusion': 
                assert T == 32 and N == 4, 'Please set T and N properly.'
                net = SkateFormerwithStrokeFusion(
                    in_channels=2,
                    channels=(96, 192, 192),
                    depths=(2, 2, 2),
                    # Down sampling D=(len(depths)-1) times, if stride is 2
                    # => T_final = T / 2^D
                    # => T_final still has to be divisible by N and M
                    # => You should pick T and N in the following constraints:
                    # <<< T = N*M and T_final % N == 0 and T_final % M == 0 >>>
                    # If 2^D = 2, min(T) = 4 = 2*2
                    # If 2^D = 4, min(T) = 16 = 4*4
                    # If 2^D = 8, min(T) = 64 = 8*8
                    # IF 2^D = 16, min(T) = 256 = 16*16
                    num_classes=hyp.n_classes,
                    num_frames=T,
                    num_points=V,
                    num_people=2,
                    num_heads=32,
                    kernel_size=7,
                    type_1_size=(N, L),  # (N frames, L joints)
                    type_2_size=(N, V // L),  # (N frames, K distant joints)
                    type_3_size=(T // N, L),  # (M distant frames, L joints)
                    type_4_size=(T // N, V // L),  # (M distant frames, K distant joints)
                    use_index_t=True,
                    type_size=hyp.n_classes, 
                    emb_size=128, 
                    hid_size=256
                )

            case 'SkateFormer_3':
                assert T == 128 and N == 8, 'Please set T and N properly.'
                net = SkateFormer(
                    in_channels=2,
                    channels=(96, 192, 192, 192),
                    depths=(2, 2, 2, 2),
                    # Down sampling D=(len(depths)-1) times, if stride is 2
                    # => T_final = T / 2^D
                    # => T_final still has to be divisible by N and M
                    # => You should pick T and N in the following constraints:
                    # <<< T = N*M and T_final % N == 0 and T_final % M == 0 >>>
                    # If 2^D = 2, min(T) = 4 = 2*2
                    # If 2^D = 4, min(T) = 16 = 4*4
                    # If 2^D = 8, min(T) = 64 = 8*8
                    # IF 2^D = 16, min(T) = 256 = 16*16
                    num_classes=hyp.n_classes,
                    num_frames=T,
                    num_points=V,
                    num_people=2,
                    num_heads=32,
                    kernel_size=7,
                    type_1_size=(N, L),  # (N frames, L joints)
                    type_2_size=(N, V // L),  # (N frames, K distant joints)
                    type_3_size=(T // N, L),  # (M distant frames, L joints)
                    type_4_size=(T // N, V // L),  # (M distant frames, K distant joints)
                    use_index_t=True
                )
            
            case _:
                raise NotImplementedError
        
        self.model_name = model_name
        self.net = net.to(self.device)

    def seek_network_weights(self, model_info='', serial_no=1):
        serial_str = f'_{serial_no}' if serial_no != 1 else ''
        model_info = f'_{model_info}' if model_info != '' else ''
        model_postfix = model_info + serial_str

        save_name = self.model_name.lower() + model_postfix
        self.model_name += model_postfix

        weight_path = Path(f'weight/{save_name}.pt')
        if weight_path.exists():
            self.net.load_state_dict(torch.load(str(weight_path), map_location=self.device, weights_only=True))
        else:
            train_t0 = time.time()
            self.net = train_network(
                model=self.net,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                save_path=weight_path
            )
            train_t1 = time.time()
            t = timedelta(seconds=int(train_t1 - train_t0))
            print(f'Total training time: {t}')

    def test(self, show_details=False, show_confusion_matrix=False):
        pred, gt = test(self.net, self.test_loader, self.device)
        print(f'Test (num_strokes: {len(pred)}) =>')

        f1_score_each = multiclass_f1_score(pred, gt, num_classes=hyp.n_classes, average=None)
        show_f1_results(
            model_name=self.model_name,
            f1_score_each=f1_score_each,
            show_details=show_details
        )

        acc = torch.sum(pred == gt).item() / len(pred)
        print('Accuracy:', f'{acc:.3f}')

        if show_confusion_matrix:
            plot_confusion_matrix(
                y_true=gt,
                y_pred=pred,
                need_pre_argmax=False,
                model_name=self.model_name,
                font_size=6,
                save=False
            )

    def test_topk_acc(self, k=2):
        assert k > 1, 'k should be > 1'
        pred, gt = test_topk(self.net, self.test_loader, self.device, k=k)
        gt = gt.unsqueeze(1).repeat(1, k)
        acc = torch.any(pred == gt, dim=1).sum().item() / len(gt)
        print(f'Top{k} Accuracy: {acc:.3f}')

    def compare_pred_gt_on_specific_type(self, dir_path: Path):
        infer_ds = Dataset_npy(
            root_dir=dir_path,
            set_name='test_specific',
            pose_style=self.pose_style,
            seq_len=hyp.T
        )
        infer_loader = DataLoader(
            dataset=infer_ds,
            batch_size=hyp.batch_size,
        )

        pred, gt = test(self.net, infer_loader, self.device)
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()

        with pd.option_context('display.max_rows', None):
            df = pd.DataFrame(
                data={
                    'Ball Round': [Path(e).stem for e in infer_ds.data_branches],
                    'Pred': pred,
                    'GT': gt
                }
            )
            print(df)


if __name__ == '__main__':
    task = Task()
    task.prepare_dataloaders(
        root_dir=Path('rally_data_npy_collated'),
        pose_style='J_only',
        train_partial=1
    )
    task.get_network_architecture(model_name='SkateFormerwithStrokeFusion')
    task.seek_network_weights(model_info='some_no_weight_decay', serial_no=1)
    task.test(show_details=False, show_confusion_matrix=False)
    task.test_topk_acc(k=2)
    # task.compare_pred_gt_on_specific_type(
    #     Path("C:/MyResearch/stroke_classification/dataset_npy/test/Top_點扣")
    # )
