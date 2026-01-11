import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

from transformers import get_cosine_schedule_with_warmup
import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections import namedtuple
import time
from datetime import timedelta

from dataset import get_bone_pairs, prepare_npy_collated_for_temposestroke_loaders,\
                    RandomTranslation_batch, Dataset_npy
from tempose import TemPose_TF, TemPose2_TF, TemPose_TFwithtwoStageStrokeFusion, TemPose_TFwithThreeStroke, \
                ConvTemPose_TFwithStrokeFusion, TemPose_GFwithStrokeFusion
from utils import show_f1_results, plot_confusion_matrix, generate_text_error_report, class_ls_inv


Hyp = namedtuple('Hyp', [
    'n_epochs', 'batch_size', 'lr', 'warm_up_step',
    'n_classes', 'seq_len', 'early_stop_n_epochs'
])
hyp = Hyp(
    n_epochs=1600,
    early_stop_n_epochs=300,
    batch_size=128,
    lr=5e-4,
    warm_up_step=400,
    n_classes=31,
    seq_len=30
)


# only two stages
def train_one_epoch(
    model: nn.Module,
    loader,
    random_shift_fn,
    n_bones: int,
    loss_fn,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    device
):
    model.train()
    total_loss = 0.0
    special_classes = torch.tensor([0, 14, 15, 29, 30], device=device)

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
        human_pose = human_pose.to(device)
        pos = pos.to(device)
        shuttle = shuttle.to(device)
        video_len = video_len.to(device)
        labels = labels.to(device)
        pre_next_stroke = pre_next_stroke.to(device)

        # ---------- Stage 1 ----------
        if n_bones == 0:
            human_pose = random_shift_fn(human_pose)
        else:
            joints = human_pose[:, :, :, :-n_bones, :].contiguous()
            bones = human_pose[:, :, :, -n_bones:, :]
            joints = random_shift_fn(joints)
            human_pose = torch.cat([joints, bones], dim=-2)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        the_stroke_zero = torch.zeros_like(labels)
        pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
        logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_zero, the_stroke_zero)
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

        # # ---------- Debug prints ----------
        # if batch_idx < 3:  # 只印前三個 batch 避免刷屏
        #     print(f"\n=== [Batch {batch_idx}] ===")
        #     print(f"human_pose: {tuple(human_pose.shape)}")
        #     print(f"pre_next_stroke: {tuple(pre_next_stroke.shape)} | sample[0]: {pre_next_stroke[0].tolist()}")
        #     print(f"pred_stage1 shape: {tuple(pred_stage1.shape)} | pred_stage1[:5]: {pred_stage1[:5].tolist()}")
        #     print(f"pre_next_stroke_pair shape: {tuple(pre_next_stroke_pair.shape)}")
        #     print(f"pre_next_stroke_pair[:5]:\n{pre_next_stroke_pair[:5].tolist()}")

        # ---------- Stage 2 ----------
        optimizer.zero_grad()
        logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair, pred_stage1)
        loss = loss_fn(logits_stage2, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    train_loss = total_loss / len(loader)
    return train_loss

##original
# def train_one_epoch(
#     model: nn.Module,
#     loader,
#     random_shift_fn,
#     n_bones: int,
#     loss_fn,
#     optimizer: optim.Optimizer,
#     scheduler: optim.lr_scheduler.LambdaLR,
#     device
# ):
#     model.train()
#     total_loss = 0.0

#     for (human_pose, pos, shuttle), video_len, labels, pre_next_stroke in loader:
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         labels: Tensor = labels.to(device)
#         pre_next_stroke: Tensor = pre_next_stroke.to(device)

#         if n_bones == 0:
#             human_pose = random_shift_fn(human_pose)
#         else:
#             joints = human_pose[:, :, :, :-n_bones, :].contiguous()
#             bones = human_pose[:, :, :, -n_bones:, :]

#             joints = random_shift_fn(joints)
#             human_pose = torch.cat([joints, bones], dim=-2)

#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
#         logits = model(human_pose, pos, shuttle, video_len, pre_next_stroke)
#         loss: Tensor = loss_fn(logits, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#         total_loss += loss.item()
    
#     train_loss = total_loss / len(loader)
#     return train_loss


# # three stages
# def train_one_epoch(
#     model: nn.Module,
#     loader,
#     random_shift_fn,
#     n_bones: int,
#     loss_fn,
#     optimizer: optim.Optimizer,
#     scheduler: optim.lr_scheduler.LambdaLR,
#     device
# ):
#     model.train()
#     total_loss = 0.0
#     special_classes = torch.tensor([0, 14, 15, 29, 30], device=device)

#     for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
#         human_pose = human_pose.to(device)
#         pos = pos.to(device)
#         shuttle = shuttle.to(device)
#         video_len = video_len.to(device)
#         labels = labels.to(device)
#         pre_next_stroke = pre_next_stroke.to(device)

#         # ---------- Stage 1 ----------
#         if n_bones == 0:
#             human_pose = random_shift_fn(human_pose)
#         else:
#             joints = human_pose[:, :, :, :-n_bones, :].contiguous()
#             bones = human_pose[:, :, :, -n_bones:, :]
#             joints = random_shift_fn(joints)
#             human_pose = torch.cat([joints, bones], dim=-2)

#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)

#         pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
#         logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_zero)
#         pred_stage1 = torch.argmax(logits_stage1, dim=1)

#         # ---------- 建立 pre_next_stroke_pair ----------
#         first_prev = pre_next_stroke[0, 0].unsqueeze(0).detach().clone()
#         last_next = pre_next_stroke[-1, 1].unsqueeze(0).detach().clone()

#         if batch_idx == 0:
#             pre_next_stroke_pair = torch.cat([
#                 torch.tensor([0], device=device),
#                 pred_stage1,
#                 last_next
#             ])
#         else:
#             pre_next_stroke_pair = torch.cat([
#                 first_prev,
#                 pred_stage1,
#                 last_next
#             ])

#         pre_next_stroke_pair = torch.stack(
#             [pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1
#         )

#         # ---------- 修正特殊類別 ----------
#         mask_current = torch.isin(pred_stage1, special_classes)
#         pre_next_stroke_pair[mask_current, 0] = 0

#         mask_next = torch.isin(pre_next_stroke_pair[:, 1], special_classes)
#         pre_next_stroke_pair[mask_next, 1] = 0

#         # # ---------- Debug prints ----------
#         # if batch_idx < 3:  # 只印前三個 batch 避免刷屏
#         #     print(f"\n=== [Batch {batch_idx}] ===")
#         #     print(f"human_pose: {tuple(human_pose.shape)}")
#         #     print(f"pre_next_stroke: {tuple(pre_next_stroke.shape)} | sample[0]: {pre_next_stroke[0].tolist()}")
#         #     print(f"pred_stage1 shape: {tuple(pred_stage1.shape)} | pred_stage1[:5]: {pred_stage1[:5].tolist()}")
#         #     print(f"pre_next_stroke_pair shape: {tuple(pre_next_stroke_pair.shape)}")
#         #     print(f"pre_next_stroke_pair[:5]:\n{pre_next_stroke_pair[:5].tolist()}")

#         # ---------- Stage 2 ----------
#         logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair)
#         pred_stage2 = torch.argmax(logits_stage2, dim=1)
#         if batch_idx == 0:
#             pre_next_stroke_pair_stage2 = torch.cat([
#                 torch.tensor([0], device=device),
#                 pred_stage2,
#                 last_next
#             ])
#         else:
#             pre_next_stroke_pair_stage2 = torch.cat([
#                 first_prev,
#                 pred_stage2,
#                 last_next
#             ])

#         pre_next_stroke_pair_stage2 = torch.stack(
#             [pre_next_stroke_pair_stage2[:-2], pre_next_stroke_pair_stage2[2:]], dim=1
#         )

#         # ---------- 修正特殊類別 ----------
#         mask_current = torch.isin(pred_stage2, special_classes)
#         pre_next_stroke_pair_stage2[mask_current, 0] = 0

#         mask_next = torch.isin(pre_next_stroke_pair_stage2[:, 1], special_classes)
#         pre_next_stroke_pair_stage2[mask_next, 1] = 0

#         # ---------- Stage 3 ----------
#         logits_stage3 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair_stage2)
#         # pred_stage3 = torch.argmax(logits_stage3, dim=1)

#         loss = loss_fn(logits_stage3, labels)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#         total_loss += loss.item()

#     train_loss = total_loss / len(loader)
#     return train_loss


@torch.no_grad()
# two stages
def validate(model: nn.Module, loss_fn, loader, device):
    model.eval()

    total_loss = 0.0
    cum_tp = torch.zeros(hyp.n_classes)
    cum_tn = torch.zeros(hyp.n_classes)
    cum_fp = torch.zeros(hyp.n_classes)
    cum_fn = torch.zeros(hyp.n_classes)

    all_preds_stage1 = []  # 用來儲存所有 batch 的預測結果

    # 第一階段：將所有 batch 的預測結果收集起來
    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        pos: Tensor = pos.to(device)
        shuttle: Tensor = shuttle.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        pre_next_stroke: Tensor = pre_next_stroke.to(device)

        # ---------- 第一階段 ----------
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        the_stroke_zero = torch.zeros_like(labels)
        pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
        logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_zero, the_stroke_zero)
        pred_stage1 = torch.argmax(logits_stage1, dim=1)

        # 儲存每個 batch 的預測結果
        all_preds_stage1.append(pred_stage1.detach().clone())

        # print some intermediate results
        # print(f"Batch {batch_idx} - Stage 1 Preds: {pred_stage1.tolist()}")

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
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        
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
            # print(f"pre_next_stroke_pair 1: {pre_next_stroke_pair}")  # 第一個 batch
            # print(f"pre_next_stroke_pair 1 shape: {pre_next_stroke_pair.shape}")  # 第一個 batch
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

        # print(f"pre_next_stroke_pair2 shape: {pre_next_stroke_pair.shape}")  #
        # print(f"pre_next_stroke_pair2 : {pre_next_stroke_pair}") 
        # 將 pre_next_stroke_pair 重塑為 [batch_size, 2]，包含前一拍和下一拍
        pre_next_stroke_pair = pre_next_stroke_pair.view(-1, 2)
        # print(f"pre_next_stroke_pair 3 : {pre_next_stroke_pair.shape}") 

        # 模型推理
        logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair, all_preds_stage1[batch_idx])

        pred_stage2 = torch.argmax(logits_stage2, dim=1)

        loss = loss_fn(logits_stage2, labels)
        total_loss += loss.item()

        # ---------- 計算指標 ----------
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

    # ---------- 最後計算整體結果 ----------
    val_loss = total_loss / len(loader)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / (cum_tp + cum_fn)
    f1_score = 2 * precision * recall / (precision + recall)
    f1_score[f1_score.isnan()] = 0

    f1_score_avg = f1_score.mean()
    f1_score_min = f1_score.min()

    return val_loss, f1_score_avg, f1_score_min


##original
# def validate(model: nn.Module, loss_fn, loader, device):
#     model.eval()
   
#     total_loss = 0.0
#     cum_tp = torch.zeros(hyp.n_classes)
#     cum_tn = torch.zeros(hyp.n_classes)
#     cum_fp = torch.zeros(hyp.n_classes)
#     cum_fn = torch.zeros(hyp.n_classes)

#     for (human_pose, pos, shuttle), video_len, labels, pre_next_stroke in loader:
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         labels: Tensor = labels.to(device)
#         pre_next_stroke: Tensor = pre_next_stroke.to(device)

#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
#         # # ===== 第一階段 (pre_next_stroke 全 0) =====
      
#         # logits = model(human_pose, pos, shuttle, video_len, pre_next_stroke)
#         # pred = torch.argmax(logits, dim=1)

#         # ===== 第一階段 (pre_next_stroke 全 0) =====
#         pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
#         logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_zero)
#         pred_stage1 = torch.argmax(logits_stage1, dim=1)

#         # 定義特殊類別
#         special_classes = torch.tensor([14, 15, 29, 30], device=pred_stage1.device)
      
#         padded = torch.cat([torch.tensor([0], device=pred_stage1.device), 
#                     pred_stage1, 
#                     torch.tensor([0], device=pred_stage1.device)])
#         pre_next_stroke_pair = torch.stack([padded[:-2], padded[2:]], dim=1)

#         # 當前 stroke 是特殊類別 → pre 歸 0
#         mask_current = torch.isin(pred_stage1, special_classes)
#         pre_next_stroke_pair[mask_current, 0] = 0

#         # 下一個 stroke 是特殊類別 → next 歸 0
#         # (對 pre_next_stroke_pair 的上一個項目做處理)
#         for i in range(len(pred_stage1) - 1):
#             if pred_stage1[i + 1] in special_classes:
#                 pre_next_stroke_pair[i, 1] = 0

#         print(f"Pre-next stroke pairs: {pre_next_stroke_pair}")
#         print('Pre-next stroke pairs shape:' ,pre_next_stroke_pair.shape)
#         # ===== 第二階段 (再用 pred_stage1) =====
#         logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair)
#         loss: Tensor = loss_fn(logits_stage2, labels)
#         total_loss += loss.item()

#         # ===== 評估指標 =====
#         pred = F.one_hot(torch.argmax(logits_stage2, dim=1), hyp.n_classes).bool()
#         labels_onehot = F.one_hot(labels, hyp.n_classes).bool()

#         # #  # ===== 評估指標 =====
#         # loss: Tensor = loss_fn(logits, labels)
#         # total_loss += loss.item()
#         # pred = F.one_hot(torch.argmax(logits, dim=1), hyp.n_classes).bool()
#         # labels_onehot = F.one_hot(labels, hyp.n_classes).bool()

#         tp = torch.sum(pred & labels_onehot, dim=0)
#         tn = torch.sum(~pred & ~labels_onehot, dim=0)
#         fp = torch.sum(pred & ~labels_onehot, dim=0)
#         fn = torch.sum(~pred & labels_onehot, dim=0)

#         cum_tp += tp.cpu()
#         cum_tn += tn.cpu()
#         cum_fp += fp.cpu()
#         cum_fn += fn.cpu()

#     val_loss = total_loss / len(loader)

#     precision = cum_tp / (cum_tp + cum_fp)
#     recall = cum_tp / (cum_tp + cum_fn)
#     f1_score = 2 * precision * recall / (precision + recall)
#     f1_score[f1_score.isnan()] = 0

#     f1_score_avg = f1_score.mean()
#     f1_score_min = f1_score.min()
#     return val_loss, f1_score_avg, f1_score_min

# # three stages
# def validate(model: nn.Module, loss_fn, loader, device):
#     model.eval()

#     total_loss = 0.0
#     cum_tp = torch.zeros(hyp.n_classes)
#     cum_tn = torch.zeros(hyp.n_classes)
#     cum_fp = torch.zeros(hyp.n_classes)
#     cum_fn = torch.zeros(hyp.n_classes)

#     all_preds_stage1 = []  # 用來儲存所有 batch 的預測結果
#     all_preds_stage2 = []  # 用來儲存所有 batch 的預測結果
#     # 第一階段：將所有 batch 的預測結果收集起來
#     for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         labels: Tensor = labels.to(device)
#         pre_next_stroke: Tensor = pre_next_stroke.to(device)

#         # ---------- 第一階段 ----------
#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
#         pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
#         logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_zero)
#         pred_stage1 = torch.argmax(logits_stage1, dim=1)

#         # 儲存每個 batch 的預測結果
#         all_preds_stage1.append(pred_stage1.detach().clone())

#         # print some intermediate results
#         # print(f"Batch {batch_idx} - Stage 1 Preds: {pred_stage1.tolist()}")

#     # 第二階段：使用所有第一階段的預測結果來組裝 pre_next_stroke_pair
#     prev_last = torch.tensor([0], device=device)  # 用於記錄前一個 batch 最後的預測
#     next_last = torch.tensor([all_preds_stage1[1][0].item()], device=device)  # 用於記錄下一個 batch 最後的預測
  

#     # 現在將所有的預測結果組合為 pre_next_stroke_pair 並進行第二階段推理
#     for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         labels: Tensor = labels.to(device)
#         pre_next_stroke: Tensor = pre_next_stroke.to(device)
#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        
#         # ---------- 第二階段 ----------
#         # 合併前一拍和下一拍的預測結果
#         special_classes = torch.tensor([0, 14, 15, 29, 30], device=pred_stage1.device)

#         # 使用第一階段的預測結果來創建 pre_next_stroke_pair
#         if batch_idx == 0:
#             # 第一個 batch 用 0 當前一拍和下一拍
#             pre_next_stroke_pair = torch.cat([torch.tensor([0], device=device), all_preds_stage1[batch_idx], next_last.detach().clone()])
#             pre_next_stroke_pair = torch.stack([pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1)

#             # print(f"pre_next_stroke_pair : {pre_next_stroke_pair}")  # 第一個 batch
#             # print(f"pre_next_stroke_pair shape: {pre_next_stroke_pair.shape}")  # 第一個 batch
#         else:
#             # 其他 batch 若有特殊類別 → 前一拍設為 0，否則用上一批的最後一個
#             pre_next_stroke_pair = torch.cat([prev_last.detach().clone(), all_preds_stage1[batch_idx], next_last.detach().clone()])
#             pre_next_stroke_pair = torch.stack([pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1)
#             # print(f"pre_next_stroke_pair 1: {pre_next_stroke_pair}")  # 第一個 batch
#             # print(f"pre_next_stroke_pair 1 shape: {pre_next_stroke_pair.shape}")  # 第一個 batch
#         # 此 batch 的最後一拍的下一拍應該是下一個 batch 的第一拍
#         if batch_idx + 1 < len(all_preds_stage1):
#             next_last = all_preds_stage1[batch_idx + 1][0].detach().clone()

#         # 修正特殊類別：如果當前拍是特殊類別，將前一拍設為 0
#         pred_current = torch.isin(all_preds_stage1[batch_idx], special_classes)
#         pre_next_stroke_pair[pred_current, 0] = 0  # 若當前拍是特殊類別，將前一拍設為 0

#         mask_current = torch.isin(pre_next_stroke_pair[:, 1], special_classes)
#         pre_next_stroke_pair[mask_current, 1] = 0

#         # 修正下一拍：如果下一拍是特殊類別，將當前拍的下一拍設為 0
        
    

#         # 更新 prev_last 和 next_last（記錄此 batch 最後一個預測）
#         prev_last = all_preds_stage1[batch_idx][-1:].detach().clone()

#         # 將 next_last 轉換為一維張量，以便能與其他張量拼接
#         next_last = torch.tensor([next_last.item()], device=device)  # 確保它是可以拼接的

#         # print(f"pre_next_stroke_pair2 shape: {pre_next_stroke_pair.shape}")  #
#         # print(f"pre_next_stroke_pair2 : {pre_next_stroke_pair}") 
#         # 將 pre_next_stroke_pair 重塑為 [batch_size, 2]，包含前一拍和下一拍
#         pre_next_stroke_pair = pre_next_stroke_pair.view(-1, 2)
#         # print(f"pre_next_stroke_pair 3 : {pre_next_stroke_pair.shape}") 

#         # 模型推理
#         logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair)

#         pred_stage2 = torch.argmax(logits_stage2, dim=1)
#         all_preds_stage2.append(pred_stage2.detach().clone())

#     # 第三階段：使用所有第一階段的預測結果來組裝 pre_next_stroke_pair
#     prev_last_stage2 = torch.tensor([0], device=device)  # 用於記錄前一個 batch 最後的預測
#     next_last_stage2 = torch.tensor([all_preds_stage2[1][0].item()], device=device)  # 用於記錄下一個 batch 最後的預測
#     total_loss = 0.0

#     # 現在將所有的預測結果組合為 pre_next_stroke_pair 並進行第三階段推理
#     for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         labels: Tensor = labels.to(device)
#         pre_next_stroke: Tensor = pre_next_stroke.to(device)
#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        
#         # ---------- 第二階段 ----------
#         # 合併前一拍和下一拍的預測結果
#         special_classes = torch.tensor([0, 14, 15, 29, 30], device=pred_stage1.device)

#         # 使用第一階段的預測結果來創建 pre_next_stroke_pair
#         if batch_idx == 0:
#             # 第一個 batch 用 0 當前一拍和下一拍
#             pre_next_stroke_pair_stage2 = torch.cat([torch.tensor([0], device=device), all_preds_stage2[batch_idx], next_last_stage2.detach().clone()])
#             pre_next_stroke_pair_stage2 = torch.stack([pre_next_stroke_pair_stage2[:-2], pre_next_stroke_pair_stage2[2:]], dim=1)

#             # print(f"pre_next_stroke_pair : {pre_next_stroke_pair}")  # 第一個 batch
#             # print(f"pre_next_stroke_pair shape: {pre_next_stroke_pair.shape}")  # 第一個 batch
#         else:
#             # 其他 batch 若有特殊類別 → 前一拍設為 0，否則用上一批的最後一個
#             pre_next_stroke_pair_stage2 = torch.cat([prev_last_stage2.detach().clone(), all_preds_stage2[batch_idx], next_last_stage2.detach().clone()])
#             pre_next_stroke_pair_stage2 = torch.stack([pre_next_stroke_pair_stage2[:-2], pre_next_stroke_pair_stage2[2:]], dim=1)
#             # print(f"pre_next_stroke_pair 1: {pre_next_stroke_pair}")  # 第一個 batch
#             # print(f"pre_next_stroke_pair 1 shape: {pre_next_stroke_pair.shape}")  # 第一個 batch
#         # 此 batch 的最後一拍的下一拍應該是下一個 batch 的第一拍
#         if batch_idx + 1 < len(all_preds_stage2):
#             next_last_stage2 = all_preds_stage2[batch_idx + 1][0].detach().clone()

#         # 修正特殊類別：如果當前拍是特殊類別，將前一拍設為 0
#         pred_current = torch.isin(all_preds_stage2[batch_idx], special_classes)
#         pre_next_stroke_pair_stage2[pred_current, 0] = 0  # 若當前拍是特殊類別，將前一拍設為 0

#         mask_current = torch.isin(pre_next_stroke_pair_stage2[:, 1], special_classes)
#         pre_next_stroke_pair_stage2[mask_current, 1] = 0

#         # 修正下一拍：如果下一拍是特殊類別，將當前拍的下一拍設為 0
        
    

#         # 更新 prev_last 和 next_last（記錄此 batch 最後一個預測）
#         prev_last_stage2 = all_preds_stage2[batch_idx][-1:].detach().clone()

#         # 將 next_last 轉換為一維張量，以便能與其他張量拼接
#         next_last_stage2 = torch.tensor([next_last_stage2.item()], device=device)  # 確保它是可以拼接的

#         # print(f"pre_next_stroke_pair2 shape: {pre_next_stroke_pair.shape}")  #
#         # print(f"pre_next_stroke_pair2 : {pre_next_stroke_pair}") 
#         # 將 pre_next_stroke_pair 重塑為 [batch_size, 2]，包含前一拍和下一拍
#         pre_next_stroke_pair_stage3 = pre_next_stroke_pair_stage2.view(-1, 2)
#         # print(f"pre_next_stroke_pair 3 : {pre_next_stroke_pair.shape}") 

#         # 模型推理
#         logits_stage3 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair_stage3)

#         pred_stage3 = torch.argmax(logits_stage3, dim=1)
        

        

#         loss = loss_fn(logits_stage3, labels)
#         total_loss += loss.item()

#         # ---------- 計算指標 ----------
#         pred = F.one_hot(torch.argmax(logits_stage3, dim=1), hyp.n_classes).bool()
#         labels_onehot = F.one_hot(labels, hyp.n_classes).bool()

#         tp = torch.sum(pred & labels_onehot, dim=0)
#         tn = torch.sum(~pred & ~labels_onehot, dim=0)
#         fp = torch.sum(pred & ~labels_onehot, dim=0)
#         fn = torch.sum(~pred & labels_onehot, dim=0)

#         cum_tp += tp.cpu()
#         cum_tn += tn.cpu()
#         cum_fp += fp.cpu()
#         cum_fn += fn.cpu()

#     # ---------- 最後計算整體結果 ----------
#     val_loss = total_loss / len(loader)
#     precision = cum_tp / (cum_tp + cum_fp)
#     recall = cum_tp / (cum_tp + cum_fn)
#     f1_score = 2 * precision * recall / (precision + recall)
#     f1_score[f1_score.isnan()] = 0

#     f1_score_avg = f1_score.mean()
#     f1_score_min = f1_score.min()

#     return val_loss, f1_score_avg, f1_score_min

@torch.no_grad()
# two stages
def test(model: nn.Module, loader, device):
    model.eval()
    pred_ls, labels_ls = [], []
    misclassified_reports = []

    all_preds_stage1 = []  # 收集第一階段所有 batch 的預測結果

    # ========== 第一階段：預測所有 batch ==========
    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        pos: Tensor = pos.to(device)
        shuttle: Tensor = shuttle.to(device)
        video_len: Tensor = video_len.to(device)
        pre_next_stroke: Tensor = pre_next_stroke.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)

        # 第一階段輸入 pre_next_stroke 全為 0
        the_stroke_zero = torch.zeros_like(labels.to(device))
        pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
        logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_zero, the_stroke_zero)
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
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)

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
        logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair, all_preds_stage1[batch_idx])
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

    return torch.cat(pred_ls), torch.cat(labels_ls), misclassified_reports

##original
# def test(model: nn.Module, loader, device):
#     model.eval()
#     pred_ls, labels_ls = [], []
#     misclassified_reports = []

#     for (human_pose, pos, shuttle), video_len, labels, pre_next_stroke in loader:
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         pre_next_stroke: Tensor = pre_next_stroke.to(device)

#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)

#         # ========= 第一階段 =========
#         pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)  # 永遠是 0
#         logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_zero)
#         pred_stage1 = torch.argmax(logits_stage1, dim=1)  # (N,)

#         # 定義特殊類別
#         special_classes = torch.tensor([14, 15, 29, 30], device=pred_stage1.device)
      
#         padded = torch.cat([torch.tensor([0], device=pred_stage1.device), 
#                     pred_stage1, 
#                     torch.tensor([0], device=pred_stage1.device)])
#         pre_next_stroke_pair = torch.stack([padded[:-2], padded[2:]], dim=1)

#         # 當前 stroke 是特殊類別 → pre 歸 0
#         mask_current = torch.isin(pred_stage1, special_classes)
#         pre_next_stroke_pair[mask_current, 0] = 0

#         # 下一個 stroke 是特殊類別 → next 歸 0
#         # (對 pre_next_stroke_pair 的上一個項目做處理)
#         for i in range(len(pred_stage1) - 1):
#             if pred_stage1[i + 1] in special_classes:
#                 pre_next_stroke_pair[i, 1] = 0

#         # ========= 第二階段 =========
#         logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair)
#         pred_stage2 = torch.argmax(logits_stage2, dim=1).cpu()

#         # ========= 記錄 =========
#         is_misclassified = (pred_stage2 != labels)
#         if is_misclassified.any():
#             misclassified_pred = pred_stage2[is_misclassified].tolist()
#             misclassified_labels = labels[is_misclassified].tolist()
#             misclassified_pairs = list(zip(misclassified_labels, misclassified_pred))
#             misclassified_reports.extend(misclassified_pairs)

#         pred_ls.append(pred_stage2)
#         labels_ls.append(labels)

#     return torch.cat(pred_ls), torch.cat(labels_ls), misclassified_reports    

    # return torch.cat(pred_ls), torch.cat(labels_ls)



@torch.no_grad()
# two stages
def test_topk(
    model: nn.Module,
    loader,
    device,
    k=2
):
    model.eval()
    pred_ls, labels_ls = [], []
    all_preds_stage1 = []

    # ========== 第一階段：收集所有 batch 預測 ==========
    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        pos: Tensor = pos.to(device)
        shuttle: Tensor = shuttle.to(device)
        video_len: Tensor = video_len.to(device)
        pre_next_stroke: Tensor = pre_next_stroke.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        the_stroke_zero = torch.zeros_like(labels.to(device))
        pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
        logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_zero, the_stroke_zero)
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
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)

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
        logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair, all_preds_stage1[batch_idx])

        # top-k 預測（例如 top2）
        _, pred_topk = torch.topk(logits_stage2, k=k, dim=1)

        pred_ls.append(pred_topk.cpu())
        labels_ls.append(labels.cpu())

    return torch.cat(pred_ls), torch.cat(labels_ls)

# # three stages
# def test_topk(
#     model: nn.Module,
#     loader,
#     device,
#     k=2
# ):
#     model.eval()
#     pred_ls, labels_ls = [], []
#     all_preds_stage1 = []  # 用來儲存所有 batch 的預測結果
#     all_preds_stage2 = []  # 用來儲存所有 batch 的預測結果
#     # 第一階段：將所有 batch 的預測結果收集起來
#     for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         labels: Tensor = labels.to(device)
#         pre_next_stroke: Tensor = pre_next_stroke.to(device)

#         # ---------- 第一階段 ----------
#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
#         pre_next_stroke_zero = torch.zeros_like(pre_next_stroke)
#         logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_zero)
#         pred_stage1 = torch.argmax(logits_stage1, dim=1)

#         # 儲存每個 batch 的預測結果
#         all_preds_stage1.append(pred_stage1.detach().clone())

#         # print some intermediate results
#         # print(f"Batch {batch_idx} - Stage 1 Preds: {pred_stage1.tolist()}")

#     # 第二階段：使用所有第一階段的預測結果來組裝 pre_next_stroke_pair
#     prev_last = torch.tensor([0], device=device)  # 用於記錄前一個 batch 最後的預測
#     next_last = torch.tensor([all_preds_stage1[1][0].item()], device=device)  # 用於記錄下一個 batch 最後的預測
  

#     # 現在將所有的預測結果組合為 pre_next_stroke_pair 並進行第二階段推理
#     for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         labels: Tensor = labels.to(device)
#         pre_next_stroke: Tensor = pre_next_stroke.to(device)
#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        
#         # ---------- 第二階段 ----------
#         # 合併前一拍和下一拍的預測結果
#         special_classes = torch.tensor([0, 14, 15, 29, 30], device=pred_stage1.device)

#         # 使用第一階段的預測結果來創建 pre_next_stroke_pair
#         if batch_idx == 0:
#             # 第一個 batch 用 0 當前一拍和下一拍
#             pre_next_stroke_pair = torch.cat([torch.tensor([0], device=device), all_preds_stage1[batch_idx], next_last.detach().clone()])
#             pre_next_stroke_pair = torch.stack([pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1)

#             # print(f"pre_next_stroke_pair : {pre_next_stroke_pair}")  # 第一個 batch
#             # print(f"pre_next_stroke_pair shape: {pre_next_stroke_pair.shape}")  # 第一個 batch
#         else:
#             # 其他 batch 若有特殊類別 → 前一拍設為 0，否則用上一批的最後一個
#             pre_next_stroke_pair = torch.cat([prev_last.detach().clone(), all_preds_stage1[batch_idx], next_last.detach().clone()])
#             pre_next_stroke_pair = torch.stack([pre_next_stroke_pair[:-2], pre_next_stroke_pair[2:]], dim=1)
#             # print(f"pre_next_stroke_pair 1: {pre_next_stroke_pair}")  # 第一個 batch
#             # print(f"pre_next_stroke_pair 1 shape: {pre_next_stroke_pair.shape}")  # 第一個 batch
#         # 此 batch 的最後一拍的下一拍應該是下一個 batch 的第一拍
#         if batch_idx + 1 < len(all_preds_stage1):
#             next_last = all_preds_stage1[batch_idx + 1][0].detach().clone()

#         # 修正特殊類別：如果當前拍是特殊類別，將前一拍設為 0
#         pred_current = torch.isin(all_preds_stage1[batch_idx], special_classes)
#         pre_next_stroke_pair[pred_current, 0] = 0  # 若當前拍是特殊類別，將前一拍設為 0

#         mask_current = torch.isin(pre_next_stroke_pair[:, 1], special_classes)
#         pre_next_stroke_pair[mask_current, 1] = 0

#         # 修正下一拍：如果下一拍是特殊類別，將當前拍的下一拍設為 0
        
    

#         # 更新 prev_last 和 next_last（記錄此 batch 最後一個預測）
#         prev_last = all_preds_stage1[batch_idx][-1:].detach().clone()

#         # 將 next_last 轉換為一維張量，以便能與其他張量拼接
#         next_last = torch.tensor([next_last.item()], device=device)  # 確保它是可以拼接的

#         # print(f"pre_next_stroke_pair2 shape: {pre_next_stroke_pair.shape}")  #
#         # print(f"pre_next_stroke_pair2 : {pre_next_stroke_pair}") 
#         # 將 pre_next_stroke_pair 重塑為 [batch_size, 2]，包含前一拍和下一拍
#         pre_next_stroke_pair = pre_next_stroke_pair.view(-1, 2)
#         # print(f"pre_next_stroke_pair 3 : {pre_next_stroke_pair.shape}") 

#         # 模型推理
#         logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair)

#         pred_stage2 = torch.argmax(logits_stage2, dim=1)
#         all_preds_stage2.append(pred_stage2.detach().clone())

#     # 第三階段：使用所有第一階段的預測結果來組裝 pre_next_stroke_pair
#     prev_last_stage2 = torch.tensor([0], device=device)  # 用於記錄前一個 batch 最後的預測
#     next_last_stage2 = torch.tensor([all_preds_stage2[1][0].item()], device=device)  # 用於記錄下一個 batch 最後的預測


#     # 現在將所有的預測結果組合為 pre_next_stroke_pair 並進行第三階段推理
#     for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_next_stroke) in enumerate(loader):
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         labels: Tensor = labels.to(device)
#         pre_next_stroke: Tensor = pre_next_stroke.to(device)
#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        
#         # ---------- 第二階段 ----------
#         # 合併前一拍和下一拍的預測結果
#         special_classes = torch.tensor([0, 14, 15, 29, 30], device=pred_stage1.device)

#         # 使用第一階段的預測結果來創建 pre_next_stroke_pair
#         if batch_idx == 0:
#             # 第一個 batch 用 0 當前一拍和下一拍
#             pre_next_stroke_pair_stage2 = torch.cat([torch.tensor([0], device=device), all_preds_stage2[batch_idx], next_last_stage2.detach().clone()])
#             pre_next_stroke_pair_stage2 = torch.stack([pre_next_stroke_pair_stage2[:-2], pre_next_stroke_pair_stage2[2:]], dim=1)

#             # print(f"pre_next_stroke_pair : {pre_next_stroke_pair}")  # 第一個 batch
#             # print(f"pre_next_stroke_pair shape: {pre_next_stroke_pair.shape}")  # 第一個 batch
#         else:
#             # 其他 batch 若有特殊類別 → 前一拍設為 0，否則用上一批的最後一個
#             pre_next_stroke_pair_stage2 = torch.cat([prev_last_stage2.detach().clone(), all_preds_stage2[batch_idx], next_last_stage2.detach().clone()])
#             pre_next_stroke_pair_stage2 = torch.stack([pre_next_stroke_pair_stage2[:-2], pre_next_stroke_pair_stage2[2:]], dim=1)
#             # print(f"pre_next_stroke_pair 1: {pre_next_stroke_pair}")  # 第一個 batch
#             # print(f"pre_next_stroke_pair 1 shape: {pre_next_stroke_pair.shape}")  # 第一個 batch
#         # 此 batch 的最後一拍的下一拍應該是下一個 batch 的第一拍
#         if batch_idx + 1 < len(all_preds_stage2):
#             next_last_stage2 = all_preds_stage2[batch_idx + 1][0].detach().clone()

#         # 修正特殊類別：如果當前拍是特殊類別，將前一拍設為 0
#         pred_current = torch.isin(all_preds_stage2[batch_idx], special_classes)
#         pre_next_stroke_pair_stage2[pred_current, 0] = 0  # 若當前拍是特殊類別，將前一拍設為 0

#         mask_current = torch.isin(pre_next_stroke_pair_stage2[:, 1], special_classes)
#         pre_next_stroke_pair_stage2[mask_current, 1] = 0

#         # 修正下一拍：如果下一拍是特殊類別，將當前拍的下一拍設為 0
        
#          # 更新 prev_last 和 next_last（記錄此 batch 最後一個預測）
#         prev_last_stage2 = all_preds_stage2[batch_idx][-1:].detach().clone()

#         # 將 next_last 轉換為一維張量，以便能與其他張量拼接
#         next_last_stage2 = torch.tensor([next_last_stage2.item()], device=device)  # 確保它是可以拼接的

   
#         # print(f"pre_next_stroke_pair2 shape: {pre_next_stroke_pair.shape}")  #
#         # print(f"pre_next_stroke_pair2 : {pre_next_stroke_pair}") 
#         # 將 pre_next_stroke_pair 重塑為 [batch_size, 2]，包含前一拍和下一拍
#         pre_next_stroke_pair_stage3 = pre_next_stroke_pair_stage2.view(-1, 2)
#         # print(f"pre_next_stroke_pair 3 : {pre_next_stroke_pair.shape}") 

#         # 模型推理
#         logits_stage3 = model(human_pose, pos, shuttle, video_len, pre_next_stroke_pair_stage3)


#         # top-k 預測（例如 top2）
#         _, pred_topk = torch.topk(logits_stage3, k=k, dim=1)

#         pred_ls.append(pred_topk.cpu())
#         labels_ls.append(labels.cpu())

#     return torch.cat(pred_ls), torch.cat(labels_ls)


def train_network(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    save_path: Path,
    n_bones
):
    random_shift_fn = RandomTranslation_batch()

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=hyp.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=hyp.warm_up_step,
        num_training_steps=(hyp.n_epochs * len(train_loader)),
        num_cycles=0.25
    )

    best_value = 0.0
    early_stop_count = 0

    for epoch in range(1, hyp.n_epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            random_shift_fn=random_shift_fn,
            n_bones=n_bones,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
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
    def __init__(self, n_joints=17) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.n_joints = n_joints
        self.class_ls = []

    def prepare_dataloaders(
        self,
        root_dir: Path,
        pose_style='Jn2B',
        train_partial=1.0
    ):
        self.train_loader, \
        self.val_loader, \
        self.test_loader \
            = prepare_npy_collated_for_temposestroke_loaders(
                root_dir=root_dir,
                pose_style=pose_style,
                batch_size=hyp.batch_size,
                use_cuda=self.use_cuda,
                num_workers=(0, 0, 0),
                train_partial=train_partial
            )
        
        self.pose_style = pose_style

    def get_network_architecture(self, model_name, in_channels=2):
        '''
        `model_name`
        - 'TemPose_TF' (about 1.71M)
        - 'TemPose2_TF' (about 1.81M)
        - 'TemPose_TFwithStrokeFusion' (about 1.95M)
        - 'TemPose_TFwithThreeStroke' (about 2.16M)
        '''
        n_bones = len(get_bone_pairs())

        match self.pose_style:
            case 'J_only':
                extra = 0
            case 'JnB_bone' | 'JnB_interp':
                extra = 1
            case 'Jn2B':
                extra = 2

        match model_name:
            case 'TemPose_TF':
                net = TemPose_TF(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=hyp.n_classes,
                    seq_len=hyp.seq_len,
                    depth_inter=2
                )

            case 'TemPose2_TF':
                net = TemPose2_TF(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=hyp.n_classes,
                    seq_len=hyp.seq_len,
                    depth_inter=2
                )

            case 'TemPose_TFwithtwoStageStrokeFusion':
                net = TemPose_TFwithtwoStageStrokeFusion(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=hyp.n_classes,
                    seq_len=hyp.seq_len,
                    depth_inter=2
                )
            
            case 'TemPose_TFwithThreeStroke':
                net = TemPose_TFwithThreeStroke(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=hyp.n_classes,
                    seq_len=hyp.seq_len,
                    depth_inter=2
                )

            case 'TemPose_GFwithStrokeFusion':
                net = TemPose_GFwithStrokeFusion(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=hyp.n_classes,
                    seq_len=hyp.seq_len,
                    depth_inter=2
                )


            case 'ConvTemPose_TFwithStrokeFusion':
                net = ConvTemPose_TFwithStrokeFusion(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=hyp.n_classes,
                    seq_len=hyp.seq_len,
                    depth_inter=2
                )

            

            case _:
                raise NotImplementedError
        
        self.model_name = model_name
        self.net = net.to(self.device)
        self.n_bones = n_bones

    def seek_network_weights(self, model_info='', serial_no=1):
        serial_str = f'_{serial_no}' if serial_no != 1 else ''
        model_info = f'_{model_info}' if model_info != '' else ''
        model_postfix = '_' + self.pose_style + model_info + serial_str

        first_name, last_name = self.model_name.split('_')
        save_name = first_name.lower() + '_' + last_name \
                    + model_postfix
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
                save_path=weight_path,
                n_bones=len(get_bone_pairs()) if self.pose_style != 'J_only' else 0
            )
            train_t1 = time.time()
            t = timedelta(seconds=int(train_t1 - train_t0))
            print(f'Total training time: {t}')

    def test(self, show_details=False, show_confusion_matrix=False, show_misclassified_reports = True):
        pred, gt, misclassified_reports = test(self.net, self.test_loader, self.device)
        
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

        if show_misclassified_reports:
            generate_text_error_report(misclassified_reports, class_ls_inv)
    
   

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
            seq_len=hyp.seq_len
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
    task = Task(n_joints=17)
    task.prepare_dataloaders(
        root_dir=Path("rally_data_npy_collated"),
        pose_style='JnB_bone',
        train_partial=1
    )
    task.get_network_architecture(model_name='TemPose_TFwithThreeStroke', in_channels=2)
    task.seek_network_weights(model_info='rally_data_npy_collated', serial_no=11)
    task.test(show_details=False, show_confusion_matrix=False, show_misclassified_reports = False)
    task.test_topk_acc(k=2)
   
    # task.compare_pred_gt_on_specific_type(
    #     Path("C:/MyResearch/stroke_classification/dataset_npy/test/Top_點扣")
    # )
