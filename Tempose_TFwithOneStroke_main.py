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

from dataset import get_bone_pairs, prepare_npy_collated_for_tempose_loaders,\
                    RandomTranslation_batch, Dataset_npy
from tempose import TemPose_TFwithOneStroke

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

#     for (human_pose, pos, shuttle), video_len, labels, pre_stroke in loader:
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         # nexthuman_pose: Tensor = nexthuman_pose.to(device)
#         # nextpos: Tensor = nextpos.to(device)
#         # nextshuttle: Tensor = nextshuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         labels: Tensor = labels.to(device)
#         pre_stroke: Tensor = pre_stroke.to(device)

#         if n_bones == 0:
#             human_pose = random_shift_fn(human_pose)
#             # nexthuman_pose = random_shift_fn(nexthuman_pose)
#         else:
#             joints = human_pose[:, :, :, :-n_bones, :].contiguous()
#             bones = human_pose[:, :, :, -n_bones:, :]
#             # nextjoints = nexthuman_pose[:, :, :, :-n_bones, :].contiguous()
#             # nextbones = nexthuman_pose[:, :, :, -n_bones:, :]


#             joints = random_shift_fn(joints)
#             human_pose = torch.cat([joints, bones], dim=-2)
            
#             # nextjoints = random_shift_fn(nextjoints)
#             # nexthuman_pose = torch.cat([nextjoints, nextbones], dim=-2)

#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
#         # nexthuman_pose = nexthuman_pose.view(*nexthuman_pose.shape[:-2], -1)
#         logits = model(human_pose, pos, shuttle, video_len, pre_stroke)
#         loss: Tensor = loss_fn(logits, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#         total_loss += loss.item()
    
#     train_loss = total_loss / len(loader)
#     return train_loss

def pre_stroke_pair_process(pre_stroke_num, pre_stroke, pred_stage1):
    """
    根據 pre_stroke_num 處理 pre_stroke 和 pred_stage1，生成 pre_stroke_pair。
    
    :param pre_stroke_num: 用來決定從 pre_stroke 和 pred_stage1 中選取的元素數量
    :param pre_stroke: 用來生成第一部分 pre_stroke_pair 的張量
    :param pred_stage1: 用來生成第二部分 pre_stroke_pair 的預測結果
    :return: 返回處理後的 pre_stroke_pair
    """
    
    # 1. 取得 pre_stroke 前 pre_stroke_num 個元素
    first_prev = pre_stroke[:pre_stroke_num].detach().clone().view(-1)
    # print(f"first_prev: {first_prev}")
    # 2. 計算需要從 pred_stage1 中扣掉的數量（pre_stroke_num - 1）
    remove_cnt = pre_stroke_num - 1
   

    # 3. 如果 pred_stage1 長度小於等於扣掉的數量，則返回一個空 tensor
    
        # 4. 扣掉 pred_stage1 的前 (pre_stroke_num - 1) 個數字，保留其餘部分
    pred_stage1_adjusted = pred_stage1[remove_cnt:].detach().clone()

    # 5. 合併 pre_stroke 和 pred_stage1_adjusted
    pre_stroke_pair = torch.cat([first_prev, pred_stage1_adjusted])
    # print(f"pre_stroke_pair before trim: {pre_stroke_pair.shape}")
    # 6. 最後去掉 pre_stroke_pair 的最後一個元素
    
    pre_stroke_pair = pre_stroke_pair[:-1]
    
    return pre_stroke_pair

def get_pre_stroke_num(batch_idx, max_batch_idx, min_pre_stroke_num=1, max_pre_stroke_num=5):
    """
    根據 batch_idx 來調整 pre_stroke_num，並保持在 min 和 max 之間，指數增長方式
    """
    progress = batch_idx / max_batch_idx  # 計算訓練進度（0 到 1）
    pre_stroke_num = int(min_pre_stroke_num + (max_pre_stroke_num - min_pre_stroke_num) * (progress ** 2))  # 指數增長
    if pre_stroke_num == 0:
        pre_stroke_num = 1
    
    return pre_stroke_num


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
#     prev_last = torch.tensor([0], device=device)  # 用於記錄前一個 batch 最後的預測
#     all_preds_stage1 = []  # 存每個 batch 的預測結果（tensor）
#     pre_stroke_num = 1

#     for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_stroke) in enumerate(loader):
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         labels: Tensor = labels.to(device)
#         pre_stroke: Tensor = pre_stroke.to(device)
       
#         # ---------- 第一階段 ----------
#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
#         pre_stroke_zero = torch.zeros_like(pre_stroke)
#         logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_stroke_zero)
#         pred_stage1 = torch.argmax(logits_stage1, dim=1)

#         # 儲存第一階段結果
#         all_preds_stage1.append(pred_stage1.detach().clone())
#         pre_stroke_num = get_pre_stroke_num(batch_idx, len(loader)-1, 1, 2)
#         # print(f"\n[Batch {batch_idx}] pred_stage1: {pred_stage1.tolist()}")

#         # ---------- pre_stroke_pair 生成 ----------
#         # 檢查是否包含 [14, 15, 29, 30]
#         special_classes = [0, 14, 15, 29, 30]

#         # if batch_idx == 0:
#         #     # 第一個 batch 用 0 當前一拍
#         #     pre_stroke_pair = torch.cat([
#         #         torch.tensor([0], device=device),
#         #         pred_stage1[:-1]
#         #     ])
#         # else:
#         #     # 其他 batch 若有特殊類別 → 前一拍設為 0，否則用上一批的最後一個
#         #    pre_stroke_pair = torch.cat([prev_last.detach().clone(), pred_stage1])

#         #    pre_stroke_pair = pre_stroke_pair[:-1]
#         #    print(f"[Batch {batch_idx}] pre_stroke_pair: {pre_stroke_pair.tolist()}")
       
#         # first_prev = pre_stroke[0].detach().clone().view(-1)
#         # first_prev = pre_stroke[:pre_stroke_num].detach().clone().view(-1)
#         # remove_cnt = pre_stroke_num - 1
#         # pred_stage1_adjusted = pred_stage1[remove_cnt:].detach().clone()
#         # pre_stroke_pair = torch.cat([first_prev, pred_stage1_adjusted])
#         # pre_stroke_pair = pre_stroke_pair[:-1]
#         pre_stroke_pair = pre_stroke_pair_process(pre_stroke_num, pre_stroke, pred_stage1).view(-1, 1)
#         # pre_stroke_pair = torch.cat([first_prev, pred_stage1])
#         # pre_stroke_pair = pre_stroke_pair[:-1]
      
#         for idx in range(1, len(pre_stroke_pair)):
#             if pre_stroke_pair[idx].item() in special_classes:  
#                 pre_stroke_pair[idx-1] = 0
#             # pre_stroke_pair = torch.cat([
#             #     torch.tensor([0], device=device) if contains_special else prev_last,
#             #     pred_stage1[:-1]
#             # ])
       
#         # print(f"[Batch {batch_idx}] pre_stroke_num: {pre_stroke_num}")
#         # print(f"[Batch {batch_idx}] pre_stroke_pair after trim: {pre_stroke_pair.shape}")

      
#         # ---------- 第二階段 ----------
#         logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_stroke_pair)
#         loss: Tensor = loss_fn(logits_stage2 , labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#         total_loss += loss.item()
    
#     train_loss = total_loss / len(loader)
#     return train_loss

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
    prev_last = torch.tensor([0], device=device)  # 用於記錄前一個 batch 最後的預測
    all_preds_stage1 = []  # 存每個 batch 的預測結果（tensor）
  

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        pos: Tensor = pos.to(device)
        shuttle: Tensor = shuttle.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        pre_stroke: Tensor = pre_stroke.to(device)
       
        # ---------- 第一階段 ----------
        if n_bones == 0:
            human_pose = random_shift_fn(human_pose)
        else:
            joints = human_pose[:, :, :, :-n_bones, :].contiguous()
            bones = human_pose[:, :, :, -n_bones:, :]
            joints = random_shift_fn(joints)
            human_pose = torch.cat([joints, bones], dim=-2)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        pre_stroke_zero = torch.zeros_like(pre_stroke)
        logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_stroke_zero)
        pred_stage1 = torch.argmax(logits_stage1, dim=1)

        # 儲存第一階段結果
        all_preds_stage1.append(pred_stage1.detach().clone())
       
        # print(f"\n[Batch {batch_idx}] pred_stage1: {pred_stage1.tolist()}")

        # ---------- pre_stroke_pair 生成 ----------
        # 檢查是否包含 [14, 15, 29, 30]
        special_classes = [0, 14, 15, 29, 30]

        # if batch_idx == 0:
        #     # 第一個 batch 用 0 當前一拍
        #     pre_stroke_pair = torch.cat([
        #         torch.tensor([0], device=device),
        #         pred_stage1[:-1]
        #     ])
        # else:
        #     # 其他 batch 若有特殊類別 → 前一拍設為 0，否則用上一批的最後一個
        #    pre_stroke_pair = torch.cat([prev_last.detach().clone(), pred_stage1])

        #    pre_stroke_pair = pre_stroke_pair[:-1]
        #    print(f"[Batch {batch_idx}] pre_stroke_pair: {pre_stroke_pair.tolist()}")
       
        first_prev = pre_stroke[0].detach().clone().view(-1)
        # first_prev = pre_stroke[:pre_stroke_num].detach().clone().view(-1)
        # remove_cnt = pre_stroke_num - 1
        # pred_stage1_adjusted = pred_stage1[remove_cnt:].detach().clone()
        # pre_stroke_pair = torch.cat([first_prev, pred_stage1_adjusted])
        # pre_stroke_pair = pre_stroke_pair[:-1]
        # pre_stroke_pair = pre_stroke_pair_process(pre_stroke_num, pre_stroke, pred_stage1).view(-1, 1)
        pre_stroke_pair = torch.cat([first_prev, pred_stage1])
        pre_stroke_pair = pre_stroke_pair[:-1]
      
        for idx in range(1, len(pre_stroke_pair)):
            if pre_stroke_pair[idx].item() in special_classes:  
                pre_stroke_pair[idx-1] = 0
            # pre_stroke_pair = torch.cat([
            #     torch.tensor([0], device=device) if contains_special else prev_last,
            #     pred_stage1[:-1]
            # ])
       
        # print(f"[Batch {batch_idx}] pre_stroke_num: {pre_stroke_num}")
        # print(f"[Batch {batch_idx}] pre_stroke_pair after trim: {pre_stroke_pair.shape}")

      
        # ---------- 第二階段 ----------
        logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_stroke_pair)
        loss: Tensor = loss_fn(logits_stage2 , labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    
    train_loss = total_loss / len(loader)
    return train_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loss_fn,
    loader: DataLoader,
    device: torch.device
):
    model.eval()
    total_loss = 0.0
    cum_tp = torch.zeros(hyp.n_classes)
    cum_tn = torch.zeros(hyp.n_classes)
    cum_fp = torch.zeros(hyp.n_classes)
    cum_fn = torch.zeros(hyp.n_classes)

    prev_last = torch.tensor([0], device=device)  # 用於記錄前一個 batch 最後的預測
    all_preds_stage1 = []  # 存每個 batch 的預測結果（tensor）

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_stroke) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        pos: Tensor = pos.to(device)
        shuttle: Tensor = shuttle.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        pre_stroke: Tensor = pre_stroke.to(device)

        # ---------- 第一階段 ----------
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        pre_stroke_zero = torch.zeros_like(pre_stroke)
        logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_stroke_zero)
        pred_stage1 = torch.argmax(logits_stage1, dim=1)

        # 儲存第一階段結果
        all_preds_stage1.append(pred_stage1.detach().clone())

        # print(f"\n[Batch {batch_idx}] pred_stage1: {pred_stage1.tolist()}")

        # ---------- pre_stroke_pair 生成 ----------
        # 檢查是否包含 [14, 15, 29, 30]
        special_classes = [0, 14, 15, 29, 30]

        if batch_idx == 0:
            # 第一個 batch 用 0 當前一拍
            pre_stroke_pair = torch.cat([
                torch.tensor([0], device=device),
                pred_stage1[:-1]
            ])
        else:
            # 其他 batch 若有特殊類別 → 前一拍設為 0，否則用上一批的最後一個
           pre_stroke_pair = torch.cat([prev_last.detach().clone(), pred_stage1])

           pre_stroke_pair = pre_stroke_pair[:-1]
        #    print(f"[Batch {batch_idx}] pre_stroke_pair: {pre_stroke_pair.tolist()}")
           for idx in range(1, len(pre_stroke_pair)):
            if pre_stroke_pair[idx].item() in special_classes:  
                pre_stroke_pair[idx-1] = 0
            # pre_stroke_pair = torch.cat([
            #     torch.tensor([0], device=device) if contains_special else prev_last,
            #     pred_stage1[:-1]
            # ])
        
        

        # 更新 prev_last（記錄此 batch 最後一個預測）
        prev_last = pred_stage1[-1:].detach().clone()
        # print(f"[Batch {batch_idx}] pre_last: {prev_last.tolist()}")
        # print(f"[Batch {batch_idx}] pre_stroke_pair: {pre_stroke_pair.tolist()}")

        # ---------- 第二階段 ----------
        logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_stroke_pair)

        pred_stage2 = torch.argmax(logits_stage2, dim=1)
    #     print(f"[Stage2 Batch {batch_idx}] Label前3:", labels[:3].cpu().numpy().tolist(),
    #   "→ Pred前3:", pred_stage2[:3].cpu().numpy().tolist())
        
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


@torch.no_grad()
def test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
):
    model.eval()
    pred_ls = []
    labels_ls = []
    misclassified_reports = []  # 用來收集錯誤的預測與真實標籤配對

    all_preds_stage1 = []  # 儲存每個 batch 第一階段預測
    prev_last = torch.tensor([0], device=device)  # 上一批的最後一筆預測

    # with torch.no_grad():
    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_stroke) in enumerate(loader):
        # ---------- 資料搬移 ----------
        human_pose: Tensor = human_pose.to(device)
        pos: Tensor = pos.to(device)
        shuttle: Tensor = shuttle.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        pre_stroke: Tensor = pre_stroke.to(device)

        # ---------- 第一階段 ----------
        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        pre_stroke_zero = torch.zeros_like(pre_stroke)
        logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_stroke_zero)
        pred_stage1 = torch.argmax(logits_stage1, dim=1)

        all_preds_stage1.append(pred_stage1.detach().clone())

            # print(f"\n[Batch {batch_idx}] pred_stage1: {pred_stage1.tolist()}")

            # ---------- pre_stroke_pair 生成 ----------
        special_classes = [0, 14, 15, 29, 30]

        if batch_idx == 0:
            # 第一個 batch 用 0 當前一拍
            pre_stroke_pair = torch.cat([
                torch.tensor([0], device=device),
                pred_stage1[:-1]
            ])
        else:
            # 其他 batch 若有特殊類別 → 前一拍設為 0，否則用上一批的最後一個
            pre_stroke_pair = torch.cat([prev_last.detach().clone(), pred_stage1])

            pre_stroke_pair = pre_stroke_pair[:-1]
                #    print(f"[Batch {batch_idx}] pre_stroke_pair: {pre_stroke_pair.tolist()}")
            for idx in range(1, len(pre_stroke_pair)):
                if pre_stroke_pair[idx].item() in special_classes:  
                    pre_stroke_pair[idx-1] = 0
            # pre_stroke_pair = torch.cat([
            #     torch.tensor([0], device=device) if contains_special else prev_last,
            #     pred_stage1[:-1]
            # ])
        
        

        # 更新 prev_last（記錄此 batch 最後一個預測）
        prev_last = pred_stage1[-1:].detach().clone()

            # print(f"[Batch {batch_idx}] pre_stroke_pair: {pre_stroke_pair.tolist()}")

        # ---------- 第二階段 ----------
        logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_stroke_pair)
        pred_stage2 = torch.argmax(logits_stage2, dim=1)

        # ---------- 錯誤統計 ----------
        is_misclassified = (pred_stage2 != labels)
        if is_misclassified.any():
            misclassified_pred = pred_stage2[is_misclassified].tolist()
            misclassified_labels = labels[is_misclassified].tolist()
            misclassified_pairs = list(zip(misclassified_labels, misclassified_pred))
            misclassified_reports.extend(misclassified_pairs)

        pred_ls.append(pred_stage2.cpu())
        labels_ls.append(labels.cpu())

    return torch.cat(pred_ls), torch.cat(labels_ls), misclassified_reports
       


    # return torch.cat(pred_ls), torch.cat(labels_ls)
# def test_groundtruth(model: nn.Module, loader, device):
#     model.eval()
#     pred_ls, labels_ls = [], []
#     misclassified_reports = []

#     for (human_pose, pos, shuttle), video_len, labels, pre_stroke in loader:
#         human_pose: Tensor = human_pose.to(device)
#         pos: Tensor = pos.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         video_len: Tensor = video_len.to(device)
#         pre_stroke: Tensor = pre_stroke.to(device)

#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)


#         logits = model(human_pose, pos, shuttle, video_len, pre_stroke)
#         pred = torch.argmax(logits, dim=1)  # (N,)


        
#         # ========= 記錄 =========
#         is_misclassified = (pred.cpu() != labels)
#         if is_misclassified.any():
#             misclassified_pred = pred[is_misclassified].tolist()
#             misclassified_labels = labels[is_misclassified].tolist()
#             misclassified_pairs = list(zip(misclassified_labels, misclassified_pred))
#             misclassified_reports.extend(misclassified_pairs)

#         pred_ls.append(pred.cpu())
#         labels_ls.append(labels)

#     return torch.cat(pred_ls), torch.cat(labels_ls), misclassified_reports



@torch.no_grad()
def test_topk(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    k=2
):
    model.eval()
    pred_ls = []
    labels_ls = []

    prev_last = torch.tensor([0], device=device)  # 記錄上一批最後一個預測

    # with torch.no_grad():
    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, pre_stroke) in enumerate(loader):
        # ---------- 資料搬移 ----------
        human_pose = human_pose.to(device)
        pos = pos.to(device)
        shuttle = shuttle.to(device)
        video_len = video_len.to(device)
        labels = labels.to(device)
        pre_stroke = pre_stroke.to(device)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)

        # ---------- 第一階段 ----------
        pre_stroke_zero = torch.zeros_like(pre_stroke)
        logits_stage1 = model(human_pose, pos, shuttle, video_len, pre_stroke_zero)
        pred_stage1 = torch.argmax(logits_stage1, dim=1)

        # ---------- pre_stroke_pair 生成 ----------
        special_classes = [0, 14, 15, 29, 30]

        if batch_idx == 0:
            # 第一個 batch 用 0 當前一拍
            pre_stroke_pair = torch.cat([torch.tensor([0], device=device), pred_stage1[:-1]])
        else:
            # 其他 batch 先接 prev_last
            pre_stroke_pair = torch.cat([prev_last.detach().clone(), pred_stage1])

            pre_stroke_pair = pre_stroke_pair[:-1]

            # 特殊類別處理
            for idx in range(1, len(pre_stroke_pair)):
                if pre_stroke_pair[idx].item() in special_classes:
                    pre_stroke_pair[idx-1] = 0

        # ---------- 更新 prev_last ----------
        prev_last = pred_stage1[-1:].detach().clone()

        # ---------- 第二階段 ----------
        logits_stage2 = model(human_pose, pos, shuttle, video_len, pre_stroke_pair)
            
        # Top-k 預測
        _, pred = torch.topk(logits_stage2, k=k, dim=1)
        pred_ls.append(pred.cpu())
        labels_ls.append(labels.cpu())

            # 可選 debug
            # print(f"Batch {batch_idx} pre_stroke_pair[:5]:", pre_stroke_pair[:5].tolist(), "prev_last:", prev_last.item())

    return torch.cat(pred_ls), torch.cat(labels_ls)






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
            = prepare_npy_collated_for_tempose_loaders(
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
        - 'TemPose_TFwithOneStroke' (about 1.92M)
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

            case 'TemPose_TFwithOneStroke':
                net = TemPose_TFwithOneStroke(
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
    task.get_network_architecture(model_name='TemPose_TFwithOneStroke', in_channels=2)
    task.seek_network_weights(model_info='rally_data_npy_collated', serial_no=11)
    task.test(show_details=False, show_confusion_matrix=False, show_misclassified_reports = False)
    task.test_topk_acc(k=2)
    
    # task.compare_pred_gt_on_specific_type(
    #     Path("C:/MyResearch/stroke_classification/dataset_npy/test/Top_點扣")
    # )
