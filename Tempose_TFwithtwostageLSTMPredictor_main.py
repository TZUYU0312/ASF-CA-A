import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

from transformers import get_cosine_schedule_with_warmup
import copy
import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections import namedtuple
import time
from datetime import timedelta

from dataset import get_bone_pairs, prepare_npy_collated_for_temposeRNN_loaders, \
                    RandomTranslation_batch, Dataset_npy
from TemposewithRNNPredictor import Tempose_TFwithtwostageBiLSTMPredictor, Tempose_TFwithtwostageSingleLSTMPredictor
from utils import show_f1_results, plot_confusion_matrix


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
    n_classes=32,
    seq_len=30
)
PADDING_TOKEN_ID = 31



def pad_and_repeat_rally(sequence, rally_len, max_len, padding_value):
        """將序列填充到 max_len，並重複 rally_len 次。"""
        padding_needed = max_len - len(sequence)
        # print('padding_needed:', padding_needed)
        padded_sequence = sequence + [padding_value] * padding_needed
        if len(padded_sequence) != max_len:
            padded_sequence = sequence[:max_len]
        # print('padded_sequence:', padded_sequence, len(padded_sequence))
        repeated_rally = [padded_sequence] * rally_len
        return repeated_rally

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

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, ballrounds, rally) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        ballrounds: Tensor = ballrounds.to(device)
        rally: Tensor = rally.to(device)
       
        # print('ballrounds:', ballrounds, ballrounds.shape)
        # print('rally:', rally, rally.shape)
        qlen = labels.shape[0]
       
        
        if n_bones == 0:
            human_pose = random_shift_fn(human_pose)
        else:
            joints = human_pose[:, :, :, :-n_bones, :].contiguous()
            bones = human_pose[:, :, :, -n_bones:, :]

            joints = random_shift_fn(joints)
            human_pose = torch.cat([joints, bones], dim=-2)

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)
        # ===== 第一階段 (rally 全 0) =====
        if batch_idx  == 0:
            logits_stage1 = model(human_pose, shuttle,pos, video_len, ballrounds ,rally)
        else:
            ballrounds_zero = torch.zeros_like(ballrounds)
            # rally_zero = torch.zeros_like(rally)
            rally_zero = None
            logits_stage1 = model(human_pose, shuttle,pos, video_len, ballrounds_zero ,rally_zero)

        pred_stage1 = torch.argmax(logits_stage1, dim=1)
        # print('batch_idx:', batch_idx)
        # print('pred_stage1:', pred_stage1, pred_stage1.shape)
        target_values = torch.tensor([0, 14, 15, 29, 30]).to(device)
        mask = torch.isin(pred_stage1, target_values)
        indices = torch.nonzero(mask, as_tuple=False).squeeze()
        if indices.numel() <= 2 or (pred_stage1.shape[0] != hyp.batch_size):
            logits = model(human_pose, shuttle, pos, video_len, ballrounds, rally)
        else:
            # 3. 第一個符合條件的索引 (第幾個數字)
            # 由於 Tensor 索引從 0 開始，所以第 'indices[0]' 個數字
            first_index = indices[0].item()
            # 實際是第幾個數字 (從 1 開始計數)
          

            # 4. 最後一個符合條件的索引
            last_index = indices[-1].item()
            # 實際是第幾個數字 (從 1 開始計數)
          
            extracted_segment = pred_stage1[first_index:last_index ].tolist()
            extracted_segment_rally = rally[first_index:last_index+1].tolist()
            first_rally = rally[:first_index]
           
            first_ballrounds = ballrounds[:first_index]
            second_rally = rally[last_index - qlen : ]
            second_ballrounds = ballrounds[ last_index - qlen :]
            # print('first_rally:', first_rally, first_rally.shape)
            # print('first_ballround:',  first_ballrounds.shape)
            # print('extracted_segment:',  len(extracted_segment))
            # print('second_rally:', second_rally, second_rally.shape)
            # print('second_ballround:',  second_ballrounds.shape)
            # 先包裝rally ballrounds
            found_indices = []
            special_classes = [0, 14, 15, 29, 30]

            for index, value in enumerate(extracted_segment):
            # 檢查當前元素的值是否在 special_classes 列表中
                if value in special_classes:
                    # 如果是，將索引和值一起記錄下來 (可選，這裡只記錄索引)
                    found_indices.append(index)
            temp_ballrounds = []
            temp_rally = []
            current_count = 1  # 計數器從 1 開始
            for i in range(len(extracted_segment)):

                if i in found_indices:
                    # 如果是分界點，重設計數器為 1
                    current_count = 1
                if current_count > 65:
                    current_count = 1
                
                # 將計數值添加到列表中 (以 [數字] 的格式，雖然 Python 列表中無法直接表示 [[1]]，但我們使用單個數字)
                temp_ballrounds.append(current_count)
                
                # 計數器遞增
                current_count += 1
            
            temp_ballrounds = torch.tensor(temp_ballrounds).view(-1, 1).to(device)
            ballrounds_new = torch.cat(
                (first_ballrounds, temp_ballrounds, second_ballrounds),
                dim=0 # 沿著第 0 維度 (行) 合併
            )
            # print('ballrounds_new:', len(ballrounds_new), ballrounds_new)


            num_pairs = len(found_indices) 
            found_indices_sequence = found_indices + [len(extracted_segment)]  # 加上結尾索引，方便處理最後一段
            # print('found_indices_sequence:', found_indices_sequence)
            if num_pairs > 1:
                # 如果沒有找到有效的回合，則跳過後續處理

                for i in range(num_pairs):
                    
                    start_idx = found_indices_sequence[i]      # 當前回合的起始索引 (例如 0, 11)
                    end_idx = found_indices_sequence[i+1]      # 當前回合的結束索引 (例如 11, 17)
                    # print('start_idx:', start_idx, 'end_idx:', end_idx)
                    rally_len = end_idx - start_idx   # 回合的長度 (例如 11, 6)
                    
                    # 取出當前回合的序列：temp[start_idx] 到 temp[end_idx - 1]
                    rally_sequence = extracted_segment[start_idx:end_idx] 
                    # print('rally_sequence:', rally_sequence, len(rally_sequence))
                    
                    # # 填充並重複
                    # print('rally_len:', rally_len)
                    # print('rally_sequence:', rally_sequence, len(rally_sequence))   
                    repeated_rally = pad_and_repeat_rally(rally_sequence, rally_len, 65, PADDING_TOKEN_ID)
                    # print('repeated_rally:', repeated_rally, len(repeated_rally))
                    temp_rally.extend(repeated_rally)

            elif num_pairs == 1:
                # print("Only one special class found, processing accordingly.")
                for i in range(num_pairs):
                    start_idx = found_indices_sequence[i]      # 當前回合的起始索引 (例如 0, 11)
                    end_idx = found_indices_sequence[i+1]      # 當前回合的結束索引 (例如 11, 17)
                    # print('start_idx:', start_idx, 'end_idx:', end_idx)
                    rally_len = end_idx - start_idx   # 回合的長度 (例如 11, 6)
                        
                    # 取出當前回合的序列：temp[start_idx] 到 temp[end_idx - 1]
                    rally_sequence = extracted_segment[start_idx:end_idx] 
                        # print('rally_sequence:', rally_sequence, len(rally_sequence))
                        
                        # # 填充並重複
                    repeated_rally = pad_and_repeat_rally(rally_sequence, rally_len, 65, PADDING_TOKEN_ID)
                    # repeated_rally = [rally_sequence] * len(extracted_segment)
                    # print('repeated_rally:', repeated_rally, len(repeated_rally))
                    temp_rally.extend(repeated_rally)
            
            else:
                # print("No special classes found, processing entire segment as one rally.")                    
                start_idx = 0      # 當前回合的起始索引 (例如 0, 11)
                end_idx = len(extracted_segment)      # 當前回合的結束索引 (例如 11, 17)
                # print('start_idx:', start_idx, 'end_idx:', end_idx)
                rally_len = end_idx - start_idx   # 回合的長度 (例如 11, 6)
                    
                # 取出當前回合的序列：temp[start_idx] 到 temp[end_idx - 1]
                rally_sequence = extracted_segment[start_idx:end_idx] 
                    # print('rally_sequence:', rally_sequence, len(rally_sequence))
                    
                    # # 填充並重複
                repeated_rally = pad_and_repeat_rally(rally_sequence, rally_len, 65, PADDING_TOKEN_ID)
                # repeated_rally = [rally_sequence] * len(extracted_segment)
                # print('repeated_rally:', repeated_rally, len(repeated_rally))
                temp_rally.extend(repeated_rally)
           

            # print('temp_rally:', len(temp_rally))
            # print(f"DEBUG: Type of temp_rally: {type(temp_rally)}")
            for i, seq in enumerate(temp_rally):
                if len(seq) != 65:
                    print(f"⚠️ Rally {i} has length {len(seq)} instead of 65")
                    print(f"Sequence: {seq}")
            temp_rally = torch.tensor(temp_rally).to(device)
            # print('temp_rally tensor:', temp_rally, temp_rally.shape)
            # print('first_rally tensor:', first_rally, first_rally.shape)
            # print('second_rally tensor:', second_rally, second_rally.shape)
           
            # rally_new = torch.cat(
            #     (first_rally, temp_rally, second_rally),
            #     dim=0 # 沿著第 0 維度 (行) 合併
            # )

            # 假設 first_rally, temp_rally, second_rally 都是 torch.Tensor
            rallies = []

            if first_rally.numel() > 0:   # 檢查是否不是空 tensor
                rallies.append(first_rally)
            if temp_rally.numel() > 0:
                rallies.append(temp_rally)
            if second_rally.numel() > 0:
                rallies.append(second_rally)

            if len(rallies) > 0:
                rally_new = torch.cat(rallies, dim=0)
            # print('temp_rally:', temp_rally, temp_rally.shape)
            # print('rally_new:', rally_new, rally_new.shape)
            if ballrounds_new.shape[0] != rally_new.shape[0]:
                # print(f"⚠️ Warning: ballrounds shape {ballrounds_new.shape} does not match x shape {x.shape}. Adjusting ballrounds.")
                # print(f"⚠️ Warning: Original ballrounds: {ballrounds_new}")
                rally_new = torch.cat((rally_new, rally_new[-1:].clone()), dim=0)

           
            #     # print(f"⚠️ Warning: Adjusted x shape: {x.shape}")
            #     # print(f"⚠️ Warning: Adjusted ballrounds shape: {ballrounds.shape}")
            # elif ballrounds_new.shape[0] < rally_new.shape[0]:
            #     rally_new = rally_new[:(ballrounds_new.shape[0] - rally_new.shape[0])] 

            logits = model(human_pose, shuttle, pos, video_len, ballrounds_new, rally_new)

        loss: Tensor = loss_fn(logits, labels)

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
    qlen_list = []
    temp_pred = []

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, ballrounds, rally) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        ballrounds: Tensor = ballrounds.to(device)
        rally: Tensor = rally.to(device)
   

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)  # for faster dataset version

        # ===== 第一階段 (rally 全 0) =====
       
        ballrounds_zero = torch.zeros_like(ballrounds)
        # rally_zero = torch.zeros_like(rally)
        rally_zero = None
        logits_stage1 = model(human_pose, shuttle,pos, video_len, ballrounds_zero ,rally_zero)
        pred_stage1 = torch.argmax(logits_stage1, dim=1)
        all_preds_stage1.append(pred_stage1.detach().clone())
        # qlen_list.extend(len(loader))
        temp_pred.extend(pred_stage1.tolist())
    # print('temp:', len(temp_pred), temp_pred)
    # print('all_preds_stage1 shape:', len(all_preds_stage1[0]))
        
     
    # ===== 第二階段 (再用 pred_stage1) =====
    # 先包裝rally ballrounds
    found_indices = []
    special_classes = [0, 14, 15, 29, 30]
    for index, value in enumerate(temp_pred):
    # 檢查當前元素的值是否在 special_classes 列表中
        if value in special_classes:
            # 如果是，將索引和值一起記錄下來 (可選，這裡只記錄索引)
            found_indices.append(index)
    temp_ballrounds = []
    temp_rally = []
    current_count = 1  # 計數器從 1 開始

    for i in range(len(temp_pred)):
       
        
        # 判斷當前索引是否是新一回合的開始 (即索引是否在 found_indices 中)
        if len(found_indices) < 2:
            for j in range(len(temp_pred)):
                count_value = (j % 65) + 1
                temp_ballrounds.append(count_value)

        if i in found_indices:
            # 如果是分界點，重設計數器為 1
            current_count = 1

        if current_count > 65:
            current_count = 1

        # 將計數值添加到列表中 (以 [數字] 的格式，雖然 Python 列表中無法直接表示 [[1]]，但我們使用單個數字)
        temp_ballrounds.append(current_count)
        
        # 計數器遞增
        current_count += 1
    # print('temp_ballrounds:', len(temp_ballrounds), temp_ballrounds)

    num_pairs = len(found_indices) - 1

    if num_pairs >= 1:
        # 如果沒有找到有效的回合，則跳過後續處理

        for i in range(num_pairs):
            
            start_idx = found_indices[i]      # 當前回合的起始索引 (例如 0, 11)
            end_idx = found_indices[i+1]      # 當前回合的結束索引 (例如 11, 17)
            
            rally_len = end_idx - start_idx   # 回合的長度 (例如 11, 6)
            
            # 取出當前回合的序列：temp[start_idx] 到 temp[end_idx - 1]
            rally_sequence = temp_pred[start_idx:end_idx] 
            
            # 填充並重複
            repeated_rally = pad_and_repeat_rally(rally_sequence, rally_len, 65, PADDING_TOKEN_ID)
            temp_rally.extend(repeated_rally)
    else:
        rally_count = 0
        temp = copy.deepcopy(temp_pred)
        while len(temp) > 0:
            rally_count += 1
            current_len = min(65, len(temp))
            rally_sequence = temp[:current_len]
            # # print('rally_sequence:', rally_sequence, len(rally_sequence))
            # if current_len == 65:
            #     # 完整回合：重複 qlen 次
            #     repeat_times = 65
            # else:
            #     # 剩餘回合：重複剩餘的長度次 (即 current_len 次)
            #     repeat_times = current_len
            # repeated_rally = [rally_sequence] * repeat_times
            repeated_rally = pad_and_repeat_rally(
                rally_sequence,       # 目前這一段序列
                rally_len=current_len,  # 要重複幾次
                max_len=65,             # 固定輸出長度
                padding_value=31        # 你的 PAD token
            )
            temp_rally.extend(repeated_rally)
            temp = temp[current_len:]
        # print('temp_rally:', len(temp_rally))
    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, ballrounds, rally) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        
        qlen = len(all_preds_stage1[batch_idx])
        subset_list_ballrounds = temp_ballrounds[:qlen]
        subset_list_rally = temp_rally[:qlen]
        temp_tensor = torch.tensor(subset_list_ballrounds)
        # 步驟 b: 重塑為列向量 (使用 view(-1, 1))
        ballrounds = temp_tensor.view(-1, 1).to(device)
        temp_ballrounds = temp_ballrounds[qlen:]
        lengths = [len(seq) for seq in subset_list_rally]
        # print("subset_list_rally lengths:", lengths)
        # print("unique lengths:", set(lengths))
        rally = torch.tensor(subset_list_rally).to(device)
        # print('rally tensor:', rally, rally.shape)
        temp_rally = temp_rally[qlen:]

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)  # for faster dataset version

        # ===== 第二階段 (再用 pred_stage1) =====
        if ballrounds.shape[0] != rally.shape[0]:
                # print(f"⚠️ Warning: ballrounds shape {ballrounds_new.shape} does not match x shape {x.shape}. Adjusting ballrounds.")
                # print(f"⚠️ Warning: Original ballrounds: {ballrounds_new}")
                rally = torch.cat((rally, rally[-1:].clone()), dim=0)

        logits_stage2 = model(human_pose, shuttle,pos, video_len,ballrounds,rally )
        loss: Tensor = loss_fn(logits_stage2, labels)
        total_loss += loss.item()

        # ===== 評估指標 =====
        pred = F.one_hot(torch.argmax(logits_stage2, dim=1), hyp.n_classes).bool()
        labels_onehot = F.one_hot(labels, hyp.n_classes).bool()
        

        # logits = model(human_pose, shuttle, pos, video_len, ballrounds, rally, pre_stroke)
        # loss: Tensor = loss_fn(logits, labels)
        # total_loss += loss.item()

        # pred = F.one_hot(torch.argmax(logits, dim=1), hyp.n_classes).bool()
        # labels_onehot = F.one_hot(labels, hyp.n_classes).bool()

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
    all_preds_stage1 = []  # 用來儲存所有 batch 的預測結果
    qlen_list = []
    temp_pred = []

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, ballrounds, rally) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        ballrounds: Tensor = ballrounds.to(device)
        rally: Tensor = rally.to(device)
   

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)  # for faster dataset version

        # ===== 第一階段 (rally 全 0) =====
       
        ballrounds_zero = torch.zeros_like(ballrounds)
        rally_zero = None
        # rally_zero = torch.zeros_like(rally)
        logits_stage1 = model(human_pose, shuttle,pos, video_len, ballrounds_zero ,rally_zero)
        pred_stage1 = torch.argmax(logits_stage1, dim=1)
        all_preds_stage1.append(pred_stage1.detach().clone())
        # qlen_list.extend(len(loader))
        temp_pred.extend(pred_stage1.tolist())

    # ===== 第二階段 (再用 pred_stage1) =====
    # 先包裝rally ballrounds
    found_indices = []
    special_classes = [0, 14, 15, 29, 30]
    for index, value in enumerate(temp_pred):
    # 檢查當前元素的值是否在 special_classes 列表中
        if value in special_classes:
            # 如果是，將索引和值一起記錄下來 (可選，這裡只記錄索引)
            found_indices.append(index)
    temp_ballrounds = []
    temp_rally = []
    current_count = 1  # 計數器從 1 開始
    for i in range(len(temp_pred)):
       
        
        # 判斷當前索引是否是新一回合的開始 (即索引是否在 found_indices 中)
        if len(found_indices) < 2:
            for j in range(len(temp_pred)):
                count_value = (j % 65) + 1
                temp_ballrounds.append(count_value)

        if i in found_indices:
            # 如果是分界點，重設計數器為 1
            current_count = 1

        if current_count > 65:
            current_count = 1
        
        # 將計數值添加到列表中 (以 [數字] 的格式，雖然 Python 列表中無法直接表示 [[1]]，但我們使用單個數字)
        temp_ballrounds.append(current_count)
        
        # 計數器遞增
        current_count += 1
    # print('temp_ballrounds:', len(temp_ballrounds), temp_ballrounds)

    num_pairs = len(found_indices) - 1

    if num_pairs >= 1:
        # 如果沒有找到有效的回合，則跳過後續處理

        for i in range(num_pairs):
            
            start_idx = found_indices[i]      # 當前回合的起始索引 (例如 0, 11)
            end_idx = found_indices[i+1]      # 當前回合的結束索引 (例如 11, 17)
            
            rally_len = end_idx - start_idx   # 回合的長度 (例如 11, 6)
            
            # 取出當前回合的序列：temp[start_idx] 到 temp[end_idx - 1]
            rally_sequence = temp_pred[start_idx:end_idx] 
            
            # 填充並重複
            repeated_rally = pad_and_repeat_rally(rally_sequence, rally_len, 65, PADDING_TOKEN_ID)
            temp_rally.extend(repeated_rally)
    else:
        rally_count = 0
        temp = copy.deepcopy(temp_pred)
        while len(temp) > 0:
            rally_count += 1
            current_len = min(65, len(temp))
            rally_sequence = temp[:current_len]
            # # print('rally_sequence:', rally_sequence, len(rally_sequence))
            # if current_len == 65:
            #     # 完整回合：重複 qlen 次
            #     repeat_times = 65
            # else:
            #     # 剩餘回合：重複剩餘的長度次 (即 current_len 次)
            #     repeat_times = current_len
            # repeated_rally = [rally_sequence] * repeat_times
            repeated_rally = pad_and_repeat_rally(
                rally_sequence,       # 目前這一段序列
                rally_len=current_len,  # 要重複幾次
                max_len=65,             # 固定輸出長度
                padding_value=31        # 你的 PAD token
            )
            temp_rally.extend(repeated_rally)
            temp = temp[current_len:]
    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, ballrounds, rally) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        
        qlen = len(all_preds_stage1[batch_idx])
        subset_list_ballrounds = temp_ballrounds[:qlen]
        subset_list_rally = temp_rally[:qlen]
        temp_tensor = torch.tensor(subset_list_ballrounds)
        # 步驟 b: 重塑為列向量 (使用 view(-1, 1))
        ballrounds = temp_tensor.view(-1, 1).to(device)
        temp_ballrounds = temp_ballrounds[qlen:]
        lengths = [len(seq) for seq in subset_list_rally]
        # print("subset_list_rally lengths:", lengths)
        # print("unique lengths:", set(lengths))
        rally = torch.tensor(subset_list_rally).to(device)
        # print('rally tensor:', rally, rally.shape)
        temp_rally = temp_rally[qlen:]

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)  # for faster dataset version

        # ===== 第二階段 (再用 pred_stage1) =====
        if ballrounds.shape[0] != rally.shape[0]:
                # print(f"⚠️ Warning: ballrounds shape {ballrounds_new.shape} does not match x shape {x.shape}. Adjusting ballrounds.")
                # print(f"⚠️ Warning: Original ballrounds: {ballrounds_new}")
                rally = torch.cat((rally, rally[-1:].clone()), dim=0)

        logits_stage2 = model(human_pose, shuttle,pos, video_len,ballrounds,rally )

      

        pred = torch.argmax(logits_stage2 , dim=1)
        
        pred_ls.append(pred.cpu())
        labels_ls.append(labels.cpu())

    return torch.cat(pred_ls), torch.cat(labels_ls)


# def test(
#     model: nn.Module,
#     loader,
#     device
# ):
#     model.eval()
#     pred_ls = []
#     labels_ls = []
#     for (human_pose, pos, shuttle), video_len, labels, ballrounds, rally in loader:
#         human_pose: Tensor = human_pose.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         pos: Tensor = pos.to(device)
#         video_len: Tensor = video_len.to(device)
#         ballrounds: Tensor = ballrounds.to(device)
#         # pre_stroke: Tensor = pre_stroke.to(device)
#         # qlen: Tensor = qlen.to(device)
#         rally: Tensor = rally.to(device)

#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
#         logits = model(human_pose, shuttle, pos, video_len, ballrounds, rally)

#         pred = torch.argmax(logits, dim=1).cpu()
        
#         pred_ls.append(pred)
#         labels_ls.append(labels)

#     return torch.cat(pred_ls), torch.cat(labels_ls)


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
    all_preds_stage1 = []  # 用來儲存所有 batch 的預測結果
    qlen_list = []
    temp_pred = []

    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, ballrounds, rally) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        ballrounds: Tensor = ballrounds.to(device)
        rally: Tensor = rally.to(device)
   

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)  # for faster dataset version

        # ===== 第一階段 (rally 全 0) =====
       
        ballrounds_zero = torch.zeros_like(ballrounds)
        rally_zero = None
        logits_stage1 = model(human_pose, shuttle,pos, video_len, ballrounds_zero ,rally_zero)
        pred_stage1 = torch.argmax(logits_stage1, dim=1)
        all_preds_stage1.append(pred_stage1.detach().clone())
        # qlen_list.extend(len(loader))
        temp_pred.extend(pred_stage1.tolist())

    # ===== 第二階段 (再用 pred_stage1) =====
    # 先包裝rally ballrounds
    found_indices = []
    special_classes = [0, 14, 15, 29, 30]
    for index, value in enumerate(temp_pred):
    # 檢查當前元素的值是否在 special_classes 列表中
        if value in special_classes:
            # 如果是，將索引和值一起記錄下來 (可選，這裡只記錄索引)
            found_indices.append(index)
    temp_ballrounds = []
    temp_rally = []
    current_count = 1  # 計數器從 1 開始
    for i in range(len(temp_pred)):
       
        
        # 判斷當前索引是否是新一回合的開始 (即索引是否在 found_indices 中)
        if len(found_indices) < 2:
            for j in range(len(temp_pred)):
                count_value = (j % 65) + 1
                temp_ballrounds.append(count_value)

        if i in found_indices:
            # 如果是分界點，重設計數器為 1
            current_count = 1

        if current_count > 65:
            current_count = 1
        
        # 將計數值添加到列表中 (以 [數字] 的格式，雖然 Python 列表中無法直接表示 [[1]]，但我們使用單個數字)
        temp_ballrounds.append(current_count)
        
        # 計數器遞增
        current_count += 1
    # print('temp_ballrounds:', len(temp_ballrounds), temp_ballrounds)

    num_pairs = len(found_indices) - 1

    if num_pairs >= 1:
        # 如果沒有找到有效的回合，則跳過後續處理

        for i in range(num_pairs):
            
            start_idx = found_indices[i]      # 當前回合的起始索引 (例如 0, 11)
            end_idx = found_indices[i+1]      # 當前回合的結束索引 (例如 11, 17)
            
            rally_len = end_idx - start_idx   # 回合的長度 (例如 11, 6)
            
            # 取出當前回合的序列：temp[start_idx] 到 temp[end_idx - 1]
            rally_sequence = temp_pred[start_idx:end_idx] 
            
            # 填充並重複
            repeated_rally = pad_and_repeat_rally(rally_sequence, rally_len, 65, PADDING_TOKEN_ID)
            temp_rally.extend(repeated_rally)
    else:
        rally_count = 0
        temp = copy.deepcopy(temp_pred)
        while len(temp) > 0:
            rally_count += 1
            current_len = min(65, len(temp))
            rally_sequence = temp[:current_len]
            # # print('rally_sequence:', rally_sequence, len(rally_sequence))
            # if current_len == 65:
            #     # 完整回合：重複 qlen 次
            #     repeat_times = 65
            # else:
            #     # 剩餘回合：重複剩餘的長度次 (即 current_len 次)
            #     repeat_times = current_len
            # repeated_rally = [rally_sequence] * repeat_times
            repeated_rally = pad_and_repeat_rally(
                rally_sequence,       # 目前這一段序列
                rally_len=current_len,  # 要重複幾次
                max_len=65,             # 固定輸出長度
                padding_value=31        # 你的 PAD token
            )
            temp_rally.extend(repeated_rally)
            temp = temp[current_len:]
    for batch_idx, ((human_pose, pos, shuttle), video_len, labels, ballrounds, rally) in enumerate(loader):
        human_pose: Tensor = human_pose.to(device)
        shuttle: Tensor = shuttle.to(device)
        pos: Tensor = pos.to(device)
        video_len: Tensor = video_len.to(device)
        labels: Tensor = labels.to(device)
        
        qlen = len(all_preds_stage1[batch_idx])
        subset_list_ballrounds = temp_ballrounds[:qlen]
        subset_list_rally = temp_rally[:qlen]
        temp_tensor = torch.tensor(subset_list_ballrounds)
        # 步驟 b: 重塑為列向量 (使用 view(-1, 1))
        ballrounds = temp_tensor.view(-1, 1).to(device)
        temp_ballrounds = temp_ballrounds[qlen:]
        lengths = [len(seq) for seq in subset_list_rally]
        # print("subset_list_rally lengths:", lengths)
        # print("unique lengths:", set(lengths))
        rally = torch.tensor(subset_list_rally).to(device)
        # print('rally tensor:', rally, rally.shape)
        temp_rally = temp_rally[qlen:]

        human_pose = human_pose.view(*human_pose.shape[:-2], -1)  # for faster dataset version

        # ===== 第二階段 (再用 pred_stage1) =====
        if ballrounds.shape[0] != rally.shape[0]:
                # print(f"⚠️ Warning: ballrounds shape {ballrounds_new.shape} does not match x shape {x.shape}. Adjusting ballrounds.")
                # print(f"⚠️ Warning: Original ballrounds: {ballrounds_new}")
                rally = torch.cat((rally, rally[-1:].clone()), dim=0)
            
        logits_stage2 = model(human_pose, shuttle,pos, video_len,ballrounds,rally )

        _, pred = torch.topk(logits_stage2, k=k, dim=1)    
        
        pred_ls.append(pred.cpu())
        labels_ls.append(labels.cpu())

    return torch.cat(pred_ls), torch.cat(labels_ls)

# def test_topk(
#     model: nn.Module,
#     loader,
#     device,
#     k=2
# ):
#     model.eval()
#     pred_ls = []
#     labels_ls = []
#     for (human_pose, pos, shuttle), video_len, labels, ballrounds, rally in loader:
#         human_pose: Tensor = human_pose.to(device)
#         shuttle: Tensor = shuttle.to(device)
#         pos: Tensor = pos.to(device)
#         video_len: Tensor = video_len.to(device)
#         ballrounds: Tensor = ballrounds.to(device)
#         # qlen: Tensor = qlen.to(device)
#         rally: Tensor = rally.to(device)
#         # pre_stroke: Tensor = pre_stroke.to(device)

#         human_pose = human_pose.view(*human_pose.shape[:-2], -1)
#         logits = model(human_pose, shuttle, pos, video_len, ballrounds, rally)

#         _, pred = torch.topk(logits, k=k, dim=1)    
        
#         pred_ls.append(pred.cpu())
#         labels_ls.append(labels)

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

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=PADDING_TOKEN_ID )
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

    def prepare_dataloaders(
        self,
        root_dir: Path,
        pose_style='Jn2B',
        train_partial=1.0
    ):
        self.train_loader, \
        self.val_loader, \
        self.test_loader \
            = prepare_npy_collated_for_temposeRNN_loaders(
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
        - 'BST' (about 1.89M)
        - 'Tempose_TFwithtwostageBiLSTMPredictor' (about 6.09M)
        - 'Tempose_TFwithtwostageSingleLSTMPredictor' (about 4.99M)
       
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
            case 'Tempose_TFwithtwostageBiLSTMPredictor':
                net = Tempose_TFwithtwostageBiLSTMPredictor(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=hyp.n_classes,
                    seq_len=hyp.seq_len,
                    depth_tem=2,
                    depth_inter=1
                )

            case 'Tempose_TFwithtwostageSingleLSTMPredictor':
                net = Tempose_TFwithtwostageSingleLSTMPredictor(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=hyp.n_classes,
                    seq_len=hyp.seq_len,
                    depth_tem=2,
                    depth_inter=1
                )

            case 'Tempose_TFwithtwostageLSTMPredictorSA':
                net = Tempose_TFwithtwostageLSTMPredictorSA(
                    in_dim=(self.n_joints + n_bones * extra) * in_channels,
                    n_class=hyp.n_classes,
                    seq_len=hyp.seq_len,
                    depth_tem=2,
                    depth_inter=1
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

        self.model_name = self.model_name.lower() + model_postfix

        weight_path = Path(f'weight/{self.model_name}.pt')
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

    def test(self, show_details=False, show_confusion_matrix=False):
        pred, gt = test(self.net, self.test_loader, self.device)
        print(f'Test (num_strokes: {len(pred)}) =>')
        valid_classes = torch.unique(gt).tolist()

        f1_score_each = multiclass_f1_score(pred, gt, num_classes=hyp.n_classes, average=None)
        f1_score_each = f1_score_each[valid_classes]
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
    task.get_network_architecture(model_name='Tempose_TFwithtwostageBiLSTMPredictor', in_channels=2)
    task.seek_network_weights(model_info='rally_data_npy_collated', serial_no=11)
    task.test(show_details=False, show_confusion_matrix=False)
    task.test_topk_acc(k=2)
    # task.compare_pred_gt_on_specific_type(
    #     Path("C:/MyResearch/stroke_classification/dataset_npy/test/Top_點扣")
    # )
