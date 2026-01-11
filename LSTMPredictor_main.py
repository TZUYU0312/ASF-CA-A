import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from pathlib import Path
# 假設你已經有這些模型定義，請確保 LSTM_main.py 檔案存在並包含 myLSTM 類別
from LSTMPredictor import LSTMPredictor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from copy import deepcopy
from collections import namedtuple
import time
from datetime import timedelta
from tqdm import tqdm
from torchinfo import summary
from sklearn.model_selection import train_test_split

# ---
## 類別編碼設定 (Class Encoding)
#
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
    early_stop_n_epochs= 200,
    batch_size =64,
    lr = 5e-4,
    seq_len = 30
)

# ---
## 資料處理相關函式 (Data Processing Functions)
#
def get_video_folder_name(is_train):
    """
    從 CSV 檔案讀取影片資料夾名稱和 set 數。

    Args:
        is_train (bool): True 讀取訓練集資料，False 讀取驗證集資料。
    
    Returns:
        pd.DataFrame: 包含 'video' 和 'set' 欄位的 DataFrame。
    """
    if is_train:
        df = pd.read_csv('processed_shuttle_data/set_train.csv')[['video','set']]
    else:
        df = pd.read_csv('processed_shuttle_data/set_val.csv')[['video','set']]
    
    return df

def encode_type(row, class_ls):
    """
    根據球種和球員編碼。
    """
    base_code = class_ls.get(row['type'], class_ls["未知球種"])
    
    if row['player_pos'] == 'Bottom':
        base_code += 15
    return base_code

def get_stroke_types(data, class_ls):
    """
    將 DataFrame 中的 'type' 欄位轉換為數值編碼。
    """
    df_train = data[['rally','type', 'player_pos',]].copy()
    df_train['type_encode'] = df_train.apply(lambda row: encode_type(row, class_ls), axis=1)
    return df_train

# ---
## 資料夾分割函式 (Folder Splitting Function)
#
def split_train_val(root, train_size):
    """
    將影片資料夾分割為訓練集和驗證集，並將相關資訊寫入 CSV 檔案。
    """
    all_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    train_dirs, val_dirs = train_test_split(all_dirs, test_size=train_size, random_state=42)

    output_root = Path("set_data")
    train_target = output_root / "train"
    val_target = output_root / "val"
    train_target.mkdir(parents=True, exist_ok=True)
    val_target.mkdir(parents=True, exist_ok=True)

    train_names = []
    val_names = []

    print(f"\n📁 Copying to {train_target}...")
    for d in train_dirs:
        dest = train_target / d.name
        if dest.exists(): shutil.rmtree(dest)
        shutil.copytree(d, dest)
        train_names.append(d.name)
        print(f"✅ Train: {d.name}")

    print(f"\n📁 Copying to {val_target}...")
    for d in val_dirs:
        dest = val_target / d.name
        if dest.exists(): shutil.rmtree(dest)
        shutil.copytree(d, dest)
        val_names.append(d.name)
        print(f"✅ Val: {d.name}")

    match_df = pd.read_csv("set_data/match_v1.csv")

    train_df = pd.DataFrame(train_names, columns=["video"])
    val_df = pd.DataFrame(val_names, columns=["video"])
    train_df = train_df.merge(match_df, on="video", how="left")
    val_df = val_df.merge(match_df, on="video", how="left")

    train_df.to_csv(output_root / "set_train.csv", index=False)
    val_df.to_csv(output_root / "set_val.csv", index=False)

    return train_target, val_target

# ---
# PyTorch Dataset 類別 (PyTorch Dataset Class)

class ShuttleSetData(Dataset):
    def __init__(self, folder: Path, is_one, is_reverse = True):
        """
        初始化 Dataset，讀取並處理指定 set 的資料。
        """
        super().__init__()
        
        src_sequences = []
        tgt_sequences = []
        #eos_token = class_ls["<eos>"]
        self.is_reverse = is_reverse
        
        # 根據 set 編號確定 CSV 檔案路徑
        csv_path = folder / f'set{is_one}.csv'
        
        # 檢查檔案是否存在，如果不存在則跳過
        if not csv_path.exists():
            print(f"⚠️ 警告：檔案 {csv_path} 不存在，跳過。")
            self.src_raw = [] # 儲存為張量列表
            self.tgt_raw = [] # 儲存為張量列表
            self.q_len = []
            return

        df = pd.read_csv(str(csv_path))
        df_train = get_stroke_types(df, class_ls)
        rally_list = df_train['rally'].unique()

        for rally in rally_list:
            seq = df_train[df_train['rally'] == rally]['type_encode'].tolist()
            # 前向序列的 tgt：src 序列向左位移一個位置

            # 處理反向序列
            if self.is_reverse:
                seq.reverse() # 直接反轉 seq
                #tgt_seq = seq[1:] 
            tgt_seq = seq[1:] 
            #tgt_seq = seq[1:]
            # 將序列轉換為 tensor 並儲存原始張量
            src_sequences.append(torch.tensor(seq, dtype=torch.long))
            tgt_sequences.append(torch.tensor(tgt_seq, dtype=torch.long))

        self.src_raw = src_sequences
        self.tgt_raw = tgt_sequences
        self.q_len = [len(s) for s in src_sequences] # 原始長度仍然有用

    def __len__(self):
        """返回資料集中的項目總數 (rally 的數量)。"""
        return len(self.src_raw)

    def __getitem__(self, i):
        """用來存取特定 rally 的資訊 (src, tgt, q_len)。"""
        # 返回原始張量，它們將由 collate_fn 補齊
        return self.src_raw[i], self.tgt_raw[i], self.q_len[i]
# ---
## 自訂 collate_fn (Custom Collate Function)
# 這個函數將在 DataLoader 中用於批次處理，以確保批次內所有序列長度一致
def custom_collate_fn(batch):
    """
    自訂的 collate_fn 函數，用於批次處理並補齊序列。
    確保補齊後的 src 和 tgt 張量具有相同的序列長度。
    """
    # 使用 zip(*batch) 解開元組列表，這是個很棒的寫法！
    src_sequences, tgt_sequences, q_lengths = zip(*batch)

    # 將 src 序列補齊到當前批次中的最大長度
    # pad_sequence 會自動找到批次內的最長序列長度  padding = 35
    padded_src = pad_sequence(src_sequences, batch_first=True, padding_value=PADDING_TOKEN_ID)
    
    # 將 tgt 序列也補齊到與 src 序列相同的最大長度
    # 這裡我們需要手動處理，因為 tgt 序列可能比 src 短
    max_src_len = padded_src.size(1)
    
    # 創建一個張量作為目標，其大小與 padded_src 相同 padding = 35
    padded_tgt = torch.full_like(padded_src, fill_value=PADDING_TOKEN_ID, dtype=torch.long)
    # print("padded src: ", padded_src)
    # print("padded tgt", padded_tgt)
    # 將每個 tgt 序列的數據複製到新的張量中
    # 確保不會超出邊界
    for i, tgt_seq in enumerate(tgt_sequences):
        # 複製數據，只取 tgt_seq 的長度
        # 如果 tgt_seq 比 max_src_len 短，剩下的部分會是 35（padding）
        padded_tgt[i, :len(tgt_seq)] = tgt_seq

    # # 將 q_lengths 轉換為 torch.Tensor
    q_lengths_tensor = torch.tensor(q_lengths, dtype=torch.long) 

    # 返回補齊後的 src、tgt 張量和原始序列長度
    return padded_src, padded_tgt, q_lengths_tensor


def training_data(model: nn.Module, loss_fn, loader, optimizer: optim.Optimizer, device):
    model.train()
    total_loss = 0.0

    for src, tgt, q_len in loader:
        N, L = src.shape
        src: Tensor = src.to(device)
        tgt: Tensor = tgt.to(device)
        q_len = q_len.to(device)

        ballrounds = None

        logits, _ = model(src, ballrounds, q_len) #[N, L, C]

        #logits, _ = model(src)          # [N, L, C]
        #logits = logits[:, 1:, :]    
        tgt = tgt                    # [N, L]
        loss_mask = tgt != PADDING_TOKEN_ID        # [N, L]

        logits = logits.contiguous().view(-1, logits.size(-1))  # [N*(L), C]
        tgt = tgt.contiguous().view(-1)                          # [N*(L)]
        loss_mask = loss_mask.view(-1)                           # [N*(L)]

        loss_pure: Tensor = loss_fn(logits, tgt) * loss_mask
        loss = loss_pure.sum() / loss_mask.sum()  # 更精準的平均方式

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    train_loss = total_loss / len(loader)
    return train_loss

@torch.no_grad()
def validation_data(model: nn.Module, loss_fn, loader: DataLoader, device):
    model.eval()
    total_loss = 0.0
    n_correct = 0
    total_valid_tokens = 0
    #all_features = []  # 用來存全部 batch 的 features

    for src, tgt, q_len in loader:
        N, L = src.shape
        src: Tensor = src.to(device)
        tgt: Tensor = tgt.to(device)
        q_len = q_len.to(device)

        ballrounds = None

        logits, lstm_output = model(src, ballrounds, q_len)
       # outputs = model(src)

        
        #logits, lstm_output = outputs
        #all_features.append(features)
     

        #logits = model(src)  
        tgt = tgt                     # [N, L]
        loss_mask = tgt != PADDING_TOKEN_ID          # [N, L]

        # Accuracy: 預測正確的 token 數量
        preds = torch.argmax(logits, dim=-1)  # [N, L]
        element_correct = (preds == tgt) & loss_mask
        n_correct += element_correct.sum().item()
        total_valid_tokens += loss_mask.sum().item()

        logits = logits.contiguous().view(-1, logits.size(-1))  # [N*(L), C]
        tgt = tgt.contiguous().view(-1)
        loss_mask = loss_mask.view(-1)

        loss_pure: Tensor = loss_fn(logits, tgt) * loss_mask
        loss = loss_pure.sum() / loss_mask.sum()
        total_loss += loss.item()

    val_loss = total_loss / len(loader)
    val_acc = n_correct / total_valid_tokens  # 每一拍正確率
    return val_loss, val_acc, lstm_output


def train_network(model: nn.Module, train_loader, val_loader, device, save_path: Path):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    #loss_fn = nn.CrossEntropyLoss(ignore_index=35)
    optimizer = optim.Adam(model.parameters(), hyp.lr)

    best_value = 0.0
    early_stop_count = 0

    for epoch in range(1, hyp.n_epochs + 1):
        t0 = time.time()
        train_loss = training_data(
            model=model,
            loss_fn=loss_fn,
            loader=train_loader,
            optimizer=optimizer,
            device=device
        )
        val_loss, val_acc, model_features = validation_data(
            model=model,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device
        )
        t1 = time.time()
        
        print(f'Epoch({epoch}/{hyp.n_epochs}): train_loss={train_loss:.3f}, '
              f'val_loss={val_loss:.3f}, val_acc={val_acc:.3f}, '
              f'feature shape={model_features.shape}, feature content={model_features.squeeze(1)}, '
              f'- {t1 - t0:.2f} s')
        
        early_stop_count += 1
        if best_value < val_acc:
            best_value = val_acc
            best_state = deepcopy(model.state_dict())
            print(f'Picked! => Best value {val_acc:.3f}')
            early_stop_count = 0

        if early_stop_count == hyp.early_stop_n_epochs:
            print(f'Early stop with best value {best_value:.3f}')
            break
    
    torch.save(best_state, str(save_path))
    model.load_state_dict(best_state)
    return model

# ---
## 主程式類別 (Main Class)
#
class ShuttleSetPredictionTask:
    def __init__(self) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.model = None

    def _load_datasets_from_df(self, folder_df: pd.DataFrame, root_dir: Path, set_type: str, is_reverse: bool):
        """
        輔助函式：從 DataFrame 載入指定的資料集。
        """
        datasets = []
        # 使用 iterrows() 或 itertuples() 是一種更 Pandas 的迭代方式，
        # 但為了簡單和清晰，這裡用 iloc 也沒問題。
        for i in range(len(folder_df)):
            video_name = folder_df['video'].iloc[i]
            count_set = int(folder_df['set'].iloc[i])
            for j in range(count_set):
                folder_path = root_dir / set_type / video_name
                dataset = ShuttleSetData(folder_path, j + 1, is_reverse)
                if len(dataset) > 0:
                    datasets.append(dataset)
                    print(f"   ✅ 已載入{set_type}集：{folder_path.name}/set{j+1}.csv ({len(dataset)} rallies)")
        return datasets
    
    def prepare_dataloaders(self, root_dir: Path, is_reverse: bool):
        """
        準備訓練和驗證的 DataLoader。

        Args:
            root_dir (Path): 包含 'train' 和 'val' 資料夾的根路徑 (例如: set_data)。
            is_reverse (bool): 是否使用反向序列。
        """
        # 讀取訓練和驗證集的影片資訊
        train_folders_name = get_video_folder_name(True)
        val_folders_name = get_video_folder_name(False)

        # 創建一個空的 DataLoader 列表
        train_datasets = []
        val_datasets = []
        
        # 處理訓練集資料
        print("\n--- 準備訓練集 DataLoader ---")
        for i in range(len(train_folders_name)):
            video_name = train_folders_name['video'].iloc[i]
            count_set = int(train_folders_name['set'].iloc[i])
            for j in range(count_set):
                folder_path = root_dir / "train" / video_name
                dataset = ShuttleSetData(folder_path, j + 1, is_reverse)
                if len(dataset) > 0:
                    train_datasets.append(dataset)
                    print(f"  ✅ 已載入訓練集：{folder_path.name}/set{j+1}.csv ({len(dataset)} rallies)")

        # 處理驗證集資料
        print("\n--- 準備驗證集 DataLoader ---")
        train_datasets = self._load_datasets_from_df(
            folder_df=train_folders_name,
            root_dir=root_dir,
            set_type="train",
            is_reverse=is_reverse
        )

        print("\n--- 準備驗證集 DataLoader ---")
        val_datasets = self._load_datasets_from_df(
            folder_df=val_folders_name,
            root_dir=root_dir,
            set_type="val",
            is_reverse=is_reverse
        )


        # 將所有 ShuttleSetData 實例合併成一個大的 Dataset
        from torch.utils.data import ConcatDataset
        train_full_dataset = ConcatDataset(train_datasets)
        val_full_dataset = ConcatDataset(val_datasets)

        self.train_loader = DataLoader(
            dataset=train_full_dataset,
            batch_size=hyp.batch_size,
            shuffle=True,
            pin_memory=self.use_cuda,
            collate_fn=custom_collate_fn # 使用自訂的 collate_fn
        )
        self.val_loader = DataLoader(
            dataset=val_full_dataset,
            batch_size=hyp.batch_size,
            pin_memory=self.use_cuda,
            collate_fn=custom_collate_fn # 使用自訂的 collate_fn
        )
        print(f"\n--- 資料載入完成 ---")
        print(f"訓練集總 rally 數: {len(train_full_dataset)}")
        print(f"驗證集總 rally 數: {len(val_full_dataset)}")
        print(f"train data 內容", train_full_dataset[0])
        print(f"val data 內容",val_full_dataset[0] )

    def get_network_architecture(self, model_name, verbose=False, is_reverse=False):
        """根據模型名稱建立並初始化神經網路。"""
        match model_name:
            case 'LSTMPredictor':
                net = LSTMPredictor(
                    type_size=32, # type_size = max(class_ls.values())
                    emb_size=128,
                    hid_size=256,
                    n_layers=2,
                    is_reverse=is_reverse
                )
            case _:
                raise NotImplementedError
            
        # if verbose:
        #     # summary 的 input_data 應該能涵蓋所有可能的輸入維度
        #     # 在這裡使用 batch_size=1, seq_len=1，因為 collate_fn 會處理實際長度
        #     input_data = torch.zeros((1, 1), dtype=torch.int) 
        #     summary(net, input_data=input_data, device='cpu')

        if verbose:
            # 模擬最簡輸入 (batch=1, seq_len=1)
            x = torch.zeros((1, 1), dtype=torch.int)         # dummy input token
            ballrounds = torch.tensor([1], dtype=torch.int)  # dummy hit frame index，注意最小要 1
            q_len = torch.tensor([1], dtype=torch.long)     # dummy q_len，注意最小要 1
            
            input_data = (x, ballrounds, q_len)
            summary(net, input_data=input_data, device='cpu')
                
        self.model_name = model_name
        self.model = net.to(self.device)

    def seek_network_weights_and_train(self, model_info='', serial_no=1, is_reverse = False):
        """
        找尋已訓練好的權重，若無則啟動訓練。
        """
        model_info = f'_{model_info}' if model_info != '' else ''
        serial_str = f'_{serial_no}' if serial_no != 1 else ''
        save_name = self.model_name.lower() + model_info + serial_str
        if is_reverse == False:
            weight_path = Path(f'weight/{save_name}.pt')
            if not weight_path.parent.is_dir():
                weight_path.parent.mkdir()
                
            if weight_path.exists():
                print(f"\n✅ 找到預訓練權重 {weight_path}，正在載入...")
                self.model.load_state_dict(torch.load(str(weight_path), map_location=self.device))
            else:
                print(f"\n❌ 未找到權重，開始訓練模型並儲存至 {weight_path}...")
                train_t0 = time.time()
                self.model = train_network(
                    model=self.model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    device=self.device,
                    save_path=weight_path
                )
                train_t1 = time.time()
                t = timedelta(seconds=int(train_t1 - train_t0))
                print(f'Total training time: {t}')
        else:
            weight_path = Path(f'weight/{save_name}_backward.pt')
            if not weight_path.parent.is_dir():
                weight_path.parent.mkdir()
                
            if weight_path.exists():
                print(f"\n✅ 找到預訓練權重 {weight_path}，正在載入...")
                self.model.load_state_dict(torch.load(str(weight_path), map_location=self.device))
            else:
                print(f"\n❌ 未找到權重，開始訓練模型並儲存至 {weight_path}...")
                train_t0 = time.time()
                self.model = train_network(
                    model=self.model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    device=self.device,
                    save_path=weight_path
                )
                train_t1 = time.time()
                t = timedelta(seconds=int(train_t1 - train_t0))
                print(f'Total training time: {t}')


# ---
## 主程式執行區塊 (Main Execution Block)
#
if __name__ == "__main__":
    # 步驟 1: 設定資料根目錄
    data_root_folder = Path("set") # 這裡替換成你原始資料夾的根目錄
    split_root_folder = Path("processed_shuttle_data") # 這裡是切分後存放資料的目錄

    # 步驟 2: 分割資料集 (如果還沒分割過)
    # 這會將資料夾複製到 set_data/train 和 set_data/val 並生成 CSV 檔案
    print("--- 執行資料夾分割 ---")
    if not (split_root_folder / "train").exists() or not (split_root_folder / "val").exists():
        split_train_val(data_root_folder, 0.2)
    else:
        print("📁 資料夾已存在，跳過分割。")
    
    # 步驟 3: 創建任務實例
    task = ShuttleSetPredictionTask()

    # 步驟 4: 準備 DataLoader
    # 這裡會讀取 set_data/train 和 set_data/val 資料夾下的所有 set.csv 檔案
    # is_reverse=False 表示使用正向序列，如果你想訓練反向序列，設為 True
    is_reverse = True   #選擇True / False
    task.prepare_dataloaders(root_dir=split_root_folder, is_reverse=is_reverse)

    # 步驟 5: 獲取模型架構
    # 請根據 class_ls 的總類別數量調整 type_size，
    # 這裡 class_ls 總共有 20 個項目 (未知球種和<eos>)。
    # 球員 B 的球種編碼是在原編碼上加 17
    # 在 LSTMPredictor 類別初始化時，type_size 應該設為 36。
    task.get_network_architecture(model_name='LSTMPredictor', verbose=True, is_reverse=is_reverse)

    # 步驟 6: 尋找權重並開始訓練
    # 如果 weight/mylstm_1.pt 存在，會直接載入；否則會啟動訓練
    task.seek_network_weights_and_train(serial_no=1, is_reverse= is_reverse)

    print("\n✅ 訓練流程已完成！")

