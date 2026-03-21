import gc
import math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import IDCD

torch.set_default_tensor_type(torch.FloatTensor)

class IDCDataset(Dataset):
    '''
    the dataset of IDCD.
    '''
    def __init__(self, df_log: pd.DataFrame, n_user:int, n_item:int, Q_mat = None):
        self.df_log = df_log
        self.log_mat = np.zeros((n_user, n_item))
        self.user_id = df_log['user_id'].values
        self.item_id = df_log['item_id'].values
        self.score = df_log['score'].values

        # 添加打印以检查原始分数范围
        print(f"Original score range: [{self.score.min()}, {self.score.max()}]")

        # 归一化分数到[-1, 1]范围
        score_min = self.score.min()
        score_max = self.score.max()
        self.score = 2 * (self.score - score_min) / (score_max - score_min) - 1

        print(f"Normalized score range: [{self.score.min()}, {self.score.max()}]")

        pbar = tqdm(total = df_log.shape[0],desc='Loading data')
        for i, row in df_log.iterrows():
            # 同样归一化log_mat中的分数
            normalized_score = 2 * (row['score'] - score_min) / (score_max - score_min) - 1
            self.log_mat[int(row['user_id']), int(row['item_id'])] = normalized_score
            pbar.update(1)
        pbar.close()

    def __getitem__(self, index):
        user_id = self.user_id[index]
        item_id = self.item_id[index]
        return torch.Tensor(self.log_mat[user_id,:]), \
            torch.Tensor(self.log_mat[:, item_id]), \
            torch.LongTensor([user_id]), \
            torch.LongTensor([item_id]), \
            torch.FloatTensor([self.score[index]]) \

    def __len__(self):
        return self.user_id.shape[0]

def train(model:IDCD, train_data: pd.DataFrame, valid_data: pd.DataFrame, \
    batch_size, lr, n_epoch):
    model.train()
    device = model.device
    dataset = IDCDataset(train_data, model.n_user, model.n_item)
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, \
        shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    result_per_epoch = []
    best_valid_rmse = float('inf')

    for epoch in range(n_epoch):
        model.train()
        result_epoch = {}
        pbar = tqdm(total = len(dataloader),desc = 'Epoch %d'%epoch)
        score_all = []
        pred_all = []
        epoch_loss = 0
        Theta_old = model.get_Theta_buf().numpy().copy()

        for i, (user_log, item_log, user_id, item_id, score) \
            in enumerate(dataloader):
            user_log = user_log.to(device)
            item_log = item_log.to(device)
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            score = score.to(device)

            # 添加打印以检查数据范围
            if i == 0:
                print(f"Batch score range: [{score.min().item()}, {score.max().item()}]")

            pred = model(user_log, item_log, user_id, item_id)
            loss = F.mse_loss(pred, score)
            epoch_loss += loss.item()

            score_all += score.detach().cpu().numpy().reshape(-1,).tolist()
            pred_all += pred.detach().cpu().numpy().reshape(-1,).tolist()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            pbar.update(1)

        pbar.close()
        epoch_loss /= len(dataloader)

        # 验证集评估
        if valid_data is not None:
            valid_metrics = eval(model, valid_data, batch_size=16)
            scheduler.step(valid_metrics['rmse'])

            # 保存最佳模型
            if valid_metrics['rmse'] < best_valid_rmse:
                best_valid_rmse = valid_metrics['rmse']
                torch.save(model.state_dict(), 'best_model.pt')

        # Update examinee traits
        for i in range(math.ceil(dataset.log_mat.shape[0]/batch_size)):
            idx = np.arange(i*batch_size, min(dataset.log_mat.shape[0]\
                , (i+1)*batch_size))
            model.update_Theta_buf(model.diagnose_theta(\
                torch.Tensor(dataset.log_mat[idx,:])\
                .to(device)).detach(),torch.LongTensor(idx))

        # Update question features
        for i in range(math.ceil(dataset.log_mat.shape[1]/batch_size)):
            idx = np.arange(i*batch_size, min(dataset.log_mat.shape[1]\
                ,(i+1)*batch_size))
            model.update_Psi_buf(model.diagnose_psi(\
                torch.Tensor(dataset.log_mat[:,idx].T)\
                .to(device)).detach(),torch.LongTensor(idx))
        model.train()
        Theta_new = model.get_Theta_buf().numpy().copy()

        score_all = np.array(score_all)
        pred_all = np.array(pred_all)

        Theta_norm = np.sqrt(np.sum(np.abs(Theta_new-Theta_old)))

        # 计算训练集上的评估指标
        train_metrics = get_eval_result(score_all, pred_all, None)

        print('Theta_old.head =', Theta_old[:5,0])
        print('Theta_new.head =', Theta_new[:5,0])
        print('epoch = %d, theta_norm = %.6f'%(epoch, Theta_norm))

        result_epoch['Theta_old_head'] = Theta_old[:5,:5]
        result_epoch['Theta_new_head'] = Theta_new[:5,:5]
        result_epoch['Theta_norm'] = Theta_norm
        result_epoch['train_eval'] = train_metrics

        if valid_data is not None:
            result_epoch['valid_eval'] = valid_metrics
        result_per_epoch.append(result_epoch)
    return result_per_epoch

def get_eval_result(s_true, s_pred, s_pred_label):
    """
    计算多个评估指标
    Args:
        s_true: 真实分数
        s_pred: 预测分数
        s_pred_label: 已不再使用
    Returns:
        包含多个评估指标的字典
    """
    # 确保输入是一维数组
    s_true = np.array(s_true).flatten()
    s_pred = np.array(s_pred).flatten()

    # 均方根误差
    rmse = float(np.sqrt(mean_squared_error(s_true, s_pred)))
    # 平均绝对误差
    mae = float(np.mean(np.abs(s_true - s_pred)))
    # 皮尔逊相关系数
    pearson = float(np.corrcoef(s_true, s_pred)[0,1])
    # R方分数
    r2 = float(1 - np.sum((s_true - s_pred) ** 2) / np.sum((s_true - np.mean(s_true)) ** 2))

    print(f'RMSE = {rmse:.6f}, MAE = {mae:.6f}')
    print(f'Pearson = {pearson:.6f}, R² = {r2:.6f}')

    return {
        'rmse': rmse,  # 均方根误差
        'mae': mae,    # 平均绝对误差
        'pearson': pearson,  # 皮尔逊相关系数
        'r2': r2      # R方分数
    }

def eval(model:IDCD, data: pd.DataFrame, batch_size):
    model.eval()
    device = model.device
    eval_result = {}
    dataset = IDCDataset(data, model.n_user, model.n_item)
    dataloader = DataLoader(dataset = dataset, \
        batch_size = batch_size, shuffle = False)
    y_pred = []
    y_true = []
    for i, (user_log, item_log, user_id, item_id, score) \
        in enumerate(dataloader):

        user_log = user_log.to(device)
        item_log = item_log.to(device)
        user_id = user_id.to(device)
        item_id = item_id.to(device)
        pred_1_batch = model.forward_using_buf(user_id, \
            item_id).detach().cpu().numpy()
        y_pred.extend(pred_1_batch.flatten())
        y_true.extend(score.numpy().flatten())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    eval_result = get_eval_result(y_true, y_pred, None)
    return eval_result

def check_data_distribution(dataloader):
    """检查数据分布"""
    all_scores = []
    all_preds = []

    for user_log, item_log, user_id, item_id, score in dataloader:
        all_scores.extend(score.numpy().flatten())

    scores = np.array(all_scores)
    print("\nData Distribution Statistics:")
    print(f"Score Mean: {scores.mean():.4f}")
    print(f"Score Std: {scores.std():.4f}")
    print(f"Score Range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Score 25/50/75 percentiles: {np.percentile(scores, [25, 50, 75])}")
