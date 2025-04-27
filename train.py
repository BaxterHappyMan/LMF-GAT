import torch
import tqdm
from sklearn.metrics import roc_auc_score
from utils.write_logs import write_logs, write_loss
import yaml
from argparse import Namespace
from model.GAT import GAT
from torch import optim
from dataloader.dataloader import data_loader_v1, data_loader_v2, data_loader_v3
from torch_geometric.loader import DataLoader
import datetime
import os.path


# 最大最小归一化
def min_max_normalize(features):
    min_val = features.min(dim=0).values
    max_val = features.max(dim=0).values
    normalized_feature = (features - min_val)/(max_val - min_val)
    return normalized_feature

def z_score_normalize(features):
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    normalized_features = (features - mean)/std
    return normalized_features

def train(model, device, train_dataloader, nums, opt, cri, epochs, log_path, loss_path, validation_dataloader=None):
    train_loss = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        bar = tqdm.tqdm(nums, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in train_dataloader:
            opt.zero_grad()
            out = model(batch.to(device))
            # 计算交叉熵损失
            loss = cri(out, batch.node_y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            bar.update(len(batch))
        average_loss = total_loss/nums
        train_loss.append(average_loss)
        train_log = f'Epoch {epoch+1}, training average loss: {average_loss:.6f}'
        # 将训练记录写入日志
        write_logs(train_log, log_path)
        print(train_log)
        if validation_dataloader is not None:
            predict(model, device, validation_dataloader, log_path)
        bar.close()
    write_loss(train_loss, loss_path)

def predict(model, device, dataloader, log_path):
    # 验证模式
    model.eval()
    correct = 0
    total = 0
    # 记录正类预测概率值
    graph_score = []
    graph_label = []
    with torch.no_grad():
        for batch in dataloader:
            out = model(batch.to(device))
            _, prediction = torch.max(out, dim=1)
            # 计算分类正确的节点数
            correct += (prediction==batch.node_y).sum().item()
            # 计算节点总数
            total += len(batch.node_y)
            for i in range(len(batch.ptr)-1):
                start_idx = batch.ptr[i]
                end_idx = batch.ptr[i+1]
                # 当前图的节点正样本预测概率值
                prediction_pos = out[start_idx:end_idx,1].cpu()
                # 计算当前图的正类分类预测值，即正样本节点平均值
                score = torch.sum(prediction_pos)/((end_idx-start_idx).cpu())
                graph_score.append(score)
            # 图级标签
            graph_label += batch.y.cpu()
    accuracy = correct/total
    auc_score = roc_auc_score(y_true=graph_label, y_score=graph_score)
    # 计算pAUC
    pauc_score = roc_auc_score(y_true=graph_label, y_score=graph_score, max_fpr=0.1)
    log = f'Testing accuracy: {accuracy:.6f}, AUC score: {auc_score:.6f}, pAUC score: {pauc_score:.6f}'
    if log_path != 'no_log':
        write_logs(log, log_path)
    print(log)

def train_GAT(config_idx):
    # 加载超参数
    with open("config.yaml", "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    if type(config['GAT']) is not Namespace:
        config = Namespace(**config['GAT'])
    if config.in_channels != config.n_mel:
        config.in_channels = config.n_mel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GAT(in_channels=config.in_channels, num_heads=config.num_heads, out_channels=config.out_channels,
              out_channel_1=config.out_channels_1, out_channel_2=config.out_channels_2, out_channel_3=config.out_channels_3,
              n_mel=config.n_mel, sample_rate=config.sample_rate, n_fft=config.n_fft)
    model = net.to(device)
    # check idx
    data_config = config.Data_15
    if data_config is not Namespace:
        data_config = Namespace(**data_config)
    if data_config.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=config.lr)
    if data_config.criterion == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    train_dataset, test_dataset = data_loader_v3(data_config.dataset_root, data_config.dataset_normal,
                                                 data_config.dataset_anomaly, data_config.normal_vs_anomaly)
    nums = len(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    # 创建日志文件
    datestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dir = '../train_records/' + datestamp_str
    if not os.path.exists(dir):
        os.makedirs(dir)
    log_path = dir + '/' + str(config_idx) + 'log.txt'
    loss_path = dir + '/' + str(config_idx) + 'loss.txt'
    total_params = sum(param.numel() for param in model.parameters())
    print(f'参数总量：{total_params}')
    train(model, device, train_dataloader, nums, optimizer, criterion, config.epochs, log_path, loss_path, test_dataloader)
    pass

if __name__ == '__main__':
    train_GAT(15)