import random
from torch.utils.data import Subset, random_split, ConcatDataset
from dataloader.CustomGraphDataset import CustomGraphDataset


# 加载ENFI/Sensor数据集
def data_loader_v1(root, dataset_normal, dataset_anomaly, normal_vs_anomaly):
    dataset_n = CustomGraphDataset(root, dataset_normal)
    dataset_a = CustomGraphDataset(root, dataset_anomaly)
    # 从正常数据集中随机抽样
    length_anomaly = len(dataset_a)
    indices = random.sample(range(len(dataset_n)), int(normal_vs_anomaly*length_anomaly))
    # 创建子集
    subset_n = Subset(dataset_n, indices)
    # 默认为8:2划分训练集和测试集
    train_n_size = int(0.8*len(subset_n))
    test_n_size = len(subset_n) - train_n_size
    train_n, test_n = random_split(subset_n, [train_n_size, test_n_size])
    train_a_size = int (0.8*len(dataset_a))
    test_a_size = len(dataset_a) - train_a_size
    train_a, test_a = random_split(dataset_a, [train_a_size, test_a_size])
    train_dataset = ConcatDataset([train_n, train_a])
    test_dataset = ConcatDataset([test_n, test_a])
    return train_dataset, test_dataset

# 加载ENFI/Sensor数据集，包含训练集、验证集、测试集，比例0.7:0.15:0.15
def data_loader_v1_1(root, dataset_normal, dataset_anomaly, normal_vs_anomaly):
    dataset_n = CustomGraphDataset(root, dataset_normal)
    dataset_a = CustomGraphDataset(root, dataset_anomaly)
    # 从正常数据集中随机抽样
    length_anomaly = len(dataset_a)
    indices = random.sample(range(len(dataset_n)), int(normal_vs_anomaly * length_anomaly))
    # 创建子集
    subset_n = Subset(dataset_n, indices)
    # 按照比例划分训练集、验证集、测试集
    train_n_size = int(0.7*len(subset_n))
    val_n_size = int(0.15*len(subset_n))
    test_n_size = len(subset_n) - train_n_size - val_n_size
    train_a_size = int(0.7*len(dataset_a))
    val_a_size = int(0.15*len(dataset_a))
    test_a_size = len(dataset_a) - train_a_size - val_a_size
    train_n, val_n, test_n = random_split(subset_n, [train_n_size, val_n_size, test_n_size])
    train_a, val_a, test_a = random_split(dataset_a, [train_a_size, val_a_size, test_a_size])
    train_dataset = ConcatDataset([train_n, train_a])
    val_dataset = ConcatDataset([val_n, val_a])
    test_dataset = ConcatDataset([test_n, test_a])
    return train_dataset, val_dataset, test_dataset

# 加载ENFI/Phone数据集
def data_loader_v2(root, dataset_normal, dataset_anomaly, normal_vs_anomaly):
    dataset_n = CustomGraphDataset(root, dataset_normal)
    dataset_a = CustomGraphDataset(root, dataset_anomaly)
    # 从异常数据集中随机抽样
    length_anomaly = len(dataset_a)
    length_normal = len(dataset_n)
    # subset_n = None
    # subset_a = None
    if normal_vs_anomaly == 1:
        indices = random.sample(range(len(dataset_n)), length_anomaly)
        subset_n = Subset(dataset_n, indices)
        subset_a = dataset_a
    else:
        indices = random.sample(range(len(dataset_a)), int((1/normal_vs_anomaly)*length_normal))
        subset_n = dataset_n
        subset_a = Subset(dataset_a, indices)
    # 默认为8:2划分训练集和测试集
    train_n_size = int(0.8*len(subset_n))
    test_n_size = len(subset_n) - train_n_size
    train_n, test_n = random_split(subset_n, [train_n_size, test_n_size])
    train_a_size = int (0.8*len(subset_a))
    test_a_size = len(subset_a) - train_a_size
    train_a, test_a = random_split(subset_a, [train_a_size, test_a_size])
    train_dataset = ConcatDataset([train_n, train_a])
    test_dataset = ConcatDataset([test_n, test_a])
    return train_dataset, test_dataset

# DCASE 2024数据集加载, 训练集测试集默认8：2
def data_loader_v3(root, dataset_normal, dataset_anomaly, normal_vs_anomaly):
    dataset_n = CustomGraphDataset(root, dataset_normal)
    dataset_a = CustomGraphDataset(root, dataset_anomaly)
    train_n_size = int(0.8*len(dataset_n))
    test_n_size = len(dataset_n) - train_n_size
    train_a_size = int(0.8*len(dataset_a))
    test_a_size = len(dataset_a) - train_a_size
    train_n, test_n = random_split(dataset_n, [train_n_size, test_n_size])
    train_a, test_a = random_split(dataset_a, [train_a_size, test_a_size])
    train_datasets = ConcatDataset([train_n, train_a])
    test_datasets = ConcatDataset([test_n, test_a])
    return train_datasets, test_datasets
