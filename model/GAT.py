from torch import nn
import torch_geometric as geometric
import torch.nn.functional as F
from model.LearnableMelFilter import LearnableMelFilter


'''
1层图卷积+3层线性层，完成节点分类任务或者图分类任务
'''
class GAT(nn.Module):
    def __init__(self, in_channels, num_heads, out_channels, out_channel_1, out_channel_2, out_channel_3, n_mel, sample_rate, n_fft):
        super(GAT, self).__init__()
        self.gat1 = geometric.nn.GATConv(in_channels=in_channels, out_channels=out_channels, heads=num_heads)
        self.linear1 = nn.Linear(in_features=out_channels * num_heads, out_features=out_channel_1)
        self.linear2 = nn.Linear(in_features=out_channel_1, out_features=out_channel_2)
        self.linear3 = nn.Linear(in_features=out_channel_2, out_features=out_channel_3)
        self.dropout = nn.Dropout(p=0.2)
        self.fb = LearnableMelFilter(n_mel, sample_rate, n_fft)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.fb(x)
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear3(x))
        x = F.softmax(x, dim=-1)
        return x
