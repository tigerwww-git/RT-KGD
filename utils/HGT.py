import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from process import process_load


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, node_type_list, metadata):
        """建立HGT模型，模型包含四个参数
        hidden_channels(隐藏层维度)，out_channels(输出层维度)，num_heads(多头注意力机制头的数量)，num_layers（HGT层数）
        node_type_list(结点集), metadata(点集，边集（头，边，尾）)"""
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_type_list:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)  # -1表示根据输入的数据的维度自动调整

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')  # ?
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        """ x_dict：Dict[NodeType, Tensor]
            edge_index_dict：Union[Dict[EdgeType, Tensor],
                               Dict[EdgeType, SparseTensor]]  # Support both."""
        for node_type, x in x_dict.items():
            # print(x)
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()     # 将x通过线性层转为设定的维度hidden-channels

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict


# model = HGT(hidden_channels=64, out_channels=4, num_heads=2, num_layers=1, node_type_list=node_type, metadata=meta_data)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data, model = hetero_data.to(device), model.to(device)