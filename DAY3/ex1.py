import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入数据集
dataset = torchvision.datasets.CIFAR10(root="dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)

# 设置input
input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)


# 非线性激活网络
class Chen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        output = self.relu(input)
        return output

chen = Chen()

output = chen(input)
print(output)