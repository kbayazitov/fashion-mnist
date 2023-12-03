import torch


class CNN(torch.nn.Module):
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    def __init__(self):
        super(CNN, self).__init__()

        self.layers = torch.nn.Sequential()
        self.layers.add_module("c1", torch.nn.Conv2d(1, 6, kernel_size=5))
        self.layers.add_module("relu1", torch.nn.ReLU())
        self.layers.add_module("p1", torch.nn.MaxPool2d(kernel_size=2))
        self.layers.add_module("c2", torch.nn.Conv2d(6, 16, kernel_size=5))
        self.layers.add_module("relu2", torch.nn.ReLU())
        self.layers.add_module("p2", torch.nn.MaxPool2d(kernel_size=2))
        self.layers.add_module("flatten", torch.nn.Flatten())
        self.layers.add_module("linear1", torch.nn.Linear(16 * 4 * 4, 120))
        self.layers.add_module("relu3", torch.nn.ReLU())
        self.layers.add_module("linear2", torch.nn.Linear(120, 10))

    def forward(self, input):
        return self.layers(input)
