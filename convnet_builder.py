import torch
from torch import nn
from attn_blocks import CBAM

from dataclasses import dataclass, field

"""
goal - be able to define a ConvBlock using a config dataclass, in order to speed up the dev process

for example, 

```{python}

from dataclasses import dataclass

@dataclass
class ConvBlockConfig:
    in_channels: int = 3
    out_channels: int
    structure: str = "conv-bn-relu"
    depth: int = 1 # records many successive blocks should be applied

    
block_config = ConvBlockConfig(in_channels=3, out_channels=32, structure="conv-bn-relu", depth=2)
block = GenerateBlock(block_config)
```
"""

FUNCS = {
    "conv": nn.Conv2d,
    "bn": nn.BatchNorm2d,
    "relu": nn.ReLU,
    "cbam": CBAM,
    "max_pool": nn.MaxPool2d,
    "avg_pool": nn.AvgPool2d,
    "dropout": nn.Dropout2d,
}


def default_field(obj):
    return field(default_factory=lambda: obj) if isinstance(obj, dict) else obj


@dataclass
class ConvBlockConfigV1:
    in_channels: list[int] = (3, 10)  # in_channels for each conv layer
    out_channels: list[int] = (10, 3)  # out_channels for each conv layer
    kernel_size: list[int] = (3, 3)  # kernel size for each conv layer
    padding: list[int] = (1, 1)  # padding for each conv layer
    structure: str = "conv-bn-relu"  # general structure of each successive conv block
    depth: int = len(in_channels)  # records many successive blocks should be applied
    assert depth > 0, "depth must be greater than zero"
    depth = depth if depth <= len(in_channels) else len(in_channels)


@dataclass
class ConvBlockConfigV2:
    structure: dict[str:list] = default_field(
        {
            "conv": [
                (3, 32, 3, 1, 1),
                (32, 64, 3, 1, 1),
                (64, 128, 3, 1, 1),
                (128, 256, 3, 1, 1),
                (256, 512, 3, 1, 1),
                (512, 512, 3, 1, 1),
            ],  # (in_channels, out_channels, kernel_size, stride, padding)
            "cbam": [(32, 16), (64, 16), (128, 16), (256, 16), (512, 16), (512, 16)],  # (in_channels, reduction_scalar)
            "relu": [],
        }
    )
    depth: int = 6  # records how many successive blocks should be applied
    assert depth > 0, "depth must be greater than zero"
    depth = depth if depth <= len(structure.default_factory()["conv"]) else len(structure.default_factory()["conv"])


"""
Potential Update to make it even more modular:
    i) define a sequence variable that tells the class how the model should be built. keep the `structure` variable as a placeholder
    for the params. 
        .) once we add a func to our func_list, we can pop/remove the first element from the structure key. this allows us to keep track of the
        model, ensuring that accurately build the model
"""


class GenerateConvBlockV1(nn.Module):

    def __init__(self, config: ConvBlockConfigV1) -> None:
        super(GenerateConvBlockV1, self).__init__()
        self.config = config

        if isinstance(self.config.structure, str):
            self.structure = [self.config.structure] * self.config.depth
        else:
            raise ValueError(f"structure must be a str (delimited by '-'), got {type(self.config.structure)} instead")

        self.block = self._get_sequence()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.block(x)

    def _get_sequence(self) -> nn.Sequential:
        func_list = []
        for idx, (block, in_channels, out_channels, kernel_size, padding) in enumerate(
            zip(
                self.structure,
                self.config.in_channels,
                self.config.out_channels,
                self.config.kernel_size,
                self.config.padding,
            )
        ):
            for func in block.split("-"):
                if func in FUNCS.keys():
                    if func in "conv":
                        func_list.append(
                            FUNCS[func](
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                            )
                        )
                    elif func == "cbam":
                        func_list.append(
                            CBAM(
                                in_channels=out_channels,
                            )
                        )
                    elif func in ["relu"]:
                        func_list.append(FUNCS[func]())
                    else:
                        func_list.append(FUNCS[func](out_channels))
                else:
                    raise ValueError(f"function {func} not recognized")
        return nn.Sequential(*func_list)


class GenerateConvBlockV2(nn.Module):
    def __init__(self, config: ConvBlockConfigV2) -> None:
        super().__init__()
        self.config = config

        if isinstance(self.config.structure, dict):
            self.structure = self.config.structure
        else:
            raise ValueError(f"structure must be a dict, got {type(self.config.structure)} instead")

        self.block = self._get_sequence()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.block(x)

    def _get_sequence(self) -> nn.Sequential:
        func_list = []
        for idx in range(self.config.depth):
            for val in self.structure.keys():
                if val in FUNCS:
                    if len(self.structure[val]) <= idx and len(self.structure[val]) > 0:
                        continue
                    func_val = self.structure[val][idx] if len(self.structure[val]) > 0 else None
                    if func_val:
                        func_list.append(FUNCS[val](*func_val))
                    else:
                        func_list.append(FUNCS[val]())
                else:
                    raise ValueError(f"function {val} not recognized")
        return nn.Sequential(*func_list)


class ConvBlock(nn.Module):
    """
    Convolutional Block
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cbam = CBAM(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.relu(self.cbam(self.conv(x)))


class ConvNet(nn.Module):
    """
    Convolutional Neural Network - follows the standard conv-cbam-relu blocks
    """

    def __init__(self, in_channels: int, out_channels: int = 3) -> None:
        super(ConvNet, self).__init__()

        self.conv1 = ConvBlock(in_channels, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 512)
        self.conv6 = ConvBlock(512, 512)
        self.fc = nn.Linear(524_288, out_channels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return self.fc(x.flatten())


class ConvNet2(nn.Module):
    """
    Convolutional Neural Network - follows the standard conv-cbam-relu blocks
    """

    def __init__(self, in_channels: int, out_channels: int = 3) -> None:
        super(ConvNet2, self).__init__()

        block_config = ConvBlockConfigV1(
            in_channels=(in_channels, 32, 64, 128, 256, 512),
            out_channels=(32, 64, 128, 256, 512, 512),
            kernel_size=(3, 3, 3, 3, 3, 3),
            padding=(1, 1, 1, 1, 1, 1),
            structure="conv-cbam-relu",
            depth=6,
        )

        self.block = GenerateConvBlockV1(block_config)
        self.fc = nn.Linear(524_288, out_channels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.block(x)
        return self.fc(x.flatten())


class ConvNet3(nn.Module):
    """
    Convolutional Neural Network - follows the standard conv-cbam-relu blocks
    """

    def __init__(self, in_channels: int, out_channels: int = 3) -> None:
        super(ConvNet3, self).__init__()

        block_config = ConvBlockConfigV2(
            structure={
                "conv": [
                    (3, 32, 3, 1, 1),  # (in_channels, out_channels, kernel_size, stride, padding)
                    (32, 64, 3, 1, 1),
                    (64, 128, 3, 1, 1),
                    (128, 256, 3, 1, 1),
                    (256, 512, 3, 1, 1),
                    (512, 512, 3, 1, 1),
                ],
                "cbam": [
                    (32, 16),
                    (64, 16),
                    (128, 16),
                    (256, 16),
                    (512, 16),
                    (512, 16),
                ],  # (in_channels, reduction_scalar)
                "relu": [],
            },
            depth=6,
        )

        self.block = GenerateConvBlockV2(block_config)
        self.fc = nn.Linear(524_288, out_channels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.block(x)
        return self.fc(x.flatten())


class LeNet(nn.Module):
    # let's see if we can create LeNet5 - http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.config = ConvBlockConfigV2(
            structure={
                "conv": [(in_channels, 6, 5, 1, 0), (6, 16, 5, 1, 0), (16, 120, 5, 1, 0)],
                "relu": [],
                "avg_pool": [(2, 2), (2, 2)],
            }
        )
        self.block = GenerateConvBlockV2(self.config)
        self.fc_1 = nn.Linear(120, 84)
        self.fc_2 = nn.Linear(84, out_channels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.block(x)
        x = self.fc_1(x.flatten())
        x = self.fc_2(x)
        return x


if __name__ == "__main__":
    model = ConvNet(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)

    model_2 = ConvNet2(in_channels=3, out_channels=3)
    y_2 = model_2(x)
    print(y_2.shape)

    model_3 = ConvNet3(in_channels=3, out_channels=3)
    y_3 = model_3(x)
    print(y_3.shape)

    mnist = torch.randn(1, 1, 32, 32)
    le_net = LeNet(in_channels=1, out_channels=10)
    le_net_out = le_net(mnist)
    print(le_net_out.shape)
