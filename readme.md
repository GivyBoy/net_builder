# Neural Net Builder

The idea is to simplify the process of creating the `conv blocks` for a network. This works pretty well right now, but I think it can be improved and extended to create entire netowrks, without having to worry about creating a class. This will be my next update, in addition to cleaning up the code.

It currently works like this:

```{python}
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

mnist = torch.randn(64, 1, 32, 32)
le_net = LeNet(in_channels=1, out_channels=10)
le_net_out = le_net(mnist)
print(le_net_out.shape)  # torch.Size([64, 10])
```
