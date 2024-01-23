from mindspore import nn, ops


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.block0 = nn.SequentialCell(
            nn.Dense(8, 32),
            nn.ReLU(),
            nn.Dense(32, 64),
            nn.ReLU(),
            nn.Dense(64, 128),
            nn.ReLU(),
        )

        self.block1 = nn.SequentialCell(
            nn.Dense(8, 64),
            nn.ReLU(),
            nn.Dense(64, 128),
            nn.ReLU(),
        )

        self.block2 = nn.SequentialCell(
            nn.Dense(8, 128),
            nn.ReLU(),
        )
        self.block3 = nn.SequentialCell(
            nn.Dense(8, 1024),
            nn.ReLU(),
            nn.Dense(1024, 128),
            nn.ReLU(),
        )

        self.end = nn.SequentialCell(
            nn.Dense(128 * 4, 512),
            nn.ReLU(),
            nn.Dense(512, 2),
        )

    def construct(self, x):
        logits0 = self.block0(x)
        logits1 = self.block1(x)
        logits2 = self.block2(x)
        logits3 = self.block3(x)
        logits = ops.concat((logits0, logits1, logits2, logits3), axis=1)
        output = self.end(logits)
        return output
