from torch import nn


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size[0], 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.loss_func=None

    def set_opt(self, _): pass

    def forward(self, x):
        return self.net(x)