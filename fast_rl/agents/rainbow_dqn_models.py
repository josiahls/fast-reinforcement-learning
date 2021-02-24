# n-step
from fast_rl.core.layers import *

REWARD_STEPS = 2

# priority replay
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

# C51
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()

        hd_sz=512

        self.fc_val = nn.Sequential(
            GaussianNoisyLinear(input_shape[0], hd_sz),
            nn.ReLU(),
            GaussianNoisyLinear(hd_sz, N_ATOMS)
        )

        self.fc_adv = nn.Sequential(
            GaussianNoisyLinear(input_shape[0], hd_sz),
            nn.ReLU(),
            GaussianNoisyLinear(hd_sz, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

        self.loss_func=None

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float()
        val_out = self.fc_val(fx).view(batch_size, 1, N_ATOMS)
        adv_out = self.fc_adv(fx).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def set_opt(self, _): pass

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())
