r"""`fast_rl.layers` provides essential functions to building and modifying `model` architectures"""
from math import ceil

from fastai.layers import embedding
from fastai.torch_core import *
from torch.distributions import Normal


def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None,lin_cls=nn.Linear):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(lin_cls(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


class TabularModel(Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False,lin_cls=nn.Linear):
        super().__init__()
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act,lin_cls=lin_cls)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x


def init_cnn(mod: Any):
    r""" Utility for initializing cnn Modules. """
    if getattr(mod, 'bias', None) is not None: nn.init.constant_(mod.bias, 0)
    if isinstance(mod, (nn.Conv2d, nn.Linear)): nn.init.kaiming_normal_(mod.weight)
    for sub_mod in mod.children(): init_cnn(sub_mod)


def ks_stride(ks, stride, w, h, n_blocks, kern_proportion=.1, stride_proportion=0.3):
    r""" Utility for determing the the kernel size and stride. """
    kernels, strides, max_dim=[], [], max((w, h))
    for i in range(len(n_blocks)):
        kernels.append(max_dim*kern_proportion)
        strides.append(kernels[-1]*stride_proportion)
        max_dim=(max_dim-kernels[-1])/strides[-1]
        assert max_dim>1

    return ifnone(ks, map(ceil, kernels)), ifnone(stride, map(ceil, strides))


class Flatten(nn.Module):
    def forward(self, y): return y.view(y.size(0), -1)


class FakeBatchNorm(Module):
    r""" If we want all the batch norm layers gone, then we will replace the tabular batch norm with this. """
    def forward(self, xi: Tensor, *args): return xi


class GaussianNoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.sigma_weight=nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight",torch.zeros(out_features,in_features))
        self.normal=Normal(0,1)
        if bias:
            self.sigma_bias=nn.Parameter(torch.full((out_features,),sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
            self.reset_parameters()

    def reset_parameters(self):
        std=math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std,std)
        self.bias.data.uniform_(-std,std)

    def forward(self, xi):
        # self.normal=Normal(0,1)
        self.epsilon_weight.data.copy_(self.normal.sample(self.epsilon_weight.shape))
        bias=self.bias
        if bias is not None:
            # self.epsilon_bias.normal_()
            self.epsilon_bias.data.copy_(self.normal.sample(self.epsilon_bias.shape))
            bias=bias+self.sigma_bias*self.epsilon_bias
        return F.linear(xi,self.weight+self.sigma_weight*self.epsilon_weight,bias)


class GaussianNoisyFactorizedLinear(nn.Linear):
    def __init__(self, in_features, out_features,sigma_zero=0.4,bias=True):
        super().__init__(in_features, out_features,bias=bias)
        sigma_init=sigma_zero/math.sqrt(in_features)
        self.sigma_weight=nn.Parameter(torch.full((out_features,in_features),sigma_init))
        self.register_buffer("epsilon_input",torch.zeros((1,in_features)))
        self.register_buffer("epsilon_output", torch.zeros((out_features,1)))
        if bias:
            self.sigma_bias=nn.Parameter(torch.full((out_features,),sigma_init))

    def square_direction(self,x): return torch.sign(x)*torch.sqrt(torch.abs(x))

    def forward(self, xi):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        eps_in,eps_out=self.square_direction(self.epsilon_input),self.square_direction(self.epsilon_output)

        bias=self.bias
        if bias is not None:
            bias=bias+self.sigma_bias*eps_out.t()
        noise_v=torch.mul(eps_in,eps_out)
        return F.linear(xi,self.weight+self.sigma_weight*noise_v,bias)



def conv_bn_lrelu(ni: int, nf: int, ks: int = 3, stride: int = 1, pad=True, bn=True) -> nn.Sequential:
    r""" Create a sequence Conv2d->BatchNorm2d->LeakyReLu layer. (from darknet.py). Allows excluding BatchNorm2d Layer."""
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=(ks//2) if pad else 0),
        nn.BatchNorm2d(nf) if bn else FakeBatchNorm(),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))


class ChannelTranspose(Module):
    r""" Runtime image input channel changing. Useful for handling different image channel outputs from different envs. """
    def forward(self, xi: Tensor):
        return xi.transpose(3, 1).transpose(3, 2)


class StateActionSplitter(Module):
    r""" `Actor / Critic` models require breaking the state and action into 2 streams. """

    def forward(self, s_a_tuple: Tuple[Tensor]):
        r""" Returns tensors as -> (State Tensor, Action Tensor) """
        return s_a_tuple[0], s_a_tuple[1]


class StateActionPassThrough(nn.Module):
    r""" Passes action input untouched, but runs the state tensors through a sub module. """
    def __init__(self, layers):
        super().__init__()
        self.layers=layers

    def forward(self, state_action):
        return self.layers(state_action[0]), state_action[1]


class TabularEmbedWrapper(Module):
    r""" Basic `TabularModel` compatibility wrapper. Typically, state inputs will be either categorical or continuous. """
    def __init__(self, tabular_model: TabularModel):
        super().__init__()
        self.tabular_model=tabular_model

    def forward(self, xi: Tensor, *args):
        return self.tabular_model(xi, xi)


class CriticTabularEmbedWrapper(Module):
    r""" Similar to `TabularEmbedWrapper` but assumes input is state / action and requires concatenation. """
    def __init__(self, tabular_model: TabularModel, exclude_cat):
        super().__init__()
        self.tabular_model=tabular_model
        self.exclude_cat=exclude_cat

    def forward(self, args):
        if not self.exclude_cat:
            return self.tabular_model(*args)
        else:
            return self.tabular_model(0, torch.cat(args, 1))

