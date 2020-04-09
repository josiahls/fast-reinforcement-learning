from fastai.callback import OptimWrapper

from fast_rl.core.layers import *
# import copy


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr



class DQNModule(Module):

    def __init__(self, ni: int, ao: int, layers: Collection[int], discount: float = 0.99, lr=0.001,
                n_conv_blocks: Collection[int] = 0, nc=3, opt=None, emb_szs: ListSizes = None, loss_func=None,
                w=-1, h=-1, ks: Union[None, list]=None, stride: Union[None, list]=None, grad_clip=5,
                conv_kern_proportion=0.1, stride_proportion=0.1, pad=False, batch_norm=False,lin_cls=nn.Linear,
                do_grad_clipping=True):
        r"""
        Basic DQN Module.

        Args:
            ni: Number of inputs. Expecting a flat state `[1 x ni]`
            ao: Number of actions to output.
            layers: Number of layers where is determined per element.
            n_conv_blocks: If `n_conv_blocks` is not 0, then convolutional blocks will be added
                           to the head on top of existing linear layers.
            nc: Number of channels that will be expected by the convolutional blocks.
        """
        super().__init__()
        self.lin_cls=lin_cls
        self.name = 'DQN'
        self.loss = None
        self.loss_func = loss_func
        self.discount = discount
        self.gradient_clipping_norm = grad_clip
        self.lr = lr
        self.batch_norm = batch_norm
        self.switched = False
        self.do_grad_clipping=do_grad_clipping
        # self.ks, self.stride = ([], []) if len(n_conv_blocks) == 0 else ks_stride(ks, stride, w, h, n_conv_blocks, conv_kern_proportion, stride_proportion)
        self.ks, self.stride=([], []) if len(n_conv_blocks)==0 else (ifnone(ks, [10, 10, 10]), ifnone(stride, [5, 5, 5]))
        self.action_model = nn.Sequential()
        _layers = [conv_bn_lrelu(ch, self.nf, ks=ks, stride=stride, pad=pad, bn=self.batch_norm) for ch, self.nf, ks, stride in zip([nc]+n_conv_blocks[:-1],n_conv_blocks, self.ks, self.stride)]

        if _layers: ni = self.setup_conv_block(_layers=_layers, ni=ni, nc=nc, w=w, h=h)
        self.setup_linear_block(_layers=_layers, ni=ni, nc=nc, w=w, h=h, emb_szs=emb_szs, layers=layers, ao=ao)
        self.init_weights(self.action_model)
        self.opt = None
        self.set_opt(opt)

    def set_opt(self, opt):
        self.opt=OptimWrapper.create(ifnone(optim.Adam, opt), lr=self.lr, layer_groups=[self.action_model])

    def setup_conv_block(self, _layers, ni, nc, w, h):
        self.action_model.add_module('conv_block', nn.Sequential(*(self.fix_switched_channels(ni, nc, _layers) + [Flatten()])))
        training = self.action_model.training
        self.action_model.eval()
        ni = int(self.action_model(torch.zeros((1, w, h, nc) if self.switched else (1, nc, w, h))).view(-1, ).shape[0])
        self.action_model.train(training)
        return ni

    def setup_linear_block(self, _layers, ni, nc, w, h, emb_szs, layers, ao):
        tabular_model = TabularModel(emb_szs=emb_szs, n_cont=ni if not emb_szs else 0, layers=layers, out_sz=ao, use_bn=self.batch_norm,lin_cls=self.lin_cls)
        if not emb_szs: tabular_model.embeds = None
        if not self.batch_norm: tabular_model.bn_cont = FakeBatchNorm()
        self.action_model.add_module('lin_block', TabularEmbedWrapper(tabular_model))

    def fix_switched_channels(self, current_channels, expected_channels, layers: list):
        if current_channels == expected_channels:
            return layers
        else:
            self.switched = True
            return [ChannelTranspose()] + layers

    def forward(self, xi: Tensor):
        training = self.training
        if xi.shape[0] == 1: self.eval()
        pred = self.action_model(xi)
        if training: self.train()
        return pred

    def init_weights(self, m):
        if issubclass(m.__class__,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def sample_mask(self, d):
        return torch.sub(1.0, d)

    def optimize(self, sampled):
        r"""Uses ER to optimize the Q-net (without fixed targets).

        Uses the equation:

        .. math::
                Q^{*}(s, a) = \mathbb{E}_{s'∼ \Big\epsilon} \Big[r + \lambda \displaystyle\max_{a'}(Q^{*}(s' , a'))
                \;|\; s, a \Big]


        Returns (dict): Optimization information

        """
        with torch.no_grad():
            r = torch.cat([item.reward.float() for item in sampled])
            s_prime = torch.cat([item.s_prime for item in sampled])
            s = torch.cat([item.s for item in sampled])
            a = torch.cat([item.a.long() for item in sampled])
            d = torch.cat([item.done.float() for item in sampled])
        masking = self.sample_mask(d)

        y_hat = self.y_hat(s, a,s_prime,r,masking)
        y = self.y(s_prime, masking, r, y_hat,s,a)
        self.opt.zero_grad()
        loss = self.loss_func(y, y_hat)

        if self.training:
            loss.backward()
            if self.do_grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), self.gradient_clipping_norm)
                for param in self.action_model.parameters():
                    if param.grad is not None: param.grad.data.clamp_(-1, 1)
            self.opt.step()

        with torch.no_grad():
            self.loss = loss
            post_info = {'td_error': to_detach(y - y_hat).cpu().numpy()}
            return post_info

    def y_hat(self, s, a,s_prime,r,masking):
        return self.action_model(s).gather(1, a)

    def y(self, s_prime, masking, r, y_hat,s,a):
        return self.discount * self.action_model(s_prime).max(1)[0].unsqueeze(1) * masking + r.expand_as(y_hat)


class FixedTargetDQNModule(DQNModule):
    def __init__(self, ni: int, ao: int, layers: Collection[int], tau=1, **kwargs):
        super().__init__(ni, ao, layers, **kwargs)
        self.name = 'Fixed Target DQN'
        self.tau = tau
        self.target_model = deepcopy(self.action_model)

    def target_copy_over(self):
        r""" Updates the target network from calls in the FixedTargetDQNTrainer callback."""
        # self.target_net.load_state_dict(self.action_model.state_dict())
        # for target_param, local_param in zip(self.target_model.parameters(), self.action_model.parameters()):
        #     target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        self.target_model.load_state_dict(self.action_model.state_dict())

    def y(self, s_prime, masking, r, y_hat,s,a):
        r"""
        Uses the equation:

        .. math::

                Q^{*}(s, a) = \mathbb{E}_{s'∼ \Big\epsilon} \Big[r + \lambda \displaystyle\max_{a'}(Q^{*}(s' , a'))
                \;|\; s, a \Big]

        """
        return self.discount * self.target_model(s_prime).max(1)[0].unsqueeze(1) * masking + r.expand_as(y_hat)


class DoubleDQNModule(FixedTargetDQNModule):
    def __init__(self, ni: int, ao: int, layers: Collection[int], **kwargs):
        super().__init__(ni, ao, layers, **kwargs)
        self.name = 'DDQN'

    def y(self, s_prime, masking, r, y_hat,s,a):
        return self.discount * self.target_model(s_prime).gather(1, self.action_model(s_prime).argmax(1).unsqueeze(
            1)) * masking + r.expand_as(y_hat)


class DuelingBlock(nn.Module):
    def __init__(self, ao, stream_input_size,lin_cls=nn.Linear):
        super().__init__()

        self.val = lin_cls(stream_input_size, 1)
        self.adv = lin_cls(stream_input_size, ao)

    def forward(self, xi):
        r"""Splits the base neural net output into 2 streams to evaluate the advantage and v of the s space and
        corresponding actions.

        .. math::
           Q(s,a;\; \Theta, \\alpha, \\beta) = V(s;\; \Theta, \\beta) + A(s, a;\; \Theta, \\alpha) - \\frac{1}{|A|}
           \\Big\\sum_{a'} A(s, a';\; \Theta, \\alpha)

        """
        val, adv = self.val(xi), self.adv(xi)
        xi = val.expand_as(adv) + (adv - adv.mean()).squeeze(0)
        return xi


class DuelingDQNModule(FixedTargetDQNModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Dueling DQN'

    def setup_linear_block(self, _layers, ni, nc, w, h, emb_szs, layers, ao):
        tabular_model = TabularModel(emb_szs=emb_szs, n_cont=ni if not emb_szs else 0, layers=layers, out_sz=ao,
                             use_bn=self.batch_norm,lin_cls=self.lin_cls)
        if not emb_szs: tabular_model.embeds = None
        if not self.batch_norm: tabular_model.bn_cont = FakeBatchNorm()
        tabular_model.layers, removed_layer = split_model(tabular_model.layers, [last_layer(tabular_model)])
        ni = removed_layer[0].in_features
        self.action_model.add_module('lin_block', TabularEmbedWrapper(tabular_model))
        self.action_model.add_module('dueling_block', DuelingBlock(ao, ni))


class DoubleDuelingModule(DuelingDQNModule, DoubleDQNModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'DDDQN'


def distributional_loss_fn(s_log_sm_v,proj_distr_v):
    loss= (-s_log_sm_v*proj_distr_v)
    return loss.sum(dim=1).mean()


# class DistributionalDQN(FixedTargetDQNModule):
#     def __init__(self,ao,n_atoms=51,v_min=-10,v_max=10,**kwargs):
#         self.z_delta=(v_max-v_min)/(n_atoms-1)
#         self.n_atoms=n_atoms
#         self.v_min=v_min
#         self.v_max=v_max
#         super().__init__(ao=ao*n_atoms,**kwargs)
#         self.name='Distributional DQN'
#         # self.sm=nn.Softmax(dim=1)
#
#         self.loss_func=distributional_loss_fn
#
#     def init_weights(self, m):pass
#
#     def setup_linear_block(self, **kwargs):
#         super(DistributionalDQN,self).setup_linear_block(**kwargs)
#         self.action_model.register_buffer('supports', torch.arange(self.v_min, self.v_max+self.z_delta, self.z_delta))
#         self.action_model.add_module('softmax_buff',nn.Softmax(dim=1))
#
#     def both(self,xi,use_target=False):
#         if not use_target: cat_out=self(xi,False)
#         else:              cat_out=self.target_model(xi).view(xi.size()[0],-1,self.n_atoms)
#         probs=self.apply_softmax(cat_out,use_target)
#         if not use_target: weights=probs*self.action_model.supports
#         else:              weights=probs*self.target_model.supports
#         res=weights.sum(dim=2)
#         return cat_out,res
#
#     def q_vals(self,xi):
#         return self.both(xi)[1]
#
#     def apply_softmax(self,t,use_target=False):
#         if not use_target: return self.action_model.softmax_buff(t.view(-1,self.n_atoms)).view(t.size())
#         return self.target_model.softmax_buff(t.view(-1,self.n_atoms)).view(t.size())
#
#     def y(self, s_prime, masking, r, y_hat,s,a):
#         distr_v=self(s,only_q=False)
#         state_action_values=distr_v[range(s.size()[0]), a.data]
#         state_log_sm_v=F.log_softmax(state_action_values, dim=1)
#         return state_log_sm_v
#
#     def y_hat(self, s, a,s_prime,r,masking):
#         next_distr_v, next_q_vals_v=self.both(s_prime,True) # target
#         next_actions=next_q_vals_v.max(1)[1].data.cpu().numpy()
#         next_distr=self.apply_softmax(next_distr_v,True).data.cpu() # target
#         next_best_distr=next_distr[range(s_prime.size()[0]),next_actions]
#         proj_distr=distr_projection(next_best_distr,r,masking,self.v_min,self.v_max,self.n_atoms,self.discount)
#         proj_distr_v=torch.tensor(proj_distr).to(device=self.action_model.supports.device)
#         return proj_distr_v
#
#     def forward(self, xi: Tensor,only_q=True):
#         return self.q_vals(xi) if only_q else super(DistributionalDQN,self).forward(xi).view(xi.size()[0],-1,self.n_atoms)

class DistributionalDQNModule(nn.Module):
    def __init__(self, input_shape, n_actions,n_atoms=51,v_min=-10,v_max=10,):
        super(DistributionalDQNModule, self).__init__()
        self.n_atoms=n_atoms
        self.v_min=v_min
        self.v_max=v_max
        self.z_delta=(v_max-v_min)/(n_atoms-1)

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * self.n_atoms)
        )

        self.register_buffer("supports", torch.arange(self.v_min, self.v_max+self.z_delta, self.z_delta))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size()[0]
        fc_out = self.fc(x.float())
        return fc_out.view(batch_size, -1, self.n_atoms)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x): return self.both(x)[1]
    def apply_softmax(self, t): return self.softmax(t.view(-1, self.n_atoms)).view(t.size())

class DistributionalDQN(FixedTargetDQNModule):
    def __init__(self,ao,n_atoms=51,v_min=-10,v_max=10,**kwargs):
        self.z_delta=(v_max-v_min)/(n_atoms-1)
        self.n_atoms=n_atoms
        self.v_min=v_min
        self.v_max=v_max
        super().__init__(ao=ao,**kwargs)
        self.do_grad_clipping=False
        self.name='Distributional DQN'
        # self.sm=nn.Softmax(dim=1)

        self.loss_func=distributional_loss_fn

    def init_weights(self, m):pass

    def setup_linear_block(self, _layers, ni, nc, w, h, emb_szs, layers, ao,**kwargs):
        self.action_model=DistributionalDQNModule(ni,ao)

    def optimize(self, sampled):
        with torch.no_grad():
            r = torch.cat([item.reward.float() for item in sampled]).flatten().cpu().numpy()
            s_prime = torch.cat([item.s_prime for item in sampled])
            s = torch.cat([item.s for item in sampled])
            a = torch.cat([item.a.long() for item in sampled])
            d = torch.cat([item.done.float() for item in sampled]).flatten().cpu().numpy()
        # masking = self.sample_mask(d)

        batch_size=len(r)

        # next state distribution
        next_distr_v, next_qvals_v=self.target_model.both(s_prime)
        next_actions=next_qvals_v.max(1)[1].data.cpu().numpy()
        next_distr=self.target_model.apply_softmax(next_distr_v).data.cpu().numpy()

        next_best_distr=next_distr[range(batch_size), next_actions]
        dones=d.astype(np.bool)

        # project our distribution using Bellman update
        proj_distr=distr_projection(next_best_distr, r, dones, self.v_min, self.v_max, self.n_atoms, self.discount)

        # calculate net output
        distr_v=self.action_model(s)
        state_action_values=distr_v[range(batch_size), a.data]
        state_log_sm_v=F.log_softmax(state_action_values, dim=1)
        proj_distr_v=torch.tensor(proj_distr).to(self.action_model.supports.device)

        loss=-state_log_sm_v*proj_distr_v
        loss= loss.sum(dim=1).mean()

        with torch.no_grad():
            self.loss = loss
            _,y=self.action_model.both(s.to(device=self.action_model.supports.device))
            post_info = {'td_error': to_detach(y - next_qvals_v).cpu().numpy()}
            return post_info

    def q_vals(self,xi):
        return self.action_model.both(xi)[1]

    # def y(self, s_prime, masking, r, y_hat,s,a):
    #     distr_v=self(s,only_q=False)
    #     state_action_values=distr_v[range(s.size()[0]), a.data]
    #     state_log_sm_v=F.log_softmax(state_action_values, dim=1)
    #     return state_log_sm_v
    #
    # def y_hat(self, s, a,s_prime,r,masking):
    #     next_distr_v, next_q_vals_v=self.both(s_prime,True) # target
    #     next_actions=next_q_vals_v.max(1)[1].data.cpu().numpy()
    #     next_distr=self.apply_softmax(next_distr_v,True).data.cpu() # target
    #     next_best_distr=next_distr[range(s_prime.size()[0]),next_actions]
    #     proj_distr=distr_projection(next_best_distr,r,masking,self.v_min,self.v_max,self.n_atoms,self.discount)
    #     proj_distr_v=torch.tensor(proj_distr).to(device=self.action_model.supports.device)
    #     return proj_distr_v

    def forward(self, xi: Tensor,only_q=True):
        bs=xi.size()[0]
        return self.q_vals(xi) if only_q else super(DistributionalDQN,self).forward(xi).view(bs,-1,self.n_atoms)