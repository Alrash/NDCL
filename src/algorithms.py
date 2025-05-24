import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

import networks
from lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, l2_between_dicts, proj, Nonparametric
)


ALGORITHMS = [
    'ERM',
    'NDCL',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    'Fish',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'Transfer',
    'CausIRL_CORAL',
    'CausIRL_MMD',
    'EQRM',
    'RDM',
    'PGrad',
    'TCRI_HSIC',
    'Focal',
    'ReWeight',
    'BSoftmax',
    'LDAM',
    'BoDA',
    'GINIDG',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def embedding(self, x):
        return self.featurizer(x)


class NDCL(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(NDCL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.n_class = num_classes
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _contrastive_loss(mu, y_ind):
        # remove one instance of each class
        num = np.zeros_like(y_ind)
        for k in np.unique(y_ind):
            num[y_ind == k] = np.sum(y_ind == k)
        # end for k
        mu, y_ind, num = mu[num != 1], y_ind[num != 1], num[num != 1]

        loss = torch.tensor([0.], device=mu.device)
        if num.size != 0:
            # calc cos: dxk * dxk
            cos = torch.cosine_similarity(mu.unsqueeze(1), mu.unsqueeze(0), dim=-1)
            # remove diag and calc softmax for each row
            cos = torch.triu(cos, 1)[:, 1:] + torch.tril(cos, -1)[:, :-1]
            cos = F.softmax(cos, dim=-1)

            ind = np.repeat(y_ind[np.newaxis, :], y_ind.size, axis=0)
            ind = np.triu(ind, 1)[:, 1:] + np.tril(ind, -1)[:, :-1]

            # calc loss
            for row in range(y_ind.shape[0]):
                loss += -torch.log(torch.sum(cos[row][ind[row] == y_ind[row]])) / num[row]
            # end for row
            loss /= np.unique(y_ind).size
        return loss

    @staticmethod
    def _margin_loss(pos, neg, pos_y, neg_y):
        loss = torch.tensor([0.], device=pos.device)
        for k in torch.unique(pos_y):
            num = torch.sum(pos_y == k)
            # extract instances from each class
            matrix = torch.cat([pos[pos_y == k], neg[neg_y == k]], dim=0)
            cos = 1 - torch.cosine_similarity(matrix.unsqueeze(1), matrix.unsqueeze(0), dim=-1)
            cos = torch.triu(cos, 1)[:, 1:] + torch.tril(cos, -1)[:, :-1]
            cos = F.softmax(cos[:num], dim=-1)

            # calc loss: [sum: frac => -log: con loss for each row => mean: mean con loss]
            loss += torch.mean(-torch.log(torch.sum(cos[:, num - 1:], dim=-1)))
        # end for k
        return loss / torch.unique(pos_y).size(0)

    def _aug2(self, x, y, pred):
        def lamd(alpha):
            return np.random.beta(alpha, alpha)

        def obtain_num_from_pred(pred, k, thred):
            pos = torch.where(pred[:, k] >= thred)[0]
            return pos[-1].item() + 1 if pos.size(0) else 0

        if torch.unique(y).size(0) == 1:
            return None, None

        # setting
        alpha, total, num_gen = self.hparams['mixup'], x.size(0), {}
        mode = self.hparams['aug_mode'].lower() if 'aug_mode' in self.hparams.keys() else 'rev'
        assert mode in ['set', 'mean', 'rev']
        if mode != 'set':
            scaler = self.hparams['aug_scaler'] if 'aug_scaler' in self.hparams.keys() else 1.0
            scaler = 1.0

        # calc the number of data augmentation for each class
        if mode == 'set':
            num = np.max([np.min([self.hparams['batch_size'] // 2, 24]), 8])
            num_gen = {k.item(): num for k in torch.unique(y)}
        elif mode == 'mean':
            num = int(np.ceil(total * scaler / torch.unique(y).size(0)))
            redundant, classes, n_class = num * torch.unique(y).size(0) - int(total * scaler), [], []
            for k in torch.unique(y):
                classes.append(k), n_class.append(y[y == k].size(0))
            ind = np.argsort(n_class)
            for i in ind:
                num_gen[classes[i].item()] = num - 1 if redundant > 0 else num
                redundant -= 1
        else:
            total, classes, n_class = int(total * scaler), [], []
            for k in torch.unique(y):
                classes.append(k), n_class.append(y[y == k].size(0) / y.size(0))
            ratio, start, cum = np.exp(n_class) / np.exp(n_class).sum(), 0, 0
            for ind in range(len(classes)):
                cum += ratio[ind]
                num_gen[classes[ind].item()] = int(np.ceil(cum * total)) - start
                start += num_gen[classes[ind].item()]
        # end calc

        # generate data for each class
        lam, device, aug, aug_y, n_class = lamd(alpha), x.device, [], [], pred.size(-1)
        for k in torch.unique(y):
            aug_pos, pos, neg, pred_pos, pred_neg = [], x[y == k], x[y != k], pred[y == k], pred[y != k]
            # asc sort neg though pred_neg[:, k]
            _, ind = torch.sort(pred_neg[:, k], descending=True)
            neg, pred_neg = neg[ind], pred_neg[ind]
            # dec sort pos though pred_pos[:, k]
            _, ind = torch.sort(pred_pos[:, k])
            pos, pred_pos = pos[ind], pred_pos[ind]
            # find neg mix instances (> 2/K)
            num = obtain_num_from_pred(pred_neg, k, 2 / n_class)
            large_num = obtain_num_from_pred(pred_neg, k, 1 / n_class)
            if pos.size(0) == 1:
                # n_min: minimum amount of instances
                lam_1, n_sampling, n_min = lam, neg.size(0) // 2, int(np.ceil(num_gen[k.item()] / 3))
                # choose the number of sampling
                if n_min <= num:
                    n_sampling = np.min([num, n_sampling]) if n_min <= n_sampling else num
                elif n_min <= n_sampling:
                    # num < n_min <= n_sampling
                    n_sampling = n_sampling
                elif n_min <= large_num:
                    n_sampling = large_num
                else:
                    n_sampling = neg.size(0)
                # end if [choose]
                ending = 0
                while True:
                    # generate instances
                    gen = lam_1 * pos + (1 - lam_1) * neg[:n_sampling]
                    aug_pos.append(gen)
                    lam_1, ending = lamd(alpha), ending + n_sampling
                    if ending >= num_gen[k.item()]:
                        break
                # end while [generate]
            else:
                # amount of positive instances > 1
                n_sampling = 0
                for per in [0.25, 0.5, 0.75]:
                    n_pos, is_diversity = int(np.ceil(pos.size(0) * per)), False
                    if n_pos == 1:
                        continue
                    # number of sampling
                    if num_gen[k.item()] <= n_pos * num:
                        n_sampling = num
                    elif num_gen[k.item()] <= n_pos * large_num:
                        n_sampling = large_num
                    else:
                        is_diversity, n_neg = True, int(np.ceil(num_gen[k.item()] / n_pos))
                        if num > 0 and n_neg / num >= 0.25:
                            n_sampling = num
                        elif large_num > 0 and n_neg / large_num >= 0.25:
                            n_sampling = large_num
                    # end if num_gen
                    if n_sampling != 0:
                        break
                # end for per
                # adjust amount of sampling instances from neg
                n_sampling = n_sampling if n_sampling != 0 else \
                    (neg.size(0) if neg.size(0) < int(np.ceil(num_gen[k.item()] / n_pos)) else int(np.ceil(num_gen[k.item()] / n_pos)))
                ending, pos, neg = 0, pos[:n_pos], neg[:n_sampling]
                while True:
                    if ending != 0 and is_diversity:
                        lam_1 = lamd(alpha)
                        gen_neg = (lam_1 * neg[np.random.permutation(n_sampling)]
                                   + (1 - lam_1) * neg[np.random.permutation(n_sampling)])
                    else:
                        gen_neg = neg
                    # enf if [choose negative instances]
                    # generate instances
                    gen = []
                    for i in range(n_pos):
                        gen.append(lam * pos[i:i + 1] + (1 - lam) * gen_neg)
                    aug_pos.append(torch.cat(gen, dim=0))
                    ending += aug_pos[-1].size(0)
                    if ending >= num_gen[k.item()]:
                        break
                # end while [generate]
            # end if [whole generate phase]
            aug.append(torch.cat(aug_pos, dim=0)[:num_gen[k.item()]]), aug_y.append(k * torch.ones(num_gen[k.item()], device=device))
        # end for k
        return torch.cat(aug, dim=0), torch.cat(aug_y, dim=0)

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])
        all_d = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        # calc all predict results
        pred = self.predict(all_x)
        each_loss = F.cross_entropy(pred, all_y, reduction='none')
        pred = F.softmax(pred, dim=-1)

        cls_loss = torch.tensor([0.], device=device)
        for k in torch.unique(all_y):
            li = each_loss[all_y == k]
            reweight = self.weights_per_env[all_d[all_y == k], k]
            penalty = F.softmax(li.detach(), dim=-1)
            cls_loss += torch.sum(li * penalty * reweight)
        cls_loss /= torch.unique(all_y).size(-1)

        # margin loss
        # margin_loss = torch.tensor([0.], device=device)
        aug, aug_y = self._aug2(all_x, all_y, pred)
        if aug is not None:
            aug_pred = F.softmax(self.predict(aug), dim=-1)
            margin_loss = self._margin_loss(pred, aug_pred, all_y, aug_y)
        else:
            margin_loss = torch.tensor([0.], device=device)

        # align_loss = torch.tensor([0.], device=device)
        center, ind, n_domain = [], [], len(minibatches)
        for d in torch.unique(all_d):
            di, yi = pred[all_d == d], all_y[all_d == d]
            for k in torch.unique(yi):
                center.append(torch.mean(di[yi == k], dim=0, keepdim=True)), ind.append(k.item())
        align_loss = self._contrastive_loss(torch.cat(center, dim=0), np.array(ind))

        loss = cls_loss + self.hparams['alpha'] * margin_loss + self.hparams['beta'] * align_loss

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'loss_cls': cls_loss.item(),
                'loss_margin': margin_loss.item(), 'loss_align': align_loss.item()}


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class PGrad(Fish):
    """
    Learning Principal Gradients for Domain Generalization
    https://openreview.net/forum?id=CgCmwcfgEdH
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(PGrad, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.network = networks.WholeFish(self.input_shape, self.num_classes, self.hparams)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=0.1,
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None
        self.split_num = 3
        self.global_update = 0

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                                weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams['lr']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def transpose_pca(self, stack_classifier, comb=True):
        stack_classifier = torch.cat(stack_classifier)
        mean_classifier = stack_classifier.mean(dim=0)
        centerized_classifier = stack_classifier - mean_classifier
        cov_classifier = centerized_classifier @ centerized_classifier.T / (stack_classifier.size(1) - 1)
        pr_direction, pr_value, _ = torch.svd(cov_classifier)
        pr_direction = centerized_classifier.T @ pr_direction
        pr_direction = pr_direction / pr_direction.norm(dim=0)
        return pr_direction, pr_value

    def stack_and_pca(self, envs_weights):
        new_stack = [[] for _ in range(self.split_num * self.num_domains + 1)]
        for i in range(self.split_num * self.num_domains + 1):
            new_stack[i] = [envs_weights[ele]['value'][i].view(1, -1) for ele in envs_weights.keys()]
            new_stack[i] = torch.cat(new_stack[i], dim=1)
        pr_direction, pr_value = self.transpose_pca(new_stack)
        return pr_direction, pr_value

    def principal_grad(self, meta_weights, inner_weights, params_stack):

        if True:
            meta_weights = ParamDict(meta_weights)
            inner_weights = ParamDict(inner_weights)
            diff_weights = meta_weights - inner_weights
            norm_diff = sum([ele.pow(2).sum() for ele in diff_weights.values()])
            norm_diff = norm_diff.sqrt()
            principle_dir, pr_value = self.stack_and_pca(params_stack)
            principle_dir *= norm_diff  # length calibration
            grad_mask = torch.zeros_like(pr_value)
            start_index = 0
            for name, value in self.network.named_parameters():
                param_size = value.numel()
                end_index = start_index + param_size
                pra_grad = principle_dir[start_index:end_index, :]
                cali_direction = diff_weights[name]
                cali_mask = (cali_direction.flatten().unsqueeze(1) * pra_grad).sum(0)
                grad_mask += cali_mask
                start_index = end_index

            cali_mask = 2 * (grad_mask > 0).float() - 1
            pra_grad = cali_mask * principle_dir  # direction calibration
            comb_num = 4
            comb_coef = pr_value[:comb_num] / pr_value[:comb_num].norm()
            pra_grad = (comb_coef * pra_grad[:, :comb_num]).sum(1)  # direction ensemble
            start_index = 0
            # Learning PGrad for model update
            for name, value in self.network.named_parameters():
                param_size = value.numel()
                end_index = start_index + param_size
                value_grad = pra_grad[start_index:end_index].view(value.size())
                start_index = end_index
                value.grad = value_grad.clone()

    def update(self, minibatches, unlabeled=None):

        self.create_clone(minibatches[0][0].device)
        params_stack = {key: {'value': []} for key, _ in self.network.named_parameters()}
        range_list = np.arange(0, len(minibatches))
        range_list = range_list[np.random.permutation(len(minibatches))]
        for i in range(self.split_num):
            "Trajectory Sampling"
            for num, index in enumerate(range_list.tolist()):
                x, y = minibatches[index]
                x = x[(i * x.size(0) // self.split_num):((i + 1) * x.size(0) // self.split_num)]
                y = y[i * y.size(0) // self.split_num:(i + 1) * y.size(0) // self.split_num]
                loss = F.cross_entropy(self.network_inner(x), y)
                self.optimizer_inner.zero_grad()
                loss.backward()
                for (key, _), env_value in zip(params_stack.items(), self.network_inner.parameters()):
                    params_stack[key]['value'].append(copy.deepcopy(env_value).unsqueeze(dim=0))
                self.optimizer_inner.step()
            range_list = range_list[np.random.permutation(len(minibatches))]
        for (key, _), env_value in zip(params_stack.items(), self.network_inner.parameters()):
            params_stack[key]['value'].append(copy.deepcopy(env_value).unsqueeze(0))

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        self.optimizer.zero_grad()

        "PGrad Learning"
        self.principal_grad(self.network.named_parameters(), self.network_inner.named_parameters(), params_stack)
        self.optimizer.step()
        self.global_update += 1
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def embedding(self, x):
        return self.network.embedding(x)


class RDM(ERM):
    """d"""
    """Domain Generalization via Risk Distribution Matching (we used to be named it DGPM2)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RDM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def my_cdist(self, x1, x2):  # (bs)
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)

        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    @staticmethod
    def _moment_penalty(p_mean, q_mean, p_var, q_var):
        return (p_mean - q_mean) ** 2 + (p_var - q_var) ** 2

    @staticmethod
    def _kl_penalty(p_mean, q_mean, p_var, q_var):
        return 0.5 * torch.log(q_var / p_var) + ((p_var) + (p_mean - q_mean) ** 2) / (2 * q_var) - 0.5

    def _js_penalty(self, p_mean, q_mean, p_var, q_var):
        m_mean = (p_mean + q_mean) / 2
        m_var = (p_var + q_var) / 4

        return self._kl_penalty(p_mean, m_mean, p_var, m_var) + self._kl_penalty(q_mean, m_mean, q_var, m_var)

    def update(self, minibatches, unlabeled=None, held_out_minibatches=None):
        matching_penalty_weight = (self.hparams['rdm_lambda'] if self.update_count
                                                                 >= self.hparams['rdm_penalty_anneal_iters'] else
                                   0.)

        variance_penalty_weight = (self.hparams['variance_weight'] if self.update_count
                                                                      >= self.hparams['rdm_penalty_anneal_iters'] else
                                   0.)

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.predict(all_x)
        losses = torch.zeros(len(minibatches)).cuda()
        all_logits_idx = 0
        all_confs_envs = None

        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            losses[i] = F.cross_entropy(logits, y)

            nll = F.cross_entropy(logits, y, reduction="none").unsqueeze(0)

            if all_confs_envs is None:
                all_confs_envs = nll
            else:
                all_confs_envs = torch.cat([all_confs_envs, nll], dim=0)

        erm_loss = losses.mean()

        ## squeeze the risks
        all_confs_envs = torch.squeeze(all_confs_envs)  # (3, bs, 7) or (3, bs)

        ## find the worst domain
        worst_env_idx = torch.argmax(torch.clone(losses))
        all_confs_worst_env = all_confs_envs[worst_env_idx]  # (bs, 7)

        ## flatten the risk
        all_confs_worst_env_flat = torch.flatten(all_confs_worst_env)
        all_confs_all_envs_flat = torch.flatten(all_confs_envs)

        matching_penalty = self.mmd(all_confs_worst_env_flat.unsqueeze(1), all_confs_all_envs_flat.unsqueeze(1))

        ## variance penalty
        variance_penalty = torch.var(all_confs_all_envs_flat)
        variance_penalty += torch.var(all_confs_worst_env_flat)

        total_loss = erm_loss + matching_penalty_weight * matching_penalty + variance_penalty_weight * variance_penalty

        if self.update_count == self.hparams['rdm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["rdm_lr"],
                weight_decay=self.hparams['weight_decay'])

        # Step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.update_count += 1

        return {'update_count': self.update_count.item(), 'total_loss': total_loss.item(), 'erm_loss': erm_loss.item(),
                'matching_penalty': matching_penalty.item(), 'variance_penalty': variance_penalty.item(), 'rdm_lambda': self.hparams['rdm_lambda']}


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        input_grad = autograd.grad(
            F.cross_entropy(disc_out, disc_labels, reduction='sum'),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_meta_test = hparams['n_meta_test']

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches, self.num_meta_test):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}

class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)

            env_grads = autograd.grad(env_loss, self.network.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, in_features, num_classes, num_domains, hparams):
        super(IGA, self).__init__(in_features, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(env_loss, self.network.parameters(),
                                        create_graph=True)

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(mean_loss, self.network.parameters(),
                                        retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['penalty'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': mean_loss.item(), 'penalty': penalty_value.item()}


class SelfReg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SelfReg, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.featurizer.n_outputs
        hidden_size = input_feat_size if input_feat_size==2048 else input_feat_size*2

        self.cdpl = nn.Sequential(
                            nn.Linear(input_feat_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, input_feat_size),
                            nn.BatchNorm1d(input_feat_size)
        )

    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        lam = np.random.beta(0.5, 0.5)

        batch_size = all_y.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex==val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y

        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)

        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end-ex)+ex
            shuffle_indices2 = torch.randperm(end-ex)+ex
            for idx in range(end-ex):
                output_2[idx+ex] = output[shuffle_indices[idx]]
                feat_2[idx+ex] = proj[shuffle_indices[idx]]
                output_3[idx+ex] = output[shuffle_indices2[idx]]
                feat_3[idx+ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam*output_2 + (1-lam)*output_3
        feat_3 = lam*feat_2 + (1-lam)*feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale*(lam*(L_ind_logit + L_ind_feat)+(1-lam)*(L_hdl_logit + L_hdl_feat))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.k = hparams["k"]
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            betas=betas
        )

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):

        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)
            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        # gradient masking applied here
        self.mask_grads(param_gradients, self.network.parameters())
        self.optimizer.step()
        self.update_count += 1

        return {'loss': mean_loss}

    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))



class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)

class TRM(Algorithm):
    """
    Learning Representations that Support Robust Transfer of Predictors
    <https://arxiv.org/abs/2110.09940>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains,hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.num_domains = num_domains
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes).cuda()
        self.clist = [nn.Linear(self.featurizer.n_outputs, num_classes).cuda() for i in range(num_domains+1)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(num_domains+1)]

        self.optimizer_f = torch.optim.Adam(
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # initial weights
        self.alpha = torch.ones((num_domains, num_domains)).cuda() - torch.eye(num_domains).cuda()

    @staticmethod
    def neum(v, model, batch):
        def hvp(y, w, v):

            # First backprop
            first_grads = autograd.grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
            first_grads = torch.nn.utils.parameters_to_vector(first_grads)
            # Elementwise products
            elemwise_products = first_grads @ v
            # Second backprop
            return_grads = autograd.grad(elemwise_products, w, create_graph=True)
            return_grads = torch.nn.utils.parameters_to_vector(return_grads)
            return return_grads

        v = v.detach()
        h_estimate = v
        cnt = 0.
        model.eval()
        iter = 10
        for i in range(iter):
            model.weight.grad *= 0
            y = model(batch[0].detach())
            loss = F.cross_entropy(y, batch[1].detach())
            hv = hvp(loss, model.weight, v)
            v -= hv
            v = v.detach()
            h_estimate = v + h_estimate
            h_estimate = h_estimate.detach()
            # not converge
            if torch.max(abs(h_estimate)) > 10:
                break
            cnt += 1

        model.train()
        return h_estimate.detach()

    def update(self, minibatches, unlabeled=None):

        loss_swap = 0.0
        trm = 0.0

        if self.update_count >= self.hparams['iters']:
            # TRM
            if self.hparams['class_balanced']:
                # for stability when facing unbalanced labels across environments
                for classifier in self.clist:
                    classifier.weight.data = copy.deepcopy(self.classifier.weight.data)
            self.alpha /= self.alpha.sum(1, keepdim=True)

            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            # updating original network
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

            for i in range(30):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature.detach()), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            # collect (feature, y)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            # estimate transfer risk
            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)

                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)

                loss_P = [F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])*(self.alpha[Q, i].data.detach())
                          if i in sample_list else 0. for i in range(len(minibatches))]
                loss_P_sum = sum(loss_P)
                grad_P = autograd.grad(loss_P_sum, self.clist[Q].weight, create_graph=True)
                vec_grad_P = nn.utils.parameters_to_vector(grad_P).detach()
                vec_grad_P = self.neum(vec_grad_P, self.clist[Q], (feature_split[Q], y_split[Q]))

                loss_swap += loss_P_sum - self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q)

                for i in sample_list:
                    self.alpha[Q, i] *= (self.hparams["groupdro_eta"] * loss_P[i].data).exp()

            loss_swap /= len(minibatches)
            trm /= len(minibatches)
        else:
            # ERM
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

        nll = loss.item()
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams['iters']:
            loss_swap = (loss + loss_swap)
        else:
            loss_swap = loss

        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        loss_swap = loss_swap.item() - nll
        self.update_count += 1

        return {'nll': nll, 'trm_loss': loss_swap}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def train(self):
        self.featurizer.train()

    def eval(self):
        self.featurizer.eval()

class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IB_penalty': ib_penalty.item()}

class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        irm_penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        irm_penalty = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            irm_penalty += self._irm_penalty(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        irm_penalty /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(),
                'nll': nll.item(),
                'IRM_penalty': irm_penalty.item(),
                'IB_penalty': ib_penalty.item()}


class AbstractCAD(Algorithm):
    """Contrastive adversarial domain bottleneck (abstract class)
    from Optimal Representations for Covariate Shift <https://arxiv.org/abs/2201.00057>
    """

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, is_conditional):
        super(AbstractCAD, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        params = list(self.featurizer.parameters()) + list(self.classifier.parameters())

        # parameters for domain bottleneck loss
        self.is_conditional = is_conditional  # whether to use bottleneck conditioned on the label
        self.base_temperature = 0.07
        self.temperature = hparams['temperature']
        self.is_project = hparams['is_project']  # whether apply projection head
        self.is_normalized = hparams['is_normalized'] # whether apply normalization to representation when computing loss

        # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        # the two versions have the same optima, but we find the latter is more stable
        self.is_flipped = hparams["is_flipped"]

        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 128),
            )
            params += list(self.project.parameters())

        # Optimizers
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def bn_loss(self, z, y, dom_labels):
        """Contrastive based domain bottleneck loss
         The implementation is based on the supervised contrastive loss (SupCon) introduced by
         P. Khosla, et al., in “Supervised Contrastive Learning“.
        Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
        """
        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if not self.is_conditional:
            # unconditional CAD loss
            denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
            log_prob = logits - denominator

            mask_valid = (mask_y.sum(1) > 0)
            log_prob = log_prob[mask_valid]
            mask_d = mask_d[mask_valid]

            if self.is_flipped:  # maximize log prob of samples from different domains
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # conditional CAD loss
            if self.is_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            logits = logits[mask_valid]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if self.is_flipped:  # maximize log prob of samples from different domains and with same label
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return finite_mean(bn_loss)

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        bn_loss = self.bn_loss(all_z, all_y, all_d)
        clf_out = self.classifier(all_z)
        clf_loss = F.cross_entropy(clf_out, all_y)
        total_loss = clf_loss + self.hparams['lmbda'] * bn_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"clf_loss": clf_loss.item(), "bn_loss": bn_loss.item(), "total_loss": total_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class CAD(AbstractCAD):
    """Contrastive Adversarial Domain (CAD) bottleneck

       Properties:
       - Minimize I(D;Z)
       - Require access to domain labels but not task labels
       """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=False)


class CondCAD(AbstractCAD):
    """Conditional Contrastive Adversarial Domain (CAD) bottleneck

    Properties:
    - Minimize I(D;Z|Y)
    - Require access to both domain labels and task labels
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CondCAD, self).__init__(input_shape, num_classes, num_domains, hparams, is_conditional=True)


class Transfer(Algorithm):
    '''Algorithm 1 in Quantifying and Improving Transferability in Domain Generalization (https://arxiv.org/abs/2106.03632)'''
    ''' tries to ensure transferability among source domains, and thus transferabiilty between source and target'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Transfer, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.d_steps_per_g = hparams['d_steps_per_g']

        # Architecture
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.adv_classifier.load_state_dict(self.classifier.state_dict())

        # Optimizers
        if self.hparams['gda']:
            self.optimizer = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr'])
        else:
            self.optimizer = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.adv_opt = torch.optim.SGD(self.adv_classifier.parameters(), lr=self.hparams['lr_d'])

    def loss_gap(self, minibatches, device):
        ''' compute gap = max_i loss_i(h) - min_j loss_j(h), return i, j, and the gap for a single batch'''
        max_env_loss, min_env_loss =  torch.tensor([-float('inf')], device=device), torch.tensor([float('inf')], device=device)
        for x, y in minibatches:
            p = self.adv_classifier(self.featurizer(x))
            loss = F.cross_entropy(p, y)
            if loss > max_env_loss:
                max_env_loss = loss
            if loss < min_env_loss:
                min_env_loss = loss
        return max_env_loss - min_env_loss

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        # outer loop
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del all_x, all_y
        gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
        self.optimizer.zero_grad()
        gap.backward()
        self.optimizer.step()
        self.adv_classifier.load_state_dict(self.classifier.state_dict())
        for _ in range(self.d_steps_per_g):
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
        return {'loss': loss.item(), 'gap': -gap.item()}

    def update_second(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count = (self.update_count + 1) % (1 + self.d_steps_per_g)
        if self.update_count.item() == 1:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss = F.cross_entropy(self.predict(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del all_x, all_y
            gap = self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            self.optimizer.zero_grad()
            gap.backward()
            self.optimizer.step()
            self.adv_classifier.load_state_dict(self.classifier.state_dict())
            return {'loss': loss.item(), 'gap': gap.item()}
        else:
            self.adv_opt.zero_grad()
            gap = -self.hparams['t_lambda'] * self.loss_gap(minibatches, device)
            gap.backward()
            self.adv_opt.step()
            self.adv_classifier = proj(self.hparams['delta'], self.adv_classifier, self.classifier)
            return {'gap': -gap.item()}


    def predict(self, x):
        return self.classifier(self.featurizer(x))


class AbstractCausIRL(ERM):
    '''Abstract class for Causality based invariant representation learning algorithm from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractCausIRL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        first = None
        second = None

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i] + 1e-16, targets[i])
            slice = np.random.randint(0, len(features[i]))
            if first is None:
                first = features[i][:slice]
                second = features[i][slice:]
            else:
                first = torch.cat((first, features[i][:slice]), 0)
                second = torch.cat((second, features[i][slice:]), 0)
        if len(first) > 1 and len(second) > 1:
            penalty = torch.nan_to_num(self.mmd(first, second))
        else:
            penalty = torch.tensor(0)
        objective /= nmb

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class CausIRL_MMD(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the MMD distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_MMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=True)


class CausIRL_CORAL(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the CORAL distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CausIRL_CORAL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, gaussian=False)


class EQRM(ERM):
    """
    Empirical Quantile Risk Minimization (EQRM).
    Algorithm 1 from [https://arxiv.org/pdf/2207.09944.pdf].
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, dist=None):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.register_buffer('alpha', torch.tensor(self.hparams["eqrm_quantile"], dtype=torch.float64))
        if dist is None:
            self.dist = Nonparametric()
        else:
            self.dist = dist

    def risk(self, x, y):
        return F.cross_entropy(self.network(x), y).reshape(1)

    def update(self, minibatches, unlabeled=None):
        env_risks = torch.cat([self.risk(x, y) for x, y in minibatches])

        if self.update_count < self.hparams["eqrm_burnin_iters"]:
            # Burn-in/annealing period uses ERM like penalty methods (which set penalty_weight=0, e.g. IRM, VREx.)
            loss = torch.mean(env_risks)
        else:
            # Loss is the alpha-quantile value
            self.dist.estimate_parameters(env_risks)
            loss = self.dist.icdf(self.alpha)

        if self.update_count == self.hparams['eqrm_burnin_iters']:
            # Reset Adam (like IRM, VREx, etc.), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["eqrm_lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        return {'loss': loss.item()}


class AbstractTCRI(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(AbstractTCRI, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

        self.spurious_featurizer = nn.Sequential(nn.Linear(self.featurizer.n_outputs,
            self.featurizer.n_outputs))
        self.domain_general_featurizer = nn.Sequential(nn.Linear(self.featurizer.n_outputs,
            self.featurizer.n_outputs))

        self.network = nn.Sequential(self.featurizer, self.domain_general_featurizer,
         self.classifier)

        self.spurious_classifiers = nn.ModuleList()
        for i in range(num_domains):
            self.spurious_classifiers.append(networks.Classifier(2*self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier']))

        self.optimizer = torch.optim.Adam(list(self.network.parameters()) + \
            list(self.spurious_featurizer.parameters())  + \
            list(self.spurious_classifiers.parameters()),
            lr = self.hparams['lr'],
            weight_decay=self.hparams['weight_decay']
          )

    def embedding(self, x):
        return self.domain_general_featurizer(self.featurizer(x))

    @staticmethod
    def tcri(X, Y, Z, sigma=1.):
        raise NotImplementedError()

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['tcri_beta'] if self.update_count
                          >= self.hparams['tcri_beta_anneal_iters'] else
                          1.0)

        nll = 0. # causal (domain general nll)
        tic_nll = 0. # total information criterion (domain specific nll)
        tcri_penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_logits = self.network(all_x)

        all_logits_idx = 0

        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)

            new_x = self.featurizer(x)

            phi_x = self.domain_general_featurizer(new_x) # domain general representation
            psi_x = self.spurious_featurizer(new_x) # domain specific representation

            tcri_penalty += self.tcri(phi_x, psi_x, y, sigma=None)

            latent_x = torch.cat([phi_x, psi_x], 1) # total information criterion

            anticausal_logits = self.spurious_classifiers[i](latent_x)
            tic_nll += F.cross_entropy(anticausal_logits, y)

        nll /= len(minibatches)
        tcri_penalty /= len(minibatches)
        tic_nll /= len(minibatches)

        loss = nll + self.hparams['tcri_alpha']*tic_nll + \
          penalty_weight * tcri_penalty

        if self.update_count == self.hparams['tcri_beta_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'tcri_penalty': tcri_penalty.item(),
            'tic_nll': tic_nll.item()}

    def predict(self, x):
        return self.network(x)


def centering(K):
  n = K.shape[0]
  unit = torch.ones([n, n]).to(K.device)
  I = torch.eye(n).to(K.device)
  Q = I - unit/n

  return torch.mm(torch.mm(Q, K), Q)

def rbf(X, sigma=None):
  GX = torch.mm(X, X.T).to(X.device)
  KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
  if sigma is None:
    mdist = torch.median(KX[KX != 0])
    sigma = torch.nan_to_num(torch.sqrt(mdist), nan=1.)
    KX *= - 0.5 / sigma / sigma
    KX = torch.exp(KX)
  return KX

def HSIC(X, Y, sigma=None):
  return torch.sum(centering(rbf(X))*centering(rbf(Y)))

class TCRI_HSIC(AbstractTCRI):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TCRI_HSIC, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

    @staticmethod
    def tcri(X, Y, Z, sigma=None):
      n = X.shape[0]

      unique_Z = torch.unique(Z)
      cov = torch.tensor(0.).to(X.device)
      for i in range(unique_Z.shape[0]):
        idx = (Z == unique_Z[i]).nonzero(as_tuple=True)[0]
        if len(idx) <= 1:
          continue
        x = X[idx]
        y = Y[idx]

        cov += HSIC(x, y)
      return cov / (unique_Z.shape[0] * n)


class Focal(ERM):
    """Focal loss, https://arxiv.org/abs/1708.02002"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Focal, self).__init__(input_shape, num_classes, num_domains, hparams)

    @staticmethod
    def focal_loss(input_values, gamma):
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values
        return loss.mean()

    def update(self, minibatches, env_feats=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = self.focal_loss(F.cross_entropy(self.predict(all_x), all_y, reduction='none'), self.hparams["gamma"])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class ReWeight(ERM):
    """Naive inverse re-weighting"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ReWeight, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.weights_per_env = {}

        env_labels = hparams['env_labels']
        if env_labels is not None:
            for i, env in enumerate(sorted(env_labels)):
                per_cls_weights = 1. / env_labels[env]
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes
                self.weights_per_env[i] = torch.FloatTensor(per_cls_weights)

    def update(self, minibatches, env_feats=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        loss = torch.tensor([0.], device=device)
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x), y, weight=self.weights_per_env[env].to(device))
        loss /= len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class BSoftmax(ERM):
    """Balanced softmax, https://arxiv.org/abs/2007.10740"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(BSoftmax, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.n_samples_per_env = {}
        env_labels = hparams['env_labels']
        if env_labels is not None:
            for i, env in enumerate(sorted(env_labels)):
                n_samples_per_cls = env_labels[env]
                n_samples_per_cls[n_samples_per_cls == np.inf] = 1
                self.n_samples_per_env[i] = torch.FloatTensor(n_samples_per_cls)

    def update(self, minibatches, env_feats=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            x = self.predict(x)
            spc = self.n_samples_per_env[env].type_as(x)
            spc = spc.unsqueeze(0).expand(x.shape[0], -1)
            x = x + spc.log()
            loss += F.cross_entropy(input=x, target=y)
        loss /= len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class LDAM(ERM):
    """LDAM loss, https://arxiv.org/abs/1906.07413"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(LDAM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.m_list = {}
        env_labels = hparams['env_labels']
        if env_labels is not None:
            for i, env in enumerate(sorted(env_labels)):
                m_list = 1. / np.sqrt(np.sqrt(env_labels[env]))
                m_list = m_list * (self.hparams['max_m'] / np.max(m_list))
                self.m_list[i] = torch.FloatTensor(m_list)

    def update(self, minibatches, env_feats=None):
        device, loss = minibatches[0][0].device, 0
        for env, (x, y) in enumerate(minibatches):
            x = self.predict(x)
            index = torch.zeros_like(x, dtype=torch.uint8)
            index.scatter_(1, y.data.view(-1, 1), 1)
            index_float = index.type(torch.FloatTensor)
            batch_m = torch.matmul(self.m_list[env][None, :].to(device), index_float.transpose(0, 1).to(device))
            batch_m = batch_m.view((-1, 1))
            x_m = x - batch_m
            output = torch.where(index, x_m, x)
            loss += F.cross_entropy(self.hparams["scale"] * output, y)
        loss /= len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class BoDA(ERM):
    """BoDA: balanced domain-class distribution alignment"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(BoDA, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.train_feats = None
        self.train_labels = None
        self.steps = 0
        self.nu = hparams["nu"]
        self.momentum = hparams["momentum"]
        self.temperature = hparams["temperature"]
        self.boda_start_step = hparams["boda_start_step"]
        self.feat_update_freq = hparams["feat_update_freq"]

        # 'env_labels' can be None in evaluation, but not in training
        env_labels = hparams['env_labels']
        if env_labels is not None:
            # number of samples per domain-class pair
            # self.n_samples_table = torch.tensor([env_labels[env] for env in sorted(env_labels)])
            self.register_buffer('n_samples_table', torch.tensor([env_labels[env] for env in sorted(env_labels)]))

            # self.centroid_classes = torch.tensor(np.hstack([np.unique(env_labels[env]) for env in sorted(env_labels)]))
            # self.centroid_envs = torch.tensor(np.hstack([
            #     i * np.ones_like(np.unique(env_labels[env])) for i, env in enumerate(sorted(env_labels))]))
            self.centroid_classes = torch.tensor(np.hstack([np.arange(env_labels[env].size) for env in sorted(env_labels)]))
            self.centroid_envs = torch.tensor(np.hstack([
                i * np.ones_like(env_labels[env]) for i, env in enumerate(sorted(env_labels))]))

            self.register_buffer('train_centroids', torch.zeros(self.centroid_classes.size(0), self.featurizer.n_outputs))

    def return_feats(self, x):
        return self.featurizer(x)

    def embedding(self, x):
        return self.return_feats(x)

    @staticmethod
    def pairwise_dist(x, y):
        return torch.cdist(x, y)

    @staticmethod
    def macro_alignment_loss(x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()
        return mean_diff + cova_diff

    def update_feature_stats(self, env_feats):
        if self.steps == 0 or self.steps % self.feat_update_freq != 0:
            return

        train_feats = [torch.stack(x, dim=0) for x in env_feats['feats'].values()]
        train_labels = [torch.stack(x, dim=0) for x in env_feats['labels'].values()]

        curr_centroids = torch.empty((0, self.train_centroids.size(-1))).to(train_feats[0].device)
        for env in range(len(train_feats)):
            curr_centroids = torch.cat((
                curr_centroids,
                torch.stack([train_feats[env][torch.where(train_labels[env] == c)[0]].mean(0)
                             for c in torch.unique(train_labels[env])])
            ))
        factor = 0 if self.steps == self.feat_update_freq else self.momentum
        self.train_centroids = \
            (1 - factor) * curr_centroids.to(self.train_centroids.device) + factor * self.train_centroids

    def update(self, minibatches, env_feats=None):
        self.update_feature_stats(env_feats)

        n_envs = len(minibatches)
        all_y = torch.cat([y for _, y in minibatches])
        all_envs = torch.cat([env * torch.ones_like(y) for env, (_, y) in enumerate(minibatches)])
        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifiers = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        # cross-entropy loss
        loss_x = 0
        for i in range(n_envs):
            loss_x += F.cross_entropy(classifiers[i], targets[i])
        loss_x /= n_envs

        # BoDA loss
        loss_b = torch.tensor([0.], device=all_y.device)
        if self.steps >= self.boda_start_step:
            pairwise_dist = -1 * self.pairwise_dist(self.train_centroids, torch.cat(features))
            # balanced distance
            n_per_sample = self.n_samples_table[all_envs.long(), all_y.long()]
            logits = torch.div(pairwise_dist, n_per_sample.to(pairwise_dist.device))
            # calibrated distance
            n_samples_numerator = self.n_samples_table[self.centroid_envs.long(), self.centroid_classes.long()]
            n_samples_denominator = self.n_samples_table[all_envs.long(), all_y.long()]
            size_h, size_w = n_samples_numerator.size(0), n_samples_denominator.size(0)
            cal_weights = (n_samples_numerator.unsqueeze(1).expand(-1, size_w) /
                           n_samples_denominator.unsqueeze(0).expand(size_h, -1)) ** self.nu
            logits *= cal_weights.to(logits.device)
            logits = torch.div(logits, self.temperature)
            mask_same_d_c = torch.eq(
                self.centroid_classes.contiguous().view(-1, 1).to(all_y.device), all_y.contiguous().view(-1, 1).T).float() * torch.eq(
                self.centroid_envs.contiguous().view(-1, 1).to(all_envs.device), all_envs.contiguous().view(-1, 1).T).float()
            log_prob = logits - torch.log((torch.exp(logits) * (1 - mask_same_d_c)).sum(0, keepdim=True))
            # compute mean of log-likelihood over positive
            mask_cls = torch.eq(self.centroid_classes.contiguous().view(-1, 1).to(all_y.device),
                                all_y.contiguous().view(-1, 1).T).float()
            mask_env = torch.eq(self.centroid_envs.contiguous().view(-1, 1).to(all_envs.device),
                                all_envs.contiguous().view(-1, 1).T).float()
            mask = mask_cls * (1 - mask_env)
            log_prob_pos = log_prob * mask
            loss_b = - log_prob_pos.sum() / mask.sum()

        # macro alignment loss
        # during warm-up stage, helps BoDA loss converge
        # in MDLT, brings marginal improvement to BoDA; in DG, helps improve performance
        # to remove, simply set "macro_weight=0" in hparams_registry
        penalty = 0
        for i in range(n_envs):
            for j in range(i + 1, n_envs):
                penalty += self.macro_alignment_loss(features[i], features[j])
        if n_envs > 1:
            penalty /= (n_envs * (n_envs - 1) / 2)

        self.optimizer.zero_grad()
        loss = loss_x + self.hparams['macro_weight'] * penalty
        if self.steps >= self.boda_start_step:
            loss += self.hparams['boda_weight'] * loss_b
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), f"Loss explosion: {loss.item()}"

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': loss_x.item(), 'loss_boda': loss_b.item(), 'penalty': penalty}


class GINIDG(ERM):
    """
    Generative Inference network for imbalanced domain generalization
    code site: https://github.com/HaifengXia/IDG
    """
    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lambd, reverse=True):
            ctx.lambd = lambd
            ctx.reverse = reverse
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.reverse:
                return (grad_output * -ctx.lambd), None, None
            else:
                return (grad_output * ctx.lambd), None, None

    class Generator(nn.Module):
        def __init__(self, num_domains, hparams):
            super(GINIDG.Generator, self).__init__()
            self.num_domain = num_domains

            self.sample_size = hparams['sample_size']
            self.zdim = hparams['zdim']
            self.dedim = hparams['dedim']
            self.gdim = hparams['gdim']

            self.mean = nn.Linear(self.gdim, self.zdim)
            self.log_sigma = nn.Linear(self.gdim, self.zdim)
            self.decoder = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.zdim, self.gdim),
            )

        def spilt(self, x):
            batch_size = x.size(0) / self.num_domain
            for i in range(self.num_domain):
                temp = x[int(i * batch_size):int((i + 1) * batch_size), :]
                if i == 0:
                    x_dom = temp
                else:
                    x_dom = x_dom + temp
            return x_dom / 3.0

        def statistic(self, x):
            mu = self.mean(x)
            log_sig = self.log_sigma(x)
            return mu, log_sig

        def get_encoder(self, mu, log_sig):
            eps = torch.randn(mu.size(0), self.zdim).to(mu.device)
            z = mu + eps * torch.sqrt(1e-8 + torch.exp(log_sig))
            return z

        def forward(self, x):
            # spilt the features into multi-source domains
            x = self.spilt(x)
            # learn the statistics of features
            mu, log_sig = self.statistic(x)
            mu = mu.repeat(self.sample_size, 1)
            log_sig = log_sig.repeat(self.sample_size, 1)
            z = self.get_encoder(mu, log_sig)
            G_x = self.decoder(z)
            return G_x

    class Discriminator(nn.Module):
        def __init__(self, repr_dim, hidden_dim1, hidden_dim2, out_dim, grl=True, reverse=True):
            super(GINIDG.Discriminator, self).__init__()
            self.grl = grl
            self.reverse = reverse
            self.model = nn.Sequential(
                nn.Linear(repr_dim, hidden_dim1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim2, out_dim),
            )
            self.lambd = 0.0

        def set_lambd(self, lambd):
            self.lambd = lambd

        def forward(self, x):
            if self.grl:
                x = GINIDG.GradReverse.apply(x, self.lambd, self.reverse)
            x = self.model(x)
            return x

    class GDiscriminator(nn.Module):
        def __init__(self, repr_dim, hidden_dim, out_dim, grl=True, reverse=True):
            super(GINIDG.GDiscriminator, self).__init__()
            self.grl = grl
            self.reverse = reverse
            self.model = nn.Sequential(
                nn.Linear(repr_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, out_dim),
            )
            self.lambd = 0.0

        def set_lambd(self, lambd):
            self.lambd = lambd

        def forward(self, x):
            if self.grl:
                x = GINIDG.GradReverse.apply(x, self.lambd, self.reverse)
            x = self.model(x)
            return x

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GINIDG, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.counter = 0
        self.max_iter = hparams['n_steps']
        # self.recon_end_indicator = hparams['recon_end_iters']
        self.recon_end_indicator = 1000

        self.grl_weight = 1.0
        grl, opts = True, {'zdim': 64, 'dedim': 1024, 'gdim': self.featurizer.n_outputs, 'sample_size': 6}
        self.generator = self.Generator(num_domains, opts)
        self.gdiscriminator = self.GDiscriminator(self.featurizer.n_outputs, 256, 2, grl=grl, reverse=True)
        self.discriminator = self.Discriminator(self.featurizer.n_outputs, 1024, 1024, num_domains, grl=grl, reverse=True)

        self.optimizer.add_param_group({'params': self.generator.parameters()})
        self.optimizer.add_param_group({'params': self.gdiscriminator.parameters()})
        self.optimizer.add_param_group({'params': self.discriminator.parameters()})

    def update(self, minibatches, unlabeled=None):
        # params setting
        alpha = (2. / (1. + np.exp(-10 * self.counter / self.max_iter)) - 1) * self.grl_weight
        self.discriminator.set_lambd(alpha)
        self.gdiscriminator.set_lambd(alpha)

        # training
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_disc = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=all_x.device)
            for i, (x, y) in enumerate(minibatches)
        ])

        all_z, num = self.featurizer(all_x), all_x.size(0)

        g_z = self.generator(all_z)
        concat_z = torch.cat([all_z, g_z], dim=0)

        concat_pred = self.classifier(concat_z)
        concat_dis = self.gdiscriminator(concat_z)
        concat_domain = self.discriminator(concat_z)

        pred = concat_pred[:num, :]
        gen_pred = concat_pred[num:, :]
        disc = concat_domain[:num, :]
        # gen_disc = concat_domain[num:, :]

        repetition = gen_pred.size(0) // pred.size(0)

        loss_class = nn.CrossEntropyLoss()(pred, all_y)
        loss_gen_class = nn.CrossEntropyLoss()(gen_pred, all_y.repeat(repetition, 1).reshape(-1))

        loss_domain = nn.CrossEntropyLoss()(disc, all_disc)
        num_samples = [num, gen_pred.size(0)]
        loss_dis = nn.CrossEntropyLoss()(concat_dis, torch.cat(
            [torch.full((num_samples[i], ), i, dtype=torch.int64, device=all_x.device) for i in range(2)]))

        loss_recon = torch.norm((all_z.repeat(repetition, 1) - g_z).abs(), 2, 1).sum() / float(gen_pred.size(0))

        total_loss = loss_class + loss_domain

        self.optimizer.zero_grad()
        loss_dis.backward(retain_graph=True)
        grad_for_Gdis_loss = []
        for param in list(self.generator.parameters()) + list(self.gdiscriminator.parameters()):
            grad_for_Gdis_loss.append(param.grad.data.clone())

        self.optimizer.zero_grad()
        loss_gen_class.backward(retain_graph=True)
        grad_for_Gclass_loss = []
        for param in list(self.network.parameters()) + list(self.generator.parameters()):
            grad_for_Gclass_loss.append(param.grad.data.clone())

        self.optimizer.zero_grad()
        loss_recon.backward(retain_graph=True)
        grad_for_recon_loss = []
        for param in self.generator.parameters():
            grad_for_recon_loss.append(param.grad.data.clone())

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_for_total_loss = []
        for param in list(self.network.parameters()) + list(self.discriminator.parameters()):
            grad_for_total_loss.append(param.grad.data.clone())

        if self.counter < self.recon_end_indicator:
            # encoder, classifier, discriminator
            for counter, param in enumerate(list(self.network.parameters()) + list(self.discriminator.parameters())):
                temp_grad = param.grad.data.clone()
                temp_grad.zero_()
                param.grad.data = temp_grad + grad_for_total_loss[counter]

            # generator
            offset = len(list(self.network.parameters()))
            for counter, param in enumerate(list(self.generator.parameters())):
                temp_grad = param.grad.data.clone()
                temp_grad.zero_()
                param.grad.data = temp_grad + 1.0 * grad_for_recon_loss[counter] + 1.0 * grad_for_Gclass_loss[offset + counter]

            # gdiscriminator
            for counter, param in enumerate(list(self.gdiscriminator.parameters())):
                temp_grad = param.grad.data.clone()
                temp_grad.zero_()
                param.grad.data = temp_grad
        else:
            # encoder, classifier
            for counter, param in enumerate(self.network.parameters()):
                temp_grad = param.grad.data.clone()
                temp_grad.zero_()
                param.grad.data = temp_grad + grad_for_total_loss[counter] + 1.0 * grad_for_Gclass_loss[counter]

            # generator
            offset = len(list(self.network.parameters()))
            for counter, param in enumerate(self.generator.parameters()):
                temp_grad = param.grad.data.clone()
                temp_grad.zero_()
                param.grad.data = temp_grad + 1.0 * grad_for_Gdis_loss[counter] + 1.0 * grad_for_Gclass_loss[offset + counter]

            # gdiscriminator
            offset = len(list(self.generator.parameters()))
            for counter, param in enumerate(self.gdiscriminator.parameters()):
                temp_grad = param.grad.data.clone()
                temp_grad.zero_()
                param.grad.data = temp_grad + 1.0 * grad_for_Gdis_loss[offset + counter]

            # discriminator
            offset = len(list(self.network.parameters()))
            for counter, param in enumerate(self.discriminator.parameters()):
                temp_grad = param.grad.data.clone()
                temp_grad.zero_()
                param.grad.data = temp_grad + grad_for_total_loss[offset + counter]

        self.optimizer.step()
        self.counter += 1

        return {'loss': total_loss.item()}
