# -*- coding: utf-8 -*-
import numpy as np
import torch

torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import pdb
from typing import Optional, List
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i, j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))







class AutoDebias(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(AutoDebias, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.lam = Variable(torch.Tensor([1]), requires_grad=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.u_i_pro = np.load('u_i_matrix.npy')

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1)
        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, x_val, y_val, args,
            num_epoch=200, batch_size=128, lr=0.001, lamb=0,
            tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam([self.W.weight, self.H.weight], lr=lr, weight_decay=lamb)
        optimizer_1 = torch.optim.Adam([self.lam], lr=lr, weight_decay=lamb)
        pos_set = []
        for u_i_pair, item in zip(x, y):
            if item == 1:
                pos_set.append(list(u_i_pair))
        neg_set_matrix = np.zeros((15400, 1000))
        for u_i_pos in pos_set:
            neg_set_matrix[u_i_pos[0] - 1][u_i_pos[1] - 1] = 1
        neg_set = np.argwhere(neg_set_matrix == 0)
        pos_set = np.array(pos_set)
        num_sample_pos = len(pos_set)

        y_val = np.reshape(y_val, (-1, 1))
        x_y_concat_val = np.hstack((x_val, y_val))
        val_index = np.arange(len(x_y_concat_val))

        total_batch = int(num_sample_pos // (batch_size / 2))
        architect = Architect(args)
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample_pos)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                np.random.shuffle(val_index)
                val_x = x_y_concat_val[val_index[:256], :-1]
                val_y = x_y_concat_val[val_index[:256], -1:]
                selected_idx = all_idx[int(batch_size / 2) * idx:(idx + 1) * int(batch_size / 2)]
                sub_x_pos = pos_set[selected_idx]
                sample_index = np.random.randint(low=0, high=len(neg_set), size=int(batch_size / 2))
                sub_x_neg = neg_set[sample_index, :]
                sub_x = np.vstack((sub_x_pos, sub_x_neg))
                sub_y_pos = np.ones(int(batch_size / 2))
                sub_y_neg = np.zeros(int(batch_size / 2))
                sub_y = np.hstack((sub_y_pos, sub_y_neg))

                index_u = sub_x[:, 0] - 1
                index_i = sub_x[:, 1] - 1
                inv_prop = torch.pow(torch.tensor(1 / self.u_i_pro[index_u, index_i], dtype=torch.float32), self.lam)
                sub_y = torch.Tensor(sub_y)
                val_y = torch.Tensor(val_y.reshape((len(val_y),)))
                optimizer_1.zero_grad()
                self.lam = architect.step(sub_x, sub_y, val_x, val_y, optimizer, self.lam, self.W, self.H,
                                          unrolled=args.unrolled, eta=0.001)
                nn.utils.clip_grad_norm(self.lam, 1)
                optimizer_1.step()
                optimizer.zero_grad()
                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred.mul(inv_prop))
                xent_loss = F.binary_cross_entropy(pred, sub_y)
                loss = xent_loss
                loss.backward()
                nn.utils.clip_grad_norm([self.W.weight, self.H.weight], 5)
                optimizer.step()
                epoch_loss += xent_loss.detach().numpy()
            print('*****************************')
            print('epoch:', epoch)
            print('epoch loss:', epoch_loss)
            print('new lambda:', self.lam.data.detach())
            if epoch % 10 == 0 and verbose:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
            if epoch == num_epoch - 1:
                print("[MF-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy()

class DR_MCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(DR_MCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, y_ips,
            num_epoch=1000, batch_size=128, lr=0.05, lamb=0,
            tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        prior_y = y_ips.mean()
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]

                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                x_sampled = x_all[ul_idxs[idx * batch_size:(idx + 1) * batch_size]]

                pred_ul, _, _ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum")

                imputation_y = torch.Tensor([prior_y] * selected_idx.shape[0])
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")

                ips_loss = (xent_loss - imputation_loss) / selected_idx.shape[0]

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_y, reduction="mean")

                loss = ips_loss + direct_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy()

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


class IPS_MCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(IPS_MCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, y_ips=None,
            num_epoch=1000, batch_size=128, lr=0.05, lamb=0,
            tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]
                sum_inv_prop = torch.sum(inv_prop)

                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(pred, sub_y,
                                                   weight=inv_prop, reduction="sum")

                xent_loss = xent_loss / sum_inv_prop

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-SNIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy()

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


class NCF(nn.Module):


    def __init__(self, num_users, num_items, embedding_k=4):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k * 2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4, batch_size=128, verbose=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                optimizer.zero_grad()
                pred, u_emb, v_emb = self.forward(sub_x, True)

                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(pred, torch.unsqueeze(torch.Tensor(sub_y), 1))

                loss = xent_loss
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[NCF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[NCF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

    def partial_fit(self, x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4):
        self.fit(x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4)

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1 - pred, pred], axis=1)

class IPS_NCF(nn.Module):

    def __init__(self, num_users, num_items, embedding_k=4):
        super(IPS_NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k * 2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, y_ips=None,
            num_epoch=1000, batch_size=128,
            lr=0.05, lamb=0, tol=1e-4, verbose=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = torch.Tensor(y[selected_idx])

                # propensity score
                inv_prop = one_over_zl[selected_idx]
                pred, u_emb, v_emb = self.forward(sub_x, True)

                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(torch.squeeze(pred), sub_y,
                                                   weight=inv_prop)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[NCF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[NCF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[NCF-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1 - pred, pred], axis=1)

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


class SNIPS_NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4):
        super(SNIPS_NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k * 2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, y_ips=None,
            num_epoch=1000, batch_size=128,
            lr=0.05, lamb=0, tol=1e-4, verbose=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = torch.Tensor(y[selected_idx])

                # propensity score
                inv_prop = one_over_zl[selected_idx]
                sum_inv_prop = torch.sum(inv_prop)

                pred, u_emb, v_emb = self.forward(sub_x, True)

                pred = self.sigmoid(pred)

                loss = F.binary_cross_entropy(torch.squeeze(pred), sub_y,
                                              weight=inv_prop, reduction="sum")
                loss = loss / sum_inv_prop

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[NCF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[NCF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[NCF-SNIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1 - pred, pred], axis=1)

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None

class LDR_MCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=32):
        super(LDR_MCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        # self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        # self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        # self.linear_1 = torch.nn.Linear(self.embedding_k * 2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        # self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.domain_linear_1 = torch.nn.Linear(self.embedding_k, self.embedding_k)
        self.domain_linear = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.class_linear_1 = torch.nn.Linear(self.embedding_k, self.embedding_k)
        self.class_linear = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.xent_func = torch.nn.BCELoss()
        self.user_latent = nn.Embedding(num_users, embedding_k)
        self.item_latent = nn.Embedding(num_items, embedding_k)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.dropout_p = 0
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.init_embedding(0)

    def init_embedding(self, init):
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a=init)

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        u_latent = self.dropout(self.user_latent.weight[user_idx])
        i_latent = self.dropout(self.item_latent.weight[item_idx])
        u_bias = self.user_bias.weight[user_idx]
        i_bias = self.item_bias.weight[item_idx]
        h1 = u_latent * i_latent
        h1 = self.relu(h1)
        out = torch.sum(h1, dim=1, keepdim=True) + u_bias + i_bias
        domain_h = GradReverse.apply(h1, 0.1)
        class_h = h1.detach()

        domain_h = self.domain_linear_1(domain_h)
        domain_h = self.relu(domain_h)
        domain_out = self.domain_linear(domain_h)

        class_h = self.class_linear_1(class_h)
        class_h = self.relu(class_h)
        class_out = self.class_linear(class_h)

        if is_training:
            return out, domain_out,class_out,u_latent, i_latent
        else:
            return out

    def fit(self, x, y, num_epoch=1000, batch_size=32,
            lr=0.05, lamb=1e-4,
            alpha=0.1, gamma=0.01, tol=1e-4, verbose=0):

        self.alpha = alpha
        self.gamma = gamma
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        x_counterfactual = generate_total_sample(self.num_users, self.num_items)
        factual_sample = len(x)
        total_batch = factual_sample // batch_size
        early_stop = 0

        for epoch in range(num_epoch):
            # sampling factuals
            factual_idx = np.arange(factual_sample)
            np.random.shuffle(factual_idx)
            # sampling counterfactuals
            counterfactual_idxs = np.arange(x_counterfactual.shape[0])
            # np.random.shuffle(counterfactual_idxs)

            epoch_loss = 0
            domain_loss = 0
            for idx in range(total_batch):
                # mini-batch factuals
                selected_idx = factual_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y)

                pred, domain_out1, class_out1,u_emb, v_emb = self.forward(sub_x, True)

                # mini-batch counterfactuals
                selected_id = np.random.choice(counterfactual_idxs, batch_size)
                x_sampled = x_counterfactual[selected_id]
                pred_ul, domain_out2, class_out2,_, _ = self.forward(x_sampled, True)

                pred = self.sigmoid(pred)

                domain_out1 = self.sigmoid(domain_out1)
                domain_out2 = self.sigmoid(domain_out2)
                class_out1 = self.sigmoid(class_out1)
                class_out2 = self.sigmoid(class_out2)
                #self.xent_func2 = torch.nn.BCELoss(torch.squeeze(((1 - domain_out1) / (domain_out1))).detach())
                #weight = torch.squeeze(((1 - domain_out1) / (domain_out1))).detach()
                weight = torch.squeeze(((1 - class_out2) / (class_out2))).detach()
                self.xent_func2 = torch.nn.BCELoss(weight)

                factual_loss = self.xent_func2(torch.squeeze(pred), sub_y)

                #                 xent_loss=F.binary_cross_entropy(pred, sub_y, weight=torch.squeeze((domain_out1/(1-domain_out1))).detach(),reduction="sum")
                #                 xent_loss=F.binary_cross_entropy(pred, sub_y, weight=torch.squeeze((domain_out1/(1-domain_out1))).detach(),reduction="sum")
                # domain loss
                # domain_loss1 = self.xent_func(torch.squeeze(domain_out1), torch.zeros([batch_size]))
                # domain_loss2 = self.xent_func(torch.squeeze(domain_out2), torch.ones([batch_size]))
                domain_loss1 = torch.mean((domain_out1))
                domain_loss2 = torch.mean((domain_out2 * domain_out2 / 4) + 1)

                class_loss1 = self.xent_func(torch.squeeze(class_out1), torch.zeros([batch_size]))
                class_loss2 = self.xent_func(torch.squeeze(class_out2), torch.ones([batch_size]))
                # pred_ul = self.sigmoid(pred_ul)
                #
                # pred_avg = pred.mean()
                # pred_ul_avg = pred_ul.mean()
                #
                # info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1 - pred_avg) * (
                #             1 - pred_ul_avg).log()) + self.gamma * torch.mean(pred * pred.log())
                loss = factual_loss + self.gamma*(domain_loss1 + domain_loss2)+ 0.6 * (class_loss1 + class_loss2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += factual_loss.detach().numpy()
            #                 domain_loss+=(domain_loss1+domain_loss2).detach().numpy()
            # print(xent_loss,self.alpha*(domain_loss1 + domain_loss2),self.gamma * (class_loss1 + class_loss2)
            #     )
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)

            if relative_loss_div < tol:
                if early_stop > 8:
                    print("[LDR-MCF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
            last_loss = epoch_loss
            if epoch % 10 == 0 and verbose:
                print("[LDR-MCF] epoch:{}, xent:{}".format(epoch, epoch_loss))
            if epoch == num_epoch - 1:
                print("[LDR-MCF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1 - pred, pred], axis=1)

# class LDR_MFwoC(nn.Module):
#     """The neural collaborative filtering method.
#     """
#
#     def __init__(self, num_users, num_items, embedding_k=32):
#         super(LDR_MFwoC, self).__init__()
#         self.num_users = num_users
#         self.num_items = num_items
#         self.embedding_k = embedding_k
#         # self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
#         # self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
#         # self.linear_1 = torch.nn.Linear(self.embedding_k * 2, self.embedding_k)
#         self.relu = torch.nn.ReLU()
#         # self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
#         self.sigmoid = torch.nn.Sigmoid()
#
#         self.domain_linear_1 = torch.nn.Linear(self.embedding_k, self.embedding_k)
#         self.domain_linear = torch.nn.Linear(self.embedding_k, 1, bias=False)
#
#         self.class_linear_1 = torch.nn.Linear(self.embedding_k, self.embedding_k)
#         self.class_linear = torch.nn.Linear(self.embedding_k, 1, bias=False)
#
#         self.xent_func = torch.nn.BCELoss()
#         self.user_latent = nn.Embedding(num_users, embedding_k)
#         self.item_latent = nn.Embedding(num_items, embedding_k)
#         self.user_bias = nn.Embedding(num_users, 1)
#         self.item_bias = nn.Embedding(num_items, 1)
#         self.dropout_p = 0
#         self.dropout = nn.Dropout(p=self.dropout_p)
#         self.init_embedding(0)
#
#     def init_embedding(self, init):
#
#         nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a=init)
#         nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a=init)
#         nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a=init)
#         nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a=init)
#         nn.init.kaiming_normal_(self.domain_linear_1.weight, mode='fan_out', a=init)
#         nn.init.kaiming_normal_(self.domain_linear.weight, mode='fan_out', a=init)
#         nn.init.kaiming_normal_(self.class_linear_1.weight, mode='fan_out', a=init)
#         nn.init.kaiming_normal_(self.class_linear.weight, mode='fan_out', a=init)
#
#     def forward(self, x, is_training=False):
#         user_idx = torch.LongTensor(x[:, 0])
#         item_idx = torch.LongTensor(x[:, 1])
#
#         u_latent = self.dropout(self.user_latent.weight[user_idx])
#         i_latent = self.dropout(self.item_latent.weight[item_idx])
#         u_bias = self.user_bias.weight[user_idx]
#         i_bias = self.item_bias.weight[item_idx]
#
#         h1 = u_latent * i_latent
#         h1 = self.relu(h1)
#
#         out = torch.sum(h1, dim=1, keepdim=True) + u_bias + i_bias
#
#         domain_h = GradReverse.apply(h1, 0.5)
#
#         class_h = h1.detach()
#
#         domain_h = self.domain_linear_1(domain_h)
#         domain_h = self.relu(domain_h)
#         domain_out = self.domain_linear(domain_h)
#
#         class_h = self.class_linear_1(class_h)
#         class_h = self.relu(class_h)
#         class_out = self.class_linear(class_h)
#
#         # out = torch.sum(U_emb.mul(V_emb), 1)
#
#         if is_training:
#             return out, domain_out, class_out, u_latent, i_latent
#         else:
#             return out
#
#     def fit(self, x, y, num_epoch=1000, batch_size=32,
#             lr=0.05, lamb=1e-4,
#             alpha=0.1, gamma=0.01, tol=1e-4, verbose=0):
#
#         self.alpha = alpha
#         self.gamma = gamma
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
#         last_loss = 1e9
#
#         x_counterfactual = generate_total_sample(self.num_users, self.num_items)
#
#         factual_sample = len(x)
#         total_batch = factual_sample // batch_size
#         early_stop = 0
#
#         for epoch in range(num_epoch):
#             # sampling factuals
#             factual_idx = np.arange(factual_sample)
#             np.random.shuffle(factual_idx)
#             # sampling counterfactuals
#             counterfactual_idxs = np.arange(x_counterfactual.shape[0])
#             epoch_loss = 0
#             for idx in range(total_batch):
#                 # mini-batch factuals
#                 selected_idx = factual_idx[batch_size * idx:(idx + 1) * batch_size]
#                 sub_x = x[selected_idx]
#                 sub_y = y[selected_idx]
#                 sub_y = torch.Tensor(sub_y)
#                 pred, domain_out1, class_out1, u_emb, v_emb = self.forward(sub_x, True)
#                 # mini-batch counterfactuals
#                 selected_id = np.random.choice(counterfactual_idxs, batch_size)
#                 x_sampled = x_counterfactual[selected_id]
#                 pred_countfactual, domain_out2, class_out2, _, _ = self.forward(x_sampled, True)
#
#                 pred = self.sigmoid(pred)
#                 domain_out1 = self.sigmoid(domain_out1)
#                 domain_out2 = self.sigmoid(domain_out2)
#
#                 # self.xent_func2 = torch.nn.BCELoss(torch.squeeze(((1 - domain_out1) / (domain_out1))).detach())
#                 weight = torch.squeeze(((1 - domain_out1) / (domain_out1))).detach()
#                 self.xent_func2 = torch.nn.BCELoss(weight)
#                 # xent_func2 = torch.nn.MSELoss(reduction='none')
#                 # xent_loss = xent_func2(torch.squeeze(pred), sub_y) * weight
#                 # xent_loss = xent_loss.sum()
#
#                 xent_loss = self.xent_func2(torch.squeeze(pred), sub_y)
#                 #                 xent_loss=F.binary_cross_entropy(pred, sub_y, weight=torch.squeeze((domain_out1/(1-domain_out1))).detach(),reduction="sum")
#                 #                 xent_loss=F.binary_cross_entropy(pred, sub_y, weight=torch.squeeze((domain_out1/(1-domain_out1))).detach(),reduction="sum")
#                 # domain loss
#                 domain_loss1 = self.xent_func(torch.squeeze(domain_out1), torch.zeros([batch_size]))
#                 domain_loss2 = self.xent_func(torch.squeeze(domain_out2), torch.ones([batch_size]))
#
#                 # pesude label of counterfactuals
#                 # pred_ul = self.sigmoid(pred_ul)
#                 #
#                 # pred_avg = pred.mean()
#                 # pred_ul_avg = pred_ul.mean()
#                 #
#                 # info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1 - pred_avg) * (
#                 #             1 - pred_ul_avg).log()) + self.gamma * torch.mean(pred * pred.log())
#                 loss = xent_loss + self.alpha * (domain_loss1 + domain_loss2)
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#                 epoch_loss += xent_loss.detach().numpy()
#             #                 domain_loss+=(domain_loss1+domain_loss2).detach().numpy()
#             #print(xent_loss,self.alpha*(domain_loss1 + domain_loss2)  )
#             relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
#
#             if relative_loss_div < tol:
#                 if early_stop > 10:
#                     print("[NCF-CVIB2] epoch:{}, xent:{}".format(epoch, epoch_loss))
#                     break
#                 early_stop += 1
#
#             last_loss = epoch_loss
#
#             if epoch % 10 == 0 and verbose:
#                 print("[NCF-CVIB2] epoch:{}, xent:{}".format(epoch, epoch_loss))
#             #                 print("[NCF-CVIB2] epoch:{}, domain:{}".format(epoch, domain_loss))
#
#             if epoch == num_epoch - 1:
#                 print("[NCF-CVIB2] Reach preset epochs, it seems does not converge.")
#
#     def predict(self, x):
#         pred = self.forward(x)
#         pred = self.sigmoid(pred)
#         return pred.detach().numpy().flatten()
#
#     def predict_proba(self, x):
#         pred = self.forward(x)
#         pred = pred.reshape(-1, 1)
#         pred = self.sigmoid(pred)
#         return np.concatenate([1 - pred, pred], axis=1)


class LDR_NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=32):
        super(LDR_NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k * 2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.domain_linear_1 = torch.nn.Linear(self.embedding_k, self.embedding_k)
        self.domain_linear = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.class_linear_1 = torch.nn.Linear(self.embedding_k, self.embedding_k)
        self.class_linear = torch.nn.Linear(self.embedding_k, 1, bias=False)

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0])
        item_idx = torch.LongTensor(x[:, 1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)
        out = self.linear_2(h1)
        domain_h = GradReverse.apply(h1, 0.01)
        class_h = h1.detach()#GradReverse.apply(h1, 0.5)

        domain_h = self.domain_linear_1(domain_h)
        domain_h = self.relu(domain_h)
        domain_out = self.domain_linear(domain_h)

        class_h = self.class_linear_1(class_h)
        class_h = self.relu(class_h)
        class_out = self.class_linear(class_h)

        if is_training:
            return out, domain_out,class_out,U_emb, V_emb
        else:
            return out

    def fit(self, x, y, num_epoch=1000, batch_size=32,
            lr=0.05, lamb=1e-4,
            alpha=0.1, gamma=0.01, tol=1e-4, verbose=0):

        self.alpha = alpha
        self.gamma = gamma

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        x_counterfactual = generate_total_sample(self.num_users, self.num_items)
        lossList1 = []
        lossList2 = []
        lossList3 = []
        num_sample = len(x)

        total_batch = num_sample // batch_size
        early_stop = 0

        for epoch in range(num_epoch):

            # sampling factuals
            factual_idx = np.arange(num_sample)
            np.random.shuffle(factual_idx)
            # sampling counterfactuals
            counterfactual_idxs = np.arange(x_counterfactual.shape[0])
            epoch_loss = 0
            domain_loss = 0
            for idx in range(total_batch):
                # mini-batch factuals
                selected_idx = factual_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y)
                pred, domain_out1, class_out1,u_emb, v_emb = self.forward(sub_x, True)

                # mini-batch counterfactuals
                selected_id = np.random.choice(counterfactual_idxs, batch_size)
                x_sampled = x_counterfactual[selected_id]
                pred_ul, domain_out2, class_out2,_, _ = self.forward(x_sampled, True)

                pred = self.sigmoid(pred)

                # domain_out1 = self.sigmoid(domain_out1)
                # domain_out2 = self.sigmoid(domain_out2)
                class_out1 = self.sigmoid(class_out1)
                class_out2 = self.sigmoid(class_out2)

                #self.xent_func2 = torch.nn.BCELoss(torch.squeeze(((1 - domain_out1) / (domain_out1))).detach())
                #weight = torch.squeeze(((1 - domain_out1) / (domain_out1))).detach()
                weight = torch.squeeze(((1 - class_out2) / (class_out2))).detach()
                self.xent_func2 = torch.nn.BCELoss(weight)
                # xent_func2 = torch.nn.MSELoss(reduction='none')
                # xent_loss = xent_func2(torch.squeeze(pred), sub_y) * weight
                # xent_loss = xent_loss.sum()

                factual_loss = self.xent_func2(torch.squeeze(pred), sub_y)
                #                 xent_loss=F.binary_cross_entropy(pred, sub_y, weight=torch.squeeze((domain_out1/(1-domain_out1))).detach(),reduction="sum")
                #                 xent_loss=F.binary_cross_entropy(pred, sub_y, weight=torch.squeeze((domain_out1/(1-domain_out1))).detach(),reduction="sum")
                # domain loss
                # domain_loss1 = self.xent_func(torch.squeeze(domain_out1), torch.zeros([batch_size]))
                # domain_loss2 = self.xent_func(torch.squeeze(domain_out2), torch.ones([batch_size]))
                domain_loss1 = torch.mean((domain_out1))
                domain_loss2 = torch.mean((domain_out2*domain_out2/4)+1)

                class_loss1 = self.xent_func(torch.squeeze(class_out1), torch.zeros([batch_size]))
                class_loss2 = self.xent_func(torch.squeeze(class_out2), torch.ones([batch_size]))

                loss = factual_loss + self.gamma*(domain_loss1 + domain_loss2)+ self.alpha * (class_loss1 + class_loss2)
                lossList1.append(factual_loss)
                lossList2.append((domain_loss1 + domain_loss2))
                lossList3.append((class_loss1 + class_loss2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += factual_loss.detach().numpy()
            #                 domain_loss+=(domain_loss1+domain_loss2).detach().numpy()
            # print(xent_loss,self.alpha*(domain_loss1 + domain_loss2),self.gamma * (class_loss1 + class_loss2)
            #     )
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)

            if relative_loss_div < tol:
                if early_stop > 10:
                    print("[NCF-CVIB2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[NCF-CVIB2] epoch:{}, xent:{}".format(epoch, epoch_loss))
            #                 print("[NCF-CVIB2] epoch:{}, domain:{}".format(epoch, domain_loss))

            if epoch == num_epoch - 1:
                print("[NCF-CVIB2] Reach preset epochs, it seems does not converge.")
        return lossList1,lossList2,lossList3
    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1 - pred, pred], axis=1)


# class LDR_NCFwoC(nn.Module):
#     """The neural collaborative filtering method.
#     """
#     def __init__(self, num_users, num_items, embedding_k=32):
#         super(LDR_NCFwoC, self).__init__()
#         self.num_users = num_users
#         self.num_items = num_items
#         self.embedding_k = embedding_k
#         self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
#         self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
#         self.linear_1 = torch.nn.Linear(self.embedding_k * 2, self.embedding_k)
#         self.relu = torch.nn.ReLU()
#         self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
#         self.sigmoid = torch.nn.Sigmoid()
#
#         self.domain_linear_1 = torch.nn.Linear(self.embedding_k, self.embedding_k)
#         self.domain_linear = torch.nn.Linear(self.embedding_k, 1, bias=False)
#
#         self.class_linear_1 = torch.nn.Linear(self.embedding_k, self.embedding_k)
#         self.class_linear = torch.nn.Linear(self.embedding_k, 1, bias=False)
#
#
#         self.xent_func = torch.nn.BCELoss()
#
#     def forward(self, x, is_training=False):
#         user_idx = torch.LongTensor(x[:, 0])
#         item_idx = torch.LongTensor(x[:, 1])
#         U_emb = self.W(user_idx)
#         V_emb = self.H(item_idx)
#
#         # concat
#         z_emb = torch.cat([U_emb, V_emb], axis=1)
#
#         h1 = self.linear_1(z_emb)
#         h1 = self.relu(h1)
#
#         out = self.linear_2(h1)
#
#
#         domain_h = GradReverse.apply(h1, 0.5)
#
#
#         class_h = GradReverse.apply(h1, 0.5)
#
#
#         domain_h = self.domain_linear_1(domain_h)
#         domain_h = self.relu(domain_h)
#         domain_out = self.domain_linear(domain_h)
#
#
#         class_h = self.class_linear_1(class_h)
#         class_h = self.relu(class_h)
#         class_out = self.class_linear(class_h)
#
#         # out = torch.sum(U_emb.mul(V_emb), 1)
#
#         if is_training:
#             return out, domain_out,class_out,U_emb, V_emb
#         else:
#             return out
#
#     def fit(self, x, y, x_test, num_epoch=1000, batch_size=32,
#             lr=0.05, lamb=1e-4,
#             alpha=0.1, gamma=0.01, tol=1e-4, verbose=0):
#
#         self.alpha = alpha
#         self.gamma = gamma
#
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
#
#         last_loss = 1e9
#
#         # generate all counterfactuals and factuals for info reg
#         x_all = generate_total_sample(self.num_users, self.num_items)
#
#         num_sample = len(x)
#
#         total_batch = num_sample // batch_size
#         early_stop = 0
#
#         for epoch in range(num_epoch):
#
#             # sampling factuals
#             all_idx = np.arange(num_sample)
#             np.random.shuffle(all_idx)
#
#             # sampling counterfactuals
#             ul_idxs = np.arange(x_test.shape[0])
#             #             np.random.shuffle(x_test)
#             epoch_loss = 0
#             domain_loss = 0
#             for idx in range(total_batch):
#                 # mini-batch factuals
#                 selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
#                 sub_x = x[selected_idx]
#                 sub_y = y[selected_idx]
#                 sub_y = torch.Tensor(sub_y)
#
#                 pred, domain_out1, class_out1,u_emb, v_emb = self.forward(sub_x, True)
#
#                 # mini-batch counterfactuals
#                 selected_id = np.random.choice(ul_idxs, batch_size)
#                 x_sampled = x_test[selected_id]
#                 pred_ul, domain_out2, class_out2,_, _ = self.forward(x_sampled, True)
#
#                 pred = self.sigmoid(pred)
#
#                 domain_out1 = self.sigmoid(domain_out1)
#                 domain_out2 = self.sigmoid(domain_out2)
#
#                 #self.xent_func2 = torch.nn.BCELoss(torch.squeeze(((1 - domain_out1) / (domain_out1))).detach())
#                 weight = torch.squeeze(((1 - domain_out1) / (domain_out1))).detach()
#                 self.xent_func2 = torch.nn.BCELoss(weight)
#                 # xent_func2 = torch.nn.MSELoss(reduction='none')
#                 # xent_loss = xent_func2(torch.squeeze(pred), sub_y) * weight
#                 # xent_loss = xent_loss.sum()
#
#
#                 xent_loss = self.xent_func2(torch.squeeze(pred), sub_y)
#                 #                 xent_loss=F.binary_cross_entropy(pred, sub_y, weight=torch.squeeze((domain_out1/(1-domain_out1))).detach(),reduction="sum")
#                 #                 xent_loss=F.binary_cross_entropy(pred, sub_y, weight=torch.squeeze((domain_out1/(1-domain_out1))).detach(),reduction="sum")
#                 # domain loss
#                 domain_loss1 = self.xent_func(torch.squeeze(domain_out1), torch.ones([batch_size]))
#                 domain_loss2 = self.xent_func(torch.squeeze(domain_out2), torch.zeros([batch_size]))
#
#                 # pesude label of counterfactuals
#                 # pred_ul = self.sigmoid(pred_ul)
#                 #
#                 # pred_avg = pred.mean()
#                 # pred_ul_avg = pred_ul.mean()
#                 #
#                 # info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1 - pred_avg) * (
#                 #             1 - pred_ul_avg).log()) + self.gamma * torch.mean(pred * pred.log())
#                 loss = xent_loss + self.alpha*(domain_loss1 + domain_loss2)
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#                 epoch_loss += xent_loss.detach().numpy()
#             #                 domain_loss+=(domain_loss1+domain_loss2).detach().numpy()
#             # print(xent_loss,self.alpha*(domain_loss1 + domain_loss2),self.gamma * (class_loss1 + class_loss2)
#             #     )
#             relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
#
#             if relative_loss_div < tol:
#                 if early_stop > 10:
#                     print("[NCF-CVIB2] epoch:{}, xent:{}".format(epoch, epoch_loss))
#                     break
#                 early_stop += 1
#
#             last_loss = epoch_loss
#
#             if epoch % 10 == 0 and verbose:
#                 print("[NCF-CVIB2] epoch:{}, xent:{}".format(epoch, epoch_loss))
#             #                 print("[NCF-CVIB2] epoch:{}, domain:{}".format(epoch, domain_loss))
#
#             if epoch == num_epoch - 1:
#                 print("[NCF-CVIB2] Reach preset epochs, it seems does not converge.")
#
#     def predict(self, x):
#         pred = self.forward(x)
#         pred = self.sigmoid(pred)
#         return pred.detach().numpy().flatten()
#
#     def predict_proba(self, x):
#         pred = self.forward(x)
#         pred = pred.reshape(-1, 1)
#         pred = self.sigmoid(pred)
#         return np.concatenate([1 - pred, pred], axis=1)


class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct



def calculate_distance(source_feature: torch.Tensor, target_feature: torch.Tensor,
              device, progress=True, training_epochs=10):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.

    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier

    Returns:
        :math:`\mathcal{A}`-distance
    """
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    feature = torch.cat([source_feature, target_feature], dim=0)
    label = torch.cat([source_label, target_label], dim=0)

    dataset = TensorDataset(feature, label)
    length = len(dataset)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    anet = ANet(feature.shape[1]).to(device)
    optimizer = SGD(anet.parameters(), lr=0.01)
    a_distance = 2.0
    for epoch in range(1):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        anet.eval()
        meter = AverageMeter("accuracy", ":4.2f")
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                meter.update(acc, x.shape[0])
        error = 1 - meter.avg / 100
        a_distance = 2 * (1 - 2 * error)
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))

    return a_distance






def one_hot(x):
    out = torch.cat([torch.unsqueeze(1 - x, 1), torch.unsqueeze(x, 1)], axis=1)
    return out


def sharpen(x, T):
    temp = x ** (1 / T)
    return temp / temp.sum(1, keepdim=True)
