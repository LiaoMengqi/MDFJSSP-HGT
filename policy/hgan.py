import torch.nn as nn
import torch.nn.functional as F
from env.base_class import State, Action
import torch
from torch.distributions import Categorical
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / norm


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_hidden, num_heads, d_kv, target_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.target_dim = target_dim

        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_kv * num_heads)
        self.w_k = nn.Linear(d_model, d_kv * num_heads)
        self.w_v = nn.Linear(d_model, d_kv * num_heads)
        self.w_o = nn.Linear(num_heads * d_kv, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        self.w_q.weight.data = self.w_q.weight.data / math.sqrt(self.d_kv)
        self.w_k.weight.data = self.w_k.weight.data / math.sqrt(self.d_kv)
        self.w_v.weight.data = self.w_v.weight.data / math.sqrt(self.d_kv)
        nn.init.xavier_uniform_(self.w_o.weight)

        nn.init.constant_(self.w_q.bias, 0)
        nn.init.constant_(self.w_k.bias, 0)
        nn.init.constant_(self.w_v.bias, 0)
        nn.init.constant_(self.w_o.bias, 0)

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_kv) and transpose for multi-head attention.
        """
        # (batch_size, seq_len, num_heads, d_kv)
        x = x.view(*x.shape[:-1], self.num_heads, self.d_kv)
        # (batch_size, num_heads, seq_len, d_kv)
        if len(x.shape) == 4:
            return x.transpose(-2, -3).contiguous()
        else:
            return x.transpose(-2, -3).transpose(-3, -4).contiguous()

    def forward(self, h, h_arc, mask):
        q = self.w_q(h)
        k = self.w_k(h_arc)
        v = self.w_v(h_arc)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attention = (q.unsqueeze(self.target_dim) * k).sum(-1) / (self.d_model ** 0.5)
        attention = attention - attention.max(dim=-1, keepdim=True)[0]
        # attention = F.softmax(attention.masked_fill(mask.unsqueeze(1) == 0, 1e-9), dim=-1)
        attention = F.softmax(attention.masked_fill(mask.unsqueeze(1) == 0, -1e9), dim=self.target_dim)

        res = (attention.unsqueeze(-1) * v).sum(dim=self.target_dim).transpose(-2, -3).contiguous()

        size = res.size()[:-2] + (-1,)
        res = self.dropout(self.w_o(res.view(size)))
        return res


class FFN(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.init_weight()
        self.act = nn.GELU()

    def init_weight(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        x = self.dropout(self.act(self.linear1(x)))
        x = self.dropout(self.linear2(x))
        return x


class HGTMachineEncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, d_kv, dropout=0.1):
        super(HGTMachineEncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.d_kv = d_kv

        self.attention = MultiHeadAttention(d_model, d_hidden, n_head, d_kv, 3, dropout)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

        self.ffn = FFN(d_model, d_hidden, dropout)

    def forward(self, h_m, h_arc, mask):
        # pre norm
        h_m = h_m + self.attention(self.norm1(h_m), self.norm2(h_arc), mask)
        h_m = h_m + self.ffn(self.norm3(h_m))
        return h_m


class HGTOperationEncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, d_kv, dropout=0.1):
        super(HGTOperationEncoderLayer, self).__init__()
        self.n_head = n_head
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.d_kv = d_kv

        self.attention = MultiHeadAttention(d_model, d_hidden, n_head, d_kv, 2, dropout)

        self.w_next = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.w_next.weight)
        nn.init.constant_(self.w_next.bias, 0)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.norm4 = RMSNorm(d_model)

        self.fnn = FFN(d_model, d_hidden, dropout)

    def forward(self, h_o, h_arc, mask, adj_matrix):
        """

        :param adj_matrix:  size=(batch,nums_operation,nums_operation)
        :param h_o:  size=(batch,nums_machine,d_machine_input)
        :param h_arc:  size=(batch,nums_operation,nums_machine,d_arc_input)
        :param mask: size=(batch,nums_operation,nums_machine)
        :return:
        """
        h_o_next = adj_matrix.float().bmm(self.w_next(self.norm4(h_o)))

        h_o = self.attention(self.norm1(h_o), self.norm2(h_arc), mask) + h_o_next + h_o

        h_o = self.fnn(self.norm3(h_o)) + h_o

        return h_o


class HGAN(nn.Module):
    def __init__(self, d_machine_raw, d_operation_raw, d_arc_raw, d_model, d_hidden, num_layers, num_head, d_kv,
                 dropout=0.1, device='cpu', cat=True, proj=False):
        super(HGAN, self).__init__()
        self.num_layers = num_layers
        self.d_machine_raw = d_machine_raw
        self.d_operation_raw = d_operation_raw
        self.d_arc_raw = d_arc_raw
        self.d_hidden = d_hidden
        self.device = device
        self.cat = cat

        self.machine_encoder_layer = HGTMachineEncoderLayer
        self.operation_encoder_layer = HGTOperationEncoderLayer

        if self.cat:
            self.w_trans_o = nn.Linear(d_operation_raw * d_model, d_model)
            self.w_trans_m = nn.Linear(d_machine_raw * d_model, d_model)
            self.w_trans_a = nn.Linear(d_arc_raw * d_model, d_model)

            nn.init.xavier_uniform_(self.w_trans_m.weight)
            nn.init.xavier_uniform_(self.w_trans_o.weight)
            nn.init.xavier_uniform_(self.w_trans_a.weight)
            nn.init.constant_(self.w_trans_o.bias, 0)
            nn.init.constant_(self.w_trans_m.bias, 0)
            nn.init.constant_(self.w_trans_a.bias, 0)

        self.proj = proj
        if self.proj:
            self.m_proj = nn.Linear(d_model, d_model)
            nn.init.xavier_uniform_(self.m_proj.weight)
            nn.init.constant_(self.m_proj.bias, 0)
            self.o_proj = nn.Linear(d_model, d_model)
            nn.init.xavier_uniform_(self.o_proj.weight)
            nn.init.constant_(self.o_proj.bias, 0)

        self.machine_encoder_layers = nn.ModuleList()
        self.operation_encoder_layers = nn.ModuleList()
        self.operation_arc_fusion = nn.ModuleList()
        self.machine_arc_fusion = nn.ModuleList()
        for i in range(num_layers):
            self.machine_encoder_layers.append(self.machine_encoder_layer(d_model, d_hidden, num_head, d_kv))
            self.operation_encoder_layers.append(self.operation_encoder_layer(d_model, d_model, num_head, d_kv))
            operation_fusion = nn.Linear(d_model * 2, d_model)
            machine_fusion = nn.Linear(d_model * 2, d_model)
            nn.init.xavier_uniform_(operation_fusion.weight)
            nn.init.xavier_uniform_(machine_fusion.weight)
            nn.init.constant_(operation_fusion.bias, 0)
            nn.init.constant_(machine_fusion.bias, 0)
            self.operation_arc_fusion.append(operation_fusion)
            self.machine_arc_fusion.append(machine_fusion)

        self.w_o = nn.Linear(d_model, d_model)
        self.w_m = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.w_o.weight)
        nn.init.xavier_uniform_(self.w_m.weight)
        nn.init.constant_(self.w_o.bias, 0)
        nn.init.constant_(self.w_m.bias, 0)
        self.flat = nn.Flatten()

        self.dropout = nn.Dropout(dropout)

        self.bn1 = nn.BatchNorm1d(d_machine_raw)
        self.bn2 = nn.BatchNorm1d(d_operation_raw)
        self.bn3 = nn.BatchNorm1d(d_arc_raw)
        mlp0 = nn.Linear(d_model * 2, d_hidden)
        mlp1 = nn.Linear(d_hidden, 1)
        nn.init.xavier_uniform_(mlp0.weight)
        nn.init.xavier_uniform_(mlp1.weight)
        nn.init.constant_(mlp0.bias, 0)
        nn.init.constant_(mlp1.bias, 0)
        self.evaluate_mlp = nn.Sequential(
            mlp0,
            nn.GELU(),
            nn.Dropout(dropout),
            mlp1,
            # nn.Dropout(dropout),
        )

        self.operation_embedding = nn.Embedding(d_operation_raw, d_model)
        self.machine_embedding = nn.Embedding(d_machine_raw, d_model)
        self.arc_embedding = nn.Embedding(d_arc_raw, d_model)

    def normalize_raw_feature(self, state):
        w_o = F.sigmoid(torch.cat([self.bn2(feature).unsqueeze(0) for feature in state.operation_raw_feature], dim=0))
        w_m = F.sigmoid(torch.cat([self.bn1(feature).unsqueeze(0) for feature in state.machine_raw_feature], dim=0))
        _, num1, num2, dim = state.arc_raw_feature.size()
        w_arc = F.sigmoid(torch.cat([self.bn3(feature.view(-1, dim)).view(num1, num2, dim).unsqueeze(0) for feature in
                                     state.arc_raw_feature], dim=0))

        if self.cat:
            h_o = w_o.unsqueeze(-1) * self.operation_embedding.weight
            h_o = self.dropout(self.w_trans_o(h_o.view(*h_o.shape[:-2], -1)))
            h_m = w_m.unsqueeze(-1) * self.machine_embedding.weight
            h_m = self.dropout(self.w_trans_m(h_m.view(*h_m.shape[:-2], -1)))
            h_arc = w_arc.unsqueeze(-1) * self.arc_embedding.weight
            h_arc = self.dropout(self.w_trans_a(h_arc.view(*h_arc.shape[:-2], -1)))
        else:
            h_o = w_o @ self.operation_embedding.weight
            h_m = w_m @ self.machine_embedding.weight
            h_arc = w_arc @ self.arc_embedding.weight

        return h_o, h_m, h_arc

    def encode(self, state: State):
        h_o, h_m, h_arc = self.normalize_raw_feature(state)
        h_arc_m = h_arc
        h_arc_o = h_arc
        machin_encoder_mask = state.get_machine_encoder_mask()
        operation_encoder_mask = state.get_operation_encoder_mask()
        for i in range(self.num_layers):
            h_arc_o = self.operation_arc_fusion[i](state.cat_operation_arc(h_arc_o, h_o))
            h_arc_m = self.machine_arc_fusion[i](state.cat_machine_arc(h_arc_m, h_m))
            h_m = self.machine_encoder_layers[i](h_m, h_arc_o, machin_encoder_mask)
            h_o = self.operation_encoder_layers[i](h_o, h_arc_m, operation_encoder_mask,
                                                   state.operation_adj_matrix)
        return h_m, h_o

    def get_prop(self, state: State):
        h_m, h_o = self.encode(state)
        action_mask, runnable_cases = state.get_action_mask()
        h_m_ = self.w_m(h_m)
        h_o_ = self.w_o(h_o)
        x = h_m_.bmm(h_o_.transpose(-1, -2))
        x = x - x.max(dim=-1, keepdim=True)[0]
        x[action_mask == 0] = float('-inf')
        prop = F.softmax(self.flat(x), dim=-1)
        return prop, runnable_cases, h_o, h_m

    def forward(self, state: State, sample=True):
        prop, runnable_cases, _, _ = self.get_prop(state)
        num_operation_max = state.mask_operation.size(-1)
        action_indexes = torch.zeros(runnable_cases.size(0), dtype=torch.long).to(self.device)
        action_prop = torch.zeros(runnable_cases.size(0), dtype=torch.float).to(self.device)
        if sample:
            distribution = Categorical(prop[runnable_cases.squeeze()].squeeze())
            action_indexes[runnable_cases.squeeze()] = distribution.sample()
            action_prop[runnable_cases.squeeze()] = distribution.log_prob(action_indexes[runnable_cases.squeeze()])
        else:
            prop = prop[runnable_cases.squeeze()]
            action_indexes[runnable_cases.squeeze()] = prop.argmax(dim=-1)
            action_prop[runnable_cases.squeeze()] = prop[torch.arange(prop.size(0)).to(self.device), action_indexes[
                runnable_cases.squeeze()]]
        action_indexes[~runnable_cases.squeeze()] = 0
        actions = Action(action_prop, runnable_cases, num_operation_max, action_indexes)
        return prop, actions

    def evaluate(self, state: State, actions: torch.Tensor):
        prop, _, h_o, h_m = self.get_prop(state)
        if self.proj:
            h_o = self.dropout(self.o_proj(h_o))
            h_m = self.dropout(self.m_proj(h_m))

        operation_mask = (~state.mask_operation_finished * state.mask_operation).transpose(-1, -2)
        h_o = torch.where(operation_mask, h_o, torch.zeros_like(h_o).to(self.device))
        pooled_ho = h_o.sum(dim=1) / operation_mask.sum(dim=1).float()
        pooled_hm = h_m.mean(dim=1)
        h_global = torch.cat([pooled_ho, pooled_hm], dim=-1)

        values = self.evaluate_mlp(h_global)

        # get action prop
        distribution = Categorical(prop.squeeze())
        return distribution.log_prob(actions), values, distribution.entropy()
