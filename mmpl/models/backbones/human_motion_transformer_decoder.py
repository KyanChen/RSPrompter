import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import math
from mmpl.registry import MODELS


def PE1d_sincos(seq_length, dim):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                          -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(1)


class PositionEmbedding(nn.Module):
    """
    Absolute pos embedding (standard), learned.
    """

    def __init__(self, seq_length, dim, dropout, grad=False):
        super().__init__()
        if grad:
            self.embed = nn.Parameter(data=PE1d_sincos(seq_length, dim), requires_grad=grad)
        else:
            self.register_buffer('embed', PE1d_sincos(seq_length, dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x.shape: bs, seq_len, feat_dim
        l = x.shape[1]
        x = x.permute(1, 0, 2) + self.embed[:l].expand(x.permute(1, 0, 2).shape)
        x = self.dropout(x.permute(1, 0, 2))
        return x


@MODELS.register_module()
class MotionTransformer(nn.Module):

    def __init__(self,
                 num_vq=1024,
                 embed_dim=512,
                 block_size=16,
                 num_layers=2,
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4):
        super().__init__()
        self.trans_base = CrossCondTransBase(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq

    def get_block_size(self):
        return self.block_size

    def forward(self, input_ids, *args, **kwargs):
        feat = self.trans_base(input_ids)
        logits = self.trans_head(feat)
        return logits

    def sample(self, prompt_ids, sample_length=32, if_categorial=False):
        x = prompt_ids
        for _ in range(prompt_ids.shape[-1], sample_length):
            logits = self.forward(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                # if idx[0] == self.num_vq:
                #     break
            # append to the sequence and continue
            x = torch.cat((x, idx), dim=1)
        return x


class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CrossCondTransBase(nn.Module):

    def __init__(self,
                 num_vq=1024,
                 embed_dim=512,
                 block_size=16,
                 num_layers=2,
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq + 1, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(
            *[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])

        self.pos_embed = PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx):
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)
            # token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)

        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(self,
                 num_vq=1024,
                 embed_dim=512,
                 block_size=16,
                 num_layers=2,
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(
            *[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits






