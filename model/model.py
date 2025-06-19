import torch
from .layers import DotMhAttn
from .utils import graph2batch
from utils.tensor_utils import generate_square_subsequent_mask


class CNYieldModel(torch.nn.Module):
    def __init__(self, encoder, condition_encoder, dim, heads, dropout=0.1):
        super(CNYieldModel, self).__init__()
        self.encoder = encoder
        self.condition_encoder = condition_encoder
        self.out_head = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, 2)
        )
        self.pool_keys = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.pooler = DotMhAttn(
            Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim,
            emb_dim=dim, num_heads=heads, dropout=dropout
        )
        self.xln = torch.nn.LayerNorm(dim)

    def forward(self, reac_graph, prod_graph, conditions, cross_mask=None):
        condition_dict = self.condition_encoder(conditions)
        x_reac, x_prod, _, _ = self.encoder(
            reac_graph=reac_graph, reac_batched_condition=condition_dict,
            prod_graph=prod_graph, prod_batched_condition=condition_dict
        )

        x_reac = graph2batch(x_reac, reac_graph.batch_mask)
        x_prod = graph2batch(x_prod, prod_graph.batch_mask)
        memory = torch.cat([x_reac, x_prod], dim=1)
        memory_mask = [reac_graph.batch_mask, prod_graph.batch_mask]
        memory_mask = torch.logical_not(torch.cat(memory_mask, dim=1))

        pool_key = self.pool_keys.repeat(memory.shape[0], 1, 1)
        pooled_results, p_attn = self.pooler(
            query=pool_key, key=memory, value=memory,
            key_padding_mask=memory_mask, attn_mask=cross_mask
        )
        reaction_emb = self.xln(pooled_results.squeeze(dim=1))
        return self.out_head(reaction_emb)


class USPTOConditionModel(torch.nn.Module):
    def __init__(self, encoder, decoder, pe, n_words, dim):
        super(USPTOConditionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pe = pe
        self.n_words = n_words
        self.word_emb = torch.nn.Embedding(n_words, dim)
        self.cat_emb = torch.nn.Parameter(torch.randn(dim))
        self.sov_emb = torch.nn.Parameter(torch.randn(dim))
        self.reg_emb = torch.nn.Parameter(torch.randn(dim))
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, num_embs)
        )

    def encode(self, reac_graph, prod_graph):
        x_reac, x_prod, _, _ = self.encoder(
            reac_graph=reac_graph, prod_graph=prod_graph
        )
        x_reac = graph2batch(x_reac, reac_graph.batch_mask)
        x_prod = graph2batch(x_prod, prod_graph.batch_mask)
        memory = torch.cat([x_reac, x_prod], dim=1)
        memory_mask = [reac_graph.batch_mask, prod_graph.batch_mask]
        memory_mask = torch.logical_not(torch.cat(memory_mask, dim=1))
        return memory, memory_mask

    def decode_a_step(
        self, memory, seq, memory_key_padding_mask=None, cross_mask=None,
        tgt_mask=None, tgt_key_padding_mask=None
    ):
        type_emb = torch.stack([
            self.zeros_like(self.cat_emb), self.cat_emb,
            self.sov_emb, self.sov_emb, self.reg_emb
        ], dim=0)
        seq_emb = self.pe(self.word_emb(seq) + type_emb[:seq.shape[-1]])
        seq = self.decoder(
            tgt=seq_emb, memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, memory_mask=cross_mask
        )
        return self.out_layer(seq)

    def forward(
        self, reac_graph, prod_graph, tgt, tgt_mask=None,
        tgt_key_padding_mask=None, cross_mask=None
    ):
        memory, memory_pad = self.encode(reac_graph, prod_graph)
        assert tgt.shape[1] <= 5, 'Invalid Format for prediction'
        result = self.decode_a_step(
            tgt, memory, tgt_mask=tgt_mask, cross_mask=cross_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_pad
        )

        return result

    def beam_search(
        self, memory, begin, memory_key_padding_mask=None,
        cross_mask=None, beam_size=10
    ):
        def sq_ft(x):
            assert x.dim() >= 2, 'no enough dim to squeeze'
            return x.reshape((-1, *x.shape[2:]))
        (bs, ml), device = memory.shape[:2], memory.device
        if memory_padding is None:
            memory_padding = torch.zeros((bs, ml), dtype=bool).to(device)

        res = torch.LongTensor([[begin]] * bs).to(device)
        belong = torch.arange(0, bs).to(device)
        log_logits = torch.zeros(bs).to(device)

        for i in range(5):
            mem_cand, mem_pad_cand, seq_cand = [], [], []
            log_cand, bel_cand = [], []
            diag_mask = generate_square_subsequent_mask(i + 1, device)
            rc_out = torch.log_softmax(self.decode_a_step(
                memory=memory,  cross_mask=cross_mask, tgt_mask=diag_mask,
                seq=res, memory_key_padding_mask=memory_key_padding_mask
            )[:, -1], dim=-1)

            dup = min(self.n_words, beam_size)

            memory = memory[:, None].repeat(1, dup, 1, 1)
            # [alive_n, beam, ml, dim]
            memory_padding = memory_padding[:, None].repeat(1, dup, 1)
            # [alive_n, dup, ml]
            res = res[:, None].repeat(1, dup, 1)
            # [alive_n, dup, seq_len]
            belong = belong[:, None].repeat(1, dup)
            # [alive_n, dup]

            top_res = torch.topk(rc_out, k=dup, dim=-1, largest=True)
            log_logits = log_logits[:, None] + top_res.values
            res = torch.cat([res, top_res.indices.unsqueeze(dim=-1)], dim=-1)

            mem_cand.append(sq_ft(memory))
            mem_pad_cand.append(sq_ft(memory_padding))
            seq_cand.append(sq_ft(res))
            log_cand.append(sq_ft(log_logits))
            bel_cand.append(sq_ft(belong))

            memory = torch.cat(mem_cand, dim=0)
            memory_padding = torch.cat(mem_pad_cand, dim=0)
            res = torch.cat(seq_cand, dim=0)
            log_logits = torch.cat(log_cand, dim=0)
            belong = torch.cat(bel_cand, dim=0)

            sort_out = torch.sort(log_logits, descending=True)
            log_logits = sort_out.values
            memory = memory[sort_out.indices]
            res = res[sort_out.indices]
            memory_padding = memory_padding[sort_out.indices]
            belong = belong[sort_out.indices]

            topk_mask = generate_topk_mask(belong, bs, beam_size)
            log_logits = log_logits[topk_mask]
            res = res[topk_mask]
            belong = belong[topk_mask]
            memory = memory[topk_mask]
            memory_padding = memory_padding[topk_mask]

        return res, log_logits, belong
