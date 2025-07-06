import torch


def convert_log_into_label(logits, mod='sigmoid'):
    if mod == 'sigmoid':
        pred = torch.zeros_like(logits)
        pred[logits >= 0] = 1
        pred[logits < 0] = 0
    elif mod == 'softmax':
        pred = torch.argmax(logits, dim=-1)
    else:
        raise NotImplementedError(f'Invalid mode {mod}')
    return pred


def correct_trans_output(trans_pred, end_idx, pad_idx):
    batch_size, max_len = trans_pred.shape
    device = trans_pred.device
    x_range = torch.arange(0, max_len, 1).unsqueeze(0)
    x_range = x_range.repeat(batch_size, 1).to(device)

    y_cand = (torch.ones_like(trans_pred).long() * max_len + 12).to(device)
    y_cand[trans_pred == end_idx] = x_range[trans_pred == end_idx]
    min_result = torch.min(y_cand, dim=-1, keepdim=True)
    end_pos = min_result.values
    trans_pred[x_range > end_pos] = pad_idx
    return trans_pred


def data_eval_trans(trans_pred, trans_lb, return_tensor=False):
    batch_size, max_len = trans_pred.shape
    line_acc = torch.sum(trans_pred == trans_lb, dim=-1) == max_len
    line_acc = line_acc.cpu()
    return line_acc if return_tensor else (line_acc.sum().item(), batch_size)


def generate_square_subsequent_mask(sz, device='cpu'):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = (mask == 0).to(device)
    return mask


def generate_tgt_mask(tgt, pad_idx, device='cpu'):
    siz = tgt.shape[1]
    tgt_pad_mask = (tgt == pad_idx).to(device)
    tgt_sub_mask = generate_square_subsequent_mask(siz, device)
    return tgt_pad_mask, tgt_sub_mask


def generate_local_global_mask(reac, prod, Qlen, total_heads, local_heads):
    reac_rc = torch.zeros_like(reac.batch_mask)
    prod_rc = torch.zeros_like(prod.batch_mask)
    reac_rc[reac.batch_mask] = reac.is_rc | (~reac.is_prod)
    prod_rc[prod.batch_mask] = prod.is_rc

    rcx = torch.cat([reac_rc, prod_rc], dim=1)
    rcx = rcx.unsqueeze(1).unsqueeze(-1)
    # [bs, 1, klen, 1]

    global_mask = torch.ones_like(rcx)

    rcx = rcx.repeat(1, Qlen, 1, local_heads)
    global_mask = global_mask.repeat(1, Qlen, 1, total_heads - local_heads)
    return torch.logical_not(torch.cat([rcx, global_mask], dim=-1))


def generate_topk_mask(belong, max_num, k):
    # belong: [bs]
    x_idx = torch.arange(max_num).to(belong)
    eq_mask = belong == x_idx[:, None]
    # i row: which index of belong equal to i
    cir_x = torch.cumsum(eq_mask, dim=-1)
    # i row: how many belong equal to i
    top_k_mask = torch.logical_and(cir_x <= k, eq_mask)
    # i row: top-k of belong i
    return torch.any(top_k_mask, dim=0)


def calc_trans_loss(trans_pred, trans_lb, ignore_index):
    batch_size, maxl, num_c = trans_pred.shape
    trans_pred = trans_pred.reshape(-1, num_c)
    trans_lb = trans_lb.reshape(-1)

    losses = torch.nn.functional.cross_entropy(
        trans_pred, trans_lb, reduction='none',
        ignore_index=ignore_index,
    )
    losses = losses.reshape(batch_size, maxl)
    loss = torch.mean(torch.sum(losses, dim=-1))
    return loss
