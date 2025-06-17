import torch


def generate_local_global_mask(reac, prod, Qlen, heads):
    assert heads >= 2, 'heads too small for dividing'
    reac_rc = torch.zeros_like(reac.batch_mask)
    prod_rc = torch.zeros_like(prod.batch_mask)
    reac_rc[reac.batch_mask] = reac.is_rc | (~reac.is_prod)
    prod_rc[prod.batch_mask] = prod.is_rc

    rcx = torch.cat([reac_rc, prod_rc], dim=1)
    rcx = rcx.unsqueeze(1).unsqueeze(-1)
    # [bs, 1, klen, 1]

    global_mask = torch.ones_like(rcx)

    x, y = heads >> 1, heads - (heads >> 1)
    rcx = rcx.repeat(1, Qlen, 1, x)
    global_mask = global_mask.repeat(1, Qlen, 1, y)
    return torch.logical_not(torch.cat([rcx, global_mask], dim=-1))


def graph2batch(
    node_feat: torch.Tensor, batch_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_node = batch_mask.shape
    answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
    answer = answer.to(node_feat.device)
    answer[batch_mask] = node_feat
    return answer
