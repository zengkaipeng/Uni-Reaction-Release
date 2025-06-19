import torch


def graph2batch(
    node_feat: torch.Tensor, batch_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_node = batch_mask.shape
    answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
    answer = answer.to(node_feat.device)
    answer[batch_mask] = node_feat
    return answer
