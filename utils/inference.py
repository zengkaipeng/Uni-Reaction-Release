import torch


def beam_search_500mt(
	model, reac, prod, device, begin_token, toker, max_len=500, beams=10,
	total_heads=None, local_heads=0, end_token='<END>', pad_token='<PAD>'
):
	model, reac, prod = model.to(device), reac.to(device), prod.to(device)
	pad_index = tokenizer.token2idx[pad_token]
    end_index = tokenizer.token2idx[end_token]
    left_idx = tokenizer.token2idx.get('(', -1)
    right_idx = tokenizer.token2idx.get(')', -1)
    bs = reac.batch_mask.shape[0]
    start_tokens = torch.LongTensor(tokenizer.encode1d([begin_token] * bs))
    start_tokens = start_tokens.reshape(-1, 1).to(device)

	with torch.no_grad():
		sm_seq, sm_logs, sm_belong = model.beam_search(
            G=prod, start_seq=start_tokens, padding_idx=pad_index,
            end_idx=end_index, beam_rc=beam_rc, beam_seq=beam_smiles,
            max_len=max_len,  left_parenthesis_idx=left_idx,
            right_parenthesis_idx=right_idx
        )

    final_answer = [[] for _ in range(bs)]
    for idx, p in enumerate(sm_seq):
        smiles = tokenizer.decode1d(p[1:].tolist()).replace('`', '.')
        smiles = smiles.replace(end_token, '').replace(pad_token, '')
        xbelong = sm_belong[idx].item()
        final_answer[xbelong].append((sm_logs[idx].item(), smiles))

    return out_answer