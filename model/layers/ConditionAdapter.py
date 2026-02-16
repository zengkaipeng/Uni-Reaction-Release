import torch


class ConditionAdaptor(torch.nn.Module):
    def __init__(
        self, emb_dim, batched_infos={}, num_keys={},
        dropout=0.1, fusion_order='ca_first'
    ):
        """
        Condition adaptor for fusing different types of conditions

        Args:
            emb_dim: Feature dimension
            batched_infos: Dict for batched (cross-attention) conditions
            num_keys: Dict for numerical (FiLM) conditions
            dropout: Dropout rate
            fusion_order: Order of condition fusion. Options:
                - 'ca_first': Cross-attention first, then FiLM
                - 'film_first': FiLM first, then cross-attention
                - 'simultaneous': Both conditions applied simultaneously
                - list of list: Custom execution plan
        """
        super(ConditionAdaptor, self).__init__()

        self.emb_dim = emb_dim
        self.dropout = dropout

        # Store original condition info
        self.batched_infos = batched_infos
        self.num_keys = num_keys

        # Cross-attention adapters for batched conditions
        self.batch_adapter = torch.nn.ModuleDict({
            k: torch.nn.MultiheadAttention(
                embed_dim=emb_dim, num_heads=v['heads'], dropout=dropout,
                batch_first=True, kdim=v['dim'], vdim=v['dim']
            ) for k, v in batched_infos.items()
        })

        # FiLM adapters for numerical conditions
        self.num_adapter = torch.nn.ModuleDict({
            k: torch.nn.ModuleDict({
                'beta': torch.nn.Linear(v, emb_dim),
                'gamma': torch.nn.Linear(v, emb_dim)
            }) for k, v in num_keys.items()
        })

        # Parse fusion order and create execution plan
        self.execution_plan = self._parse_fusion_order(fusion_order)

        # Create LayerNorm for each condition block that needs it
        self.block_norms = torch.nn.ModuleDict()
        for i, block in enumerate(self.execution_plan):
            if self._block_has_batched_condition(block):
                self.block_norms[str(i)] = torch.nn.LayerNorm(emb_dim)

        self.drop_f = torch.nn.Dropout(dropout)

    def _parse_fusion_order(self, fusion_order):
        """Parse fusion order into execution plan"""
        batched_keys = list(self.batched_infos.keys())
        num_keys = list(self.num_keys.keys())

        if isinstance(fusion_order, list):
            # Already in list of list format
            # Validate the format
            for i, block in enumerate(fusion_order):
                for j, item in enumerate(block):
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        raise ValueError(
                            f"Block {i}, item {j} should be "
                            f"a tuple/list of (key, type)"
                        )
                    key, cond_type = item
                    if cond_type not in ['batched', 'num']:
                        raise ValueError(
                            f"Block {i}, item {j}: type must be "
                            f"'batched' or 'num', got {cond_type}"
                        )
                    if cond_type == 'batched' and key not in self.batched_infos:
                        raise ValueError(
                            f"Block {i}, item {j}: batched key "
                            f"'{key}' not found in batched_infos"
                        )
                    if cond_type == 'num' and key not in self.num_keys:
                        raise ValueError(
                            f"Block {i}, item {j}: num key"
                            f" '{key}' not found in num_keys"
                        )
            return fusion_order
        elif fusion_order == 'ca_first':
            # Cross-attention first, then FiLM
            execution_plan = []
            if batched_keys:
                execution_plan.append([(k, 'batched') for k in batched_keys])
            if num_keys:
                execution_plan.append([(k, 'num') for k in num_keys])
            return execution_plan
        elif fusion_order == 'film_first':
            # FiLM first, then cross-attention
            execution_plan = []
            if num_keys:
                execution_plan.append([(k, 'num') for k in num_keys])
            if batched_keys:
                execution_plan.append([(k, 'batched') for k in batched_keys])
            return execution_plan
        elif fusion_order == 'simultaneous':
            # Both conditions applied simultaneously
            execution_plan = []
            block = []
            if batched_keys:
                block.extend([(k, 'batched') for k in batched_keys])
            if num_keys:
                block.extend([(k, 'num') for k in num_keys])
            if block:
                execution_plan.append(block)
            return execution_plan
        else:
            raise ValueError(f"Unknown fusion order: {fusion_order}")

    def _block_has_batched_condition(self, block):
        """Check if a block contains any batched condition"""
        return any(cond_type == 'batched' for _, cond_type in block)

    def _apply_condition_block(
        self, batch_feats, block, block_idx, batched_conditions, num_conditions
    ):
        """Apply a single condition block (simultaneous fusion within the block)"""
        ca_bias = torch.zeros_like(batch_feats)
        film_bias = torch.zeros_like(batch_feats)

        # 分别计算CA bias和FiLM bias
        for key, cond_type in block:
            if cond_type == 'batched' and key in self.batch_adapter:
                # Apply cross-attention condition
                this_info = batched_conditions.get(key)
                if this_info is not None:
                    bias = self._apply_single_cross_attention(
                        batch_feats, this_info, self.batch_adapter[key]
                    )
                    ca_bias += bias
            elif cond_type == 'num' and key in self.num_adapter:
                # Apply FiLM condition
                this_num = num_conditions.get(key)
                if this_num is not None:
                    bias = self._apply_single_film(
                        batch_feats, this_num, self.num_adapter[key]
                    )
                    film_bias += bias

        # 检查是否需要应用LayerNorm
        has_batched_in_block = self._block_has_batched_condition(block)

        if has_batched_in_block:
            # 获取该block对应的LayerNorm
            if str(block_idx) not in self.block_norms:
                raise RuntimeError(
                    f"Block {block_idx} contains batched conditions but no LayerNorm was created. "
                    f"This indicates a mismatch between the initialization and execution plan."
                )
            norm_layer = self.block_norms[str(block_idx)]
            # 应用LayerNorm到CA bias部分，然后添加FiLM bias
            batch_feats = norm_layer(batch_feats + ca_bias) + film_bias
        else:
            # 没有batched条件，直接添加所有bias
            batch_feats = batch_feats + ca_bias + film_bias

        return batch_feats

    def _apply_single_cross_attention(self, batch_feats, condition_info, adapter):
        """Apply single cross-attention condition"""
        if condition_info.get('group_id', None) is None:
            assert condition_info['embedding'].shape[0] == batch_feats.shape[0], \
                'The condition is not organized in batch'
            bias, _w = adapter(
                query=batch_feats, key=condition_info['embedding'],
                value=condition_info['embedding'],
                key_padding_mask=condition_info.get('padding_mask', None)
            )
            return self.drop_f(bias)
        else:
            qry = torch.index_select(
                batch_feats, 0, condition_info['group_id']
            )
            bias, _w = adapter(
                query=qry, key=condition_info['embedding'],
                value=condition_info['embedding'],
                key_padding_mask=condition_info.get('padding_mask', None)
            )
            bias = self.drop_f(bias)
            result = torch.zeros_like(batch_feats)
            result.index_add_(0, condition_info['group_id'], bias)
            return result

    def _apply_single_film(self, batch_feats, num_condition, adapter):
        """Apply single FiLM condition"""
        if isinstance(num_condition, dict):
            gamma = adapter['gamma'](num_condition['embedding'])
            beta = adapter['beta'](num_condition['embedding'])
            bias = gamma * batch_feats[num_condition['group_id']] + beta
            result = torch.zeros_like(batch_feats)
            result.index_add_(0, num_condition['group_id'], bias)
            return result
        else:
            gamma = adapter['gamma'](num_condition)
            beta = adapter['beta'](num_condition)
            return gamma * batch_feats + beta

    def forward(
        self, feats, batch_mask, batched_conditions={}, num_conditions={}
    ):
        """
        Forward pass for condition adaptor

        Args:
            feats: Input features [num_nodes, emb_dim]
            batch_mask: Mask to convert graph to batch representation
            batched_conditions: Dict of batched conditions for cross-attention
            num_conditions: Dict of numerical conditions for FiLM

        Returns:
            Conditioned features [num_nodes, emb_dim]
        """
        # Convert graph features to batch representation
        batch_feats = graph2batch(feats, batch_mask)

        # Apply conditions according to execution plan
        for i, block in enumerate(self.execution_plan):
            batch_feats = self._apply_condition_block(
                batch_feats, block, i, batched_conditions, num_conditions
            )

        # Convert back to graph representation
        return batch_feats[batch_mask]
