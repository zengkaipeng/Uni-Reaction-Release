import torch
from .GATconv import SelfLoopGATConv
from .shared import SparseEdgeUpdateLayer
from utils.tensor_utils import graph2batch


class RAlingLayer(torch.nn.Module):
    def __init__(self, dim, dropout=0):
        super(RAlingLayer, self).__init__()
        self.comm_lin = torch.nn.Sequential(
            torch.nn.Linear(dim + dim, dim + dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim + dim, dim + dim)
        )

        self.lg_lin = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim)
        )
        self.dim = dim

    def forward(self, x_prod, x_reac, reac_mask):
        new_reac = torch.zeros_like(x_reac)
        shared_result = torch.cat([x_prod, x_reac[reac_mask]], dim=-1)
        shared_result = self.comm_lin(shared_result)
        new_prod = shared_result[:, :self.dim]
        new_reac[reac_mask] = shared_result[:, self.dim:]

        if torch.any(~reac_mask).item():
            new_reac[~reac_mask] = self.lg_lin(x_reac[~reac_mask])

        return new_prod, new_reac


class RAlignGATBlock(torch.nn.Module):
    def __init__(
        self, emb_dim, heads, edge_dim, reac_batch_infos={}, reac_num_keys={},
        prod_batch_infos={}, prod_num_keys={}, dropout=0.1,
        negative_slope=0.2, edge_update=True, fusion_order='ca_first'
    ):
        super(RAlignGATBlock, self).__init__()

        # Create condition adaptors for reactant and product graphs
        self.reac_condition_adaptor = ConditionAdaptor(
            emb_dim=emb_dim,
            batched_infos=reac_batch_infos,
            num_keys=reac_num_keys,
            dropout=dropout,
            fusion_order=fusion_order
        )

        self.prod_condition_adaptor = ConditionAdaptor(
            emb_dim=emb_dim,
            batched_infos=prod_batch_infos,
            num_keys=prod_num_keys,
            dropout=dropout,
            fusion_order=fusion_order
        )

        assert emb_dim % heads == 0, 'emb_dim must be divisible by heads'

        # GNN layers
        self.reac_mpnn = SelfLoopGATConv(
            in_channels=emb_dim, out_channels=emb_dim // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, negative_slope=negative_slope
        )
        self.prod_mpnn = SelfLoopGATConv(
            in_channels=emb_dim, out_channels=emb_dim // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, negative_slope=negative_slope
        )

        self.edge_update = edge_update

        # Cross-graph alignment layer
        self.fusion_layer = RAlingLayer(emb_dim, dropout)

        # LayerNorm layers
        self.reac_mpnn_ln = torch.nn.LayerNorm(emb_dim)
        self.prod_mpnn_ln = torch.nn.LayerNorm(emb_dim)
        self.reac_fusion_ln = torch.nn.LayerNorm(emb_dim)
        self.prod_fusion_ln = torch.nn.LayerNorm(emb_dim)

        # Edge update layers (if needed)
        if self.edge_update:
            self.reac_ue = SparseEdgeUpdateLayer(edge_dim, emb_dim, dropout)
            self.prod_ue = SparseEdgeUpdateLayer(edge_dim, emb_dim, dropout)
            self.reac_edge_ln = torch.nn.LayerNorm(emb_dim)
            self.prod_edge_ln = torch.nn.LayerNorm(emb_dim)

        self.drop_f = torch.nn.Dropout(dropout)

    def forward(
        self, reac_x, reac_e, reac_eidx, reac_bmask, shared_mask,
        prod_x, prod_e, prod_eidx, prod_bmask,
        reac_batched_condition={}, reac_num_conditions={},
        prod_batched_condition={}, prod_num_conditions={}
    ):
        # 1. Message passing within each graph
        reac_conv = self.reac_mpnn(
            x=reac_x, edge_attr=reac_e, edge_index=reac_eidx
        )
        prod_conv = self.prod_mpnn(
            x=prod_x, edge_attr=prod_e, edge_index=prod_eidx
        )

        # 2. Residual connection and LayerNorm
        prod_x = self.prod_mpnn_ln(self.drop_f(prod_conv) + prod_x)
        reac_x = self.reac_mpnn_ln(self.drop_f(reac_conv) + reac_x)

        # 3. Cross-graph alignment
        prod_u, reac_u = self.fusion_layer(
            x_prod=prod_x, x_reac=reac_x, reac_mask=shared_mask
        )

        prod_x = self.prod_fusion_ln(self.drop_f(prod_u) + prod_x)
        reac_x = self.reac_fusion_ln(reac_x + self.drop_f(reac_u))

        # 4. Condition fusion using the new adaptor classes
        reac_x = self.reac_condition_adaptor(
            feats=reac_x,
            batch_mask=reac_bmask,
            batched_conditions=reac_batched_condition,
            num_conditions=reac_num_conditions
        )

        prod_x = self.prod_condition_adaptor(
            feats=prod_x,
            batch_mask=prod_bmask,
            batched_conditions=prod_batched_condition,
            num_conditions=prod_num_conditions
        )

        # 5. Edge update (if enabled)
        if self.edge_update:
            reac_e_u = self.reac_ue(
                edge_feats=reac_e, node_feats=reac_x, edge_index=reac_eidx
            )
            prod_e_u = self.prod_ue(
                edge_feats=prod_e, node_feats=prod_x, edge_index=prod_eidx
            )
            reac_e = self.reac_edge_ln(reac_e + self.drop_f(reac_e_u))
            prod_e = self.prod_edge_ln(prod_e + self.drop_f(prod_e_u))

        return reac_x, prod_x, reac_e, prod_e


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

    def forward(self, feats, batch_mask, batched_conditions={}, num_conditions={}):
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
