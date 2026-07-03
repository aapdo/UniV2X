#-------------------------------------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                                      #
# Copyright (c) DAIR-V2X. All rights reserved.                                                    #
#-------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..dense_heads.track_head_plugin import Instances
from ..utils import fusion_audit


class PhysicalQueryAdapter(nn.Module):
    """Small metadata-conditioned rank adapter for physical-shift features."""

    def __init__(self, feature_dim, rank=8, metadata_dim=8, hidden_dim=64,
                 residual_scale=1.0, init_std=1e-4,
                 metadata_ablation=None):
        super(PhysicalQueryAdapter, self).__init__()
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.metadata_dim = int(metadata_dim)
        self.residual_scale = float(residual_scale)
        self.init_std = float(init_std)
        self.metadata_ablation = metadata_ablation or dict(mode='full')

        self.down = nn.Linear(self.feature_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.feature_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim + self.metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.rank),
        )
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.normal_(self.up.weight, mean=0.0, std=self.init_std)

    @staticmethod
    def _iter_shift_records(physical_shift):
        if not physical_shift:
            return []
        if isinstance(physical_shift, (list, tuple)):
            records = []
            for item in physical_shift:
                records.extend(PhysicalQueryAdapter._iter_shift_records(item))
            return records
        if not isinstance(physical_shift, dict):
            return []
        if physical_shift.get('name', '') in [
                'compound', 'missing_modality', 'modality_dropout',
                'unseen_sensor_setup']:
            return PhysicalQueryAdapter._iter_shift_records(physical_shift.get('shifts', []))
        return [physical_shift]

    def _pack_metadata(self, physical_shift, device, dtype):
        values = [0.0 for _ in range(self.metadata_dim)]
        for shift in self._iter_shift_records(physical_shift):
            severity = float(shift.get('severity', 0))
            values[0] = max(values[0], min(severity / 3.0, 1.0))

            if 'yaw_deg' in shift and self.metadata_dim > 1:
                values[1] = max(values[1], min(abs(float(shift['yaw_deg'])) / 5.0, 1.0))
            if 'translation_m' in shift and self.metadata_dim > 2:
                trans = shift.get('translation_m', [0.0, 0.0, 0.0])
                trans_mag = sum(float(v) ** 2 for v in trans[:3]) ** 0.5
                values[2] = max(values[2], min(trans_mag / 1.0, 1.0))
            if 'keep_ratio' in shift and self.metadata_dim > 3:
                values[3] = max(values[3], 1.0 - float(shift['keep_ratio']))
            if 'drop_probability' in shift and self.metadata_dim > 4:
                values[4] = max(values[4], float(shift['drop_probability']))
            if 'timestamp_delay_s' in shift and self.metadata_dim > 5:
                values[5] = max(values[5], min(float(shift['timestamp_delay_s']) / 2.0, 1.0))
            if 'effective_frame_delay' in shift and self.metadata_dim > 6:
                values[6] = max(values[6], min(float(shift['effective_frame_delay']) / 3.0, 1.0))
            if shift.get('label_preserving', False) and self.metadata_dim > 7:
                values[7] = 1.0
        values = self._apply_metadata_ablation(values)
        return torch.tensor(values, device=device, dtype=dtype)

    def _apply_metadata_ablation(self, values):
        mode = str(self.metadata_ablation.get('mode', 'full'))
        aliases = {
            'none': 'metadata_free',
            'zero': 'metadata_free',
            'free': 'metadata_free',
            'known': 'full',
            'all': 'full',
            'sensor_only': 'availability_only',
            'sensor_degradation_only': 'availability_only',
        }
        mode = aliases.get(mode, mode)
        if mode == 'full':
            return values
        if mode == 'metadata_free':
            return [0.0 for _ in values]
        if mode in ('shuffled', 'wrong'):
            default_perm = [5, 6, 3, 4, 1, 2, 0, 7]
            perm = self.metadata_ablation.get('permutation', default_perm)
            out = [0.0 for _ in values]
            for dst_idx, src_idx in enumerate(perm[:len(values)]):
                if 0 <= int(src_idx) < len(values):
                    out[dst_idx] = values[int(src_idx)]
            return out

        group_indices = {
            'severity_only': [0],
            'geometry_only': [1, 2],
            'temporal_only': [5, 6],
            'availability_only': [3, 4],
            'label_only': [7],
        }
        keep = self.metadata_ablation.get('active_indices', group_indices.get(mode, []))
        keep = {int(idx) for idx in keep if 0 <= int(idx) < len(values)}
        return [value if idx in keep else 0.0 for idx, value in enumerate(values)]

    def forward(self, features, physical_shift=None):
        if features.numel() == 0:
            return features
        metadata = self._pack_metadata(physical_shift, features.device, features.dtype)
        pooled = features.mean(dim=0)
        gate_in = torch.cat([pooled, metadata], dim=-1)
        gate = torch.sigmoid(self.gate(gate_in)).unsqueeze(0)
        residual = self.up(self.down(features) * gate) * self.residual_scale
        return features + residual


class AgentQueryFusion(nn.Module):

    def __init__(self, pc_range, embed_dims=256, physical_query_adapter=None):
        super(AgentQueryFusion, self).__init__()

        self.pc_range = pc_range
        self.embed_dims = embed_dims

        # reference_points ---> pos_embed
        self.get_pos_embedding = nn.Linear(3, self.embed_dims)
        # cross-agent feature alignment
        self.cross_agent_align = nn.Linear(self.embed_dims+9, self.embed_dims)
        self.cross_agent_align_pos = nn.Linear(self.embed_dims+9, self.embed_dims)
        self.cross_agent_fusion = nn.Linear(self.embed_dims, self.embed_dims)
        self.physical_query_adapter = None
        self.physical_query_adapter_position = 'pre_fusion'
        if physical_query_adapter and physical_query_adapter.get('enabled', False):
            adapter_cfg = physical_query_adapter.copy()
            adapter_cfg.pop('enabled', None)
            adapter_cfg.pop('freeze_non_adapter', None)
            adapter_cfg.pop('trainable_keys', None)
            self.physical_query_adapter_position = adapter_cfg.pop(
                'position', 'pre_fusion')
            self.physical_query_adapter = PhysicalQueryAdapter(
                feature_dim=self.embed_dims, **adapter_cfg)

        # parameter initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.physical_query_adapter is not None:
            self.physical_query_adapter.reset_parameters()
    
    def _loc_norm(self, locs, pc_range):
        """
        absolute (x,y,z) in global coordinate system ---> normalized (x,y,z)
        """
        from mmdet.models.utils.transformer import inverse_sigmoid

        locs[..., 0:1] = (locs[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        locs = inverse_sigmoid(locs)

        return locs
    
    def _loc_denorm(self, ref_pts, pc_range):
        """
        normalized (x,y,z) ---> absolute (x,y,z) in global coordinate system
        """
        locs = ref_pts.sigmoid().clone()

        locs[:, 0:1] = (locs[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])

        return locs
    
    def _dis_filt(self, veh_pts, inf_pts, veh_dims):
        """
        filter according to distance
        """
        diff = torch.abs(veh_pts - inf_pts) / veh_dims
        return diff[0] <= 1 and diff[1] <= 1 and diff[2] <= 1
    
    def _query_matching(self, inf_ref_pts, veh_ref_pts, veh_mask, veh_pred_dims):
        """
        inf_ref_pts: [..., 3] (xyz)
        veh_ref_pts: [..., 3] (xyz)
        veh_pred_dims: [..., 3] (dx, dy, dz)
        """
        inf_nums = inf_ref_pts.shape[0]
        veh_nums = veh_ref_pts.shape[0]
        cost_matrix = np.ones((veh_nums, inf_nums))
        cost_matrix.fill(1e6)

        for i in veh_mask:
            # for j in range(i,inf_nums):
            for j in range(inf_nums):
                cost_matrix[i][j] = torch.sum((veh_ref_pts[i] - inf_ref_pts[j])**2)**0.5
                if not self._dis_filt(veh_ref_pts[i], inf_ref_pts[j], veh_pred_dims[i]):
                    cost_matrix[i][j] = 1e6
        
        idx_veh, idx_inf = linear_sum_assignment(cost_matrix)

        return idx_veh, idx_inf, cost_matrix
    
    def _query_fusion(self, inf, veh, inf_idx, veh_idx, cost_matrix):
        """
        Query fusion: 
            replacement for scores, ref_pts and pos_embed according to confidence_score
            fusion for features via MLP
        
        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_idx: matched idxs for inf side
        veh_idx: matched idxs for veh side
        cost_matrix
        """

        veh_accept_idx = []
        inf_accept_idx = []

        for i in range(len(veh_idx)):
            if cost_matrix[veh_idx[i]][inf_idx[i]] < 1e5:
                veh_accept_idx.append(veh_idx[i])
                inf_accept_idx.append(inf_idx[i])
        if veh_accept_idx:
            fused_query = veh.query.clone()
            for veh_i, inf_i in zip(veh_accept_idx, inf_accept_idx):
                fused_query[veh_i, self.embed_dims:] = (
                    fused_query[veh_i, self.embed_dims:] +
                    self.cross_agent_fusion(inf.query[inf_i, self.embed_dims:])
                )
            veh.query = fused_query
        
        return veh, veh_accept_idx, inf_accept_idx
    

    def _query_complementation(self, inf, veh, inf_accept_idx):
        """
        Query complementation: replace low-confidence vehicle-side query with unmatched inf-side query

        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_accept_idx: idxs of matched instances
        """
        # supply_idx = -1
        for i in range(inf.ref_pts.shape[0]):
            if i not in inf_accept_idx:
                veh = Instances.cat([veh, inf[i]])

        return veh

    def adapt_bev_embed(self, bev_embed, physical_shift=None):
        if (self.physical_query_adapter is None or
                self.physical_query_adapter_position != 'bev_post_fusion'):
            return bev_embed
        original_shape = bev_embed.shape
        bev_feat = bev_embed.reshape(-1, original_shape[-1])
        bev_feat = self.physical_query_adapter(
            bev_feat, physical_shift=physical_shift)
        return bev_feat.reshape(original_shape)

    
    def forward(self, inf, veh, ego2other_rt, other_agent_pc_range, threshold=0.3,
                physical_shift=None):
        """
        Query-based cross-agent interaction: only update ref_pts and query.

        inf: Instance from infrastructure
        veh: Instance from vehicle
        ego2other_rt: calibration parameters from infrastructure to vehicle
        """
        audit_metrics = {
            'veh_query_count_input': len(veh),
            'infra_query_count_input': len(inf),
        }
        if getattr(inf, 'scores', None) is not None:
            audit_metrics.update(fusion_audit.tensor_stats(
                'infra_score_input', inf.scores))
        if getattr(veh, 'scores', None) is not None:
            audit_metrics.update(fusion_audit.tensor_stats(
                'veh_score_input', veh.scores))

        inf_mask = torch.where(inf.obj_idxes>=0)
        audit_metrics['infra_obj_idx_valid_count'] = int(inf_mask[0].numel())
        inf = inf[inf_mask]
        if len(inf) == 0:
            fusion_audit.emit(
                'agent', 'no_valid_infra_query', audit_metrics,
                physical_shift=physical_shift)
            return veh
        inf_mask_new = torch.where(inf.obj_idxes>=0)
        
        #not care obj_idxes of inf
        inf.obj_idxes = torch.ones_like(inf.obj_idxes) * -1
                
        # ref_pts norm2absolute
        inf_ref_pts = self._loc_denorm(inf.ref_pts, other_agent_pc_range)
        veh_ref_pts = self._loc_denorm(veh.ref_pts, self.pc_range)
            
        # inf_ref_pts inf2veh
        calib_inf2veh = np.linalg.inv(ego2other_rt[0].cpu().numpy().T)
        calib_inf2veh = inf_ref_pts.new_tensor(calib_inf2veh)
        inf_ref_pts = torch.cat((inf_ref_pts, torch.ones_like(inf_ref_pts[..., :1])), -1).unsqueeze(-1)
        inf_ref_pts = torch.matmul(calib_inf2veh, inf_ref_pts).squeeze(-1)[..., :3]

        # ego_selection
        remove_ego_ins = True
        if remove_ego_ins:
            infra_count_before_ego_filter = len(inf)
            H_B, H_F = -2.04, 2.04 # H = 4.084
            W_L, W_R = -0.92, 0.92 # W = 1.85
            def del_tensor_ele(arr,index):
                arr1 = arr[0:index]
                arr2 = arr[index+1:]
                return torch.cat((arr1,arr2),dim=0)

            inf_mask_new = list(inf_mask_new)
            for ii in range(len(inf_ref_pts)):
                xx, yy = inf_ref_pts[ii][0], inf_ref_pts[ii][1]
                if xx >= H_B and xx <= H_F and yy >= W_L and yy <= W_R:
                    inf_mask_new[0] = del_tensor_ele(inf_mask_new[0], ii)
                    break
            inf_mask_new = tuple(inf_mask_new)
            inf = inf[inf_mask_new]
            inf_ref_pts = inf_ref_pts[inf_mask_new]
            audit_metrics['infra_ego_box_removed_count'] = (
                infra_count_before_ego_filter - len(inf))
        audit_metrics['infra_after_spatial_filter_count'] = len(inf)

        # matching
        veh_mask = torch.where(veh.scores >= 0.05)[0]
        veh_idx, inf_idx, cost_matrix = self._query_matching(inf_ref_pts, veh_ref_pts, veh_mask, veh.pred_boxes[..., [2,3,5]]) # veh.pred_boxes x,y,dx,dy,z,dz
        finite_costs = cost_matrix[cost_matrix < 1e6]
        candidate_count = int(len(veh_mask) * len(inf))
        audit_metrics.update({
            'veh_score_ge_005_count': int(len(veh_mask)),
            'match_candidate_count': candidate_count,
            'match_finite_candidate_count': int(finite_costs.size),
            'match_distance_gate_reject_count': (
                candidate_count - int(finite_costs.size)),
            'match_hungarian_pair_count': int(len(veh_idx)),
        })
        audit_metrics.update(fusion_audit.tensor_stats(
            'match_cost_finite', finite_costs))

        # ref_pts normalization
        inf_ref_pts = self._loc_norm(inf_ref_pts, self.pc_range)
        veh_ref_pts = self._loc_norm(veh_ref_pts, self.pc_range)
        inf.ref_pts = inf_ref_pts
        veh.ref_pts = veh_ref_pts

        # cross-agent feature alignment
        inf2veh_r = calib_inf2veh[:3,:3].reshape(1,9).repeat(inf.query.shape[0], 1)
        inf_query_pos = self.cross_agent_align_pos(
            torch.cat([inf.query[..., :self.embed_dims], inf2veh_r], -1))
        inf_query_feat = self.cross_agent_align(
            torch.cat([inf.query[..., self.embed_dims:], inf2veh_r], -1))
        if (self.physical_query_adapter is not None and
                self.physical_query_adapter_position == 'pre_fusion'):
            inf_query_feat = self.physical_query_adapter(
                inf_query_feat, physical_shift=physical_shift)
        inf.query = torch.cat([inf_query_pos, inf_query_feat], dim=-1)

        # cross-agent query fusion
        veh, veh_accept_idx, inf_accept_idx = self._query_fusion(inf, veh, inf_idx, veh_idx, cost_matrix)
        audit_metrics.update({
            'match_accepted_count': int(len(veh_accept_idx)),
            'match_rejected_count': int(len(veh_idx) - len(veh_accept_idx)),
            'match_accept_rate_over_candidates': (
                float(len(veh_accept_idx)) / max(candidate_count, 1)),
            'match_accept_rate_over_hungarian': (
                float(len(veh_accept_idx)) / max(len(veh_idx), 1)),
        })

        # cross-agent query complementation
        veh_count_before_complement = len(veh)
        veh = self._query_complementation(inf, veh, inf_accept_idx)
        audit_metrics.update({
            'complement_added_count': int(len(veh) - veh_count_before_complement),
            'veh_query_count_output': int(len(veh)),
        })

        if (self.physical_query_adapter is not None and
                self.physical_query_adapter_position == 'post_fusion'):
            veh_query_feat = self.physical_query_adapter(
                veh.query[..., self.embed_dims:], physical_shift=physical_shift)
            veh.query = torch.cat(
                [veh.query[..., :self.embed_dims], veh_query_feat], dim=-1)

        fusion_audit.emit(
            'agent', 'fusion', audit_metrics, physical_shift=physical_shift)

        return veh
