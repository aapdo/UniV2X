#----------------------------------------------------------------#
# UniV2X: End-to-End Autonomous Driving through V2X Cooperation  #
# Source code: https://github.com/AIR-THU/UniV2X                 #
# Copyright (c) DAIR-V2X. All rights reserved.                   #
# Contact: yuhaibao94@gmail.com                                  #
#----------------------------------------------------------------#

import copy
import contextlib
import torch
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

class MultiAgent(MVXTwoStageDetector):
    def __init__(self, model_ego_agent, model_other_agents={}):
        super(MultiAgent, self).__init__()
        self.model_ego_agent = model_ego_agent

        self.other_agent_names = []
        self.pc_range_dict = {}
        for name_other_agent, model_other_agent in model_other_agents.items():
            setattr(self, name_other_agent, model_other_agent)
            self.other_agent_names.append(name_other_agent)
            if hasattr(model_other_agent, 'pc_range'):
                self.pc_range_dict[name_other_agent] = model_other_agent.pc_range


    def forward(self, ego_agent_data=None, other_agent_data_dict={}, return_loss=True, w_label=True, **kwargs):
        if return_loss:
            return self.forward_train(ego_agent_data=ego_agent_data,
                                                                other_agent_data_dict=other_agent_data_dict,
                                                                return_loss=return_loss,
                                                                **kwargs)
        else:
            return self.forward_test(ego_agent_data=ego_agent_data,
                                                                other_agent_data_dict=other_agent_data_dict,
                                                                return_loss=return_loss,
                                                                w_label=w_label,
                                                                **kwargs)

    def forward_train(self, ego_agent_data=None, other_agent_data_dict={}, return_loss=True, **kwargs):
        # UniV2X TODO: hardcode with 'img_metas'
        kwargs.pop('img_metas', None)

        batch_size = self._infer_batch_size(ego_agent_data)
        if batch_size > 1:
            return self._forward_train_batched(
                ego_agent_data=ego_agent_data,
                other_agent_data_dict=other_agent_data_dict,
                return_loss=return_loss,
                batch_size=batch_size,
                **kwargs)

        return self._forward_train_single(
            ego_agent_data=ego_agent_data,
            other_agent_data_dict=other_agent_data_dict,
            return_loss=return_loss,
            **kwargs)

    def _forward_train_single(self, ego_agent_data=None, other_agent_data_dict={},
                              return_loss=True, **kwargs):

        other_agent_results = {}
        for name_other_agent in self.other_agent_names:
            model_other_agent = getattr(self, name_other_agent)
            context = (
                contextlib.nullcontext()
                if self._has_trainable_parameters(model_other_agent)
                else torch.no_grad()
            )
            with context:
                loss, univ2x_outs = getattr(self, name_other_agent)(
                        return_loss=return_loss,
                        **(other_agent_data_dict[name_other_agent]),
                        **kwargs
                )
                univ2x_outs['ego2other_rt'] = other_agent_data_dict[name_other_agent]['veh2inf_rt']
                univ2x_outs['pc_range'] = self.pc_range_dict[name_other_agent]
                univ2x_outs['physical_shift'] = self._extract_physical_shift(
                    other_agent_data_dict[name_other_agent])
                other_agent_results[name_other_agent] = univ2x_outs

        loss, univ2x_outs = self.model_ego_agent(return_loss=return_loss, other_agent_results=other_agent_results, **ego_agent_data, **kwargs)

        return loss

    def forward_test(self, ego_agent_data=None, other_agent_data_dict={}, w_label=True, return_loss=False, **kwargs):
        batch_size = self._infer_batch_size(ego_agent_data, test_mode=True)
        if batch_size > 1:
            return self._forward_test_batched(
                ego_agent_data=ego_agent_data,
                other_agent_data_dict=other_agent_data_dict,
                w_label=w_label,
                return_loss=return_loss,
                batch_size=batch_size,
                **kwargs)

        return self._forward_test_single(
            ego_agent_data=ego_agent_data,
            other_agent_data_dict=other_agent_data_dict,
            w_label=w_label,
            return_loss=return_loss,
            **kwargs)

    def _forward_test_single(self, ego_agent_data=None, other_agent_data_dict={},
                             w_label=True, return_loss=False, **kwargs):
        other_agent_results = {}
        for name_other_agent in self.other_agent_names:
            other_agent_result = getattr(self, name_other_agent)(
                    return_loss=return_loss,
                    w_label=w_label,
                    **(other_agent_data_dict[name_other_agent]),
                    **kwargs
            )
            other_agent_result[0]['ego2other_rt'] = other_agent_data_dict[name_other_agent]['veh2inf_rt']
            other_agent_result[0]['pc_range'] = self.pc_range_dict[name_other_agent]
            other_agent_result[0]['physical_shift'] = self._extract_physical_shift(
                other_agent_data_dict[name_other_agent])
            other_agent_results[name_other_agent] = other_agent_result

        result = self.model_ego_agent(return_loss=return_loss, w_label=w_label, other_agent_results=other_agent_results, **ego_agent_data, **kwargs)

        return result

    def _forward_train_batched(self, ego_agent_data=None, other_agent_data_dict={},
                               return_loss=True, batch_size=1, **kwargs):
        loss_sum = {}
        for batch_idx in range(batch_size):
            ego_single = self._slice_agent_data(
                ego_agent_data, batch_idx, batch_size, test_mode=False)
            other_single = {
                name: self._slice_agent_data(
                    data, batch_idx, batch_size, test_mode=False)
                for name, data in other_agent_data_dict.items()
            }
            kwargs_single = self._slice_agent_data(
                kwargs, batch_idx, batch_size, test_mode=False)
            loss = self._forward_train_single(
                ego_agent_data=ego_single,
                other_agent_data_dict=other_single,
                return_loss=return_loss,
                **kwargs_single)
            for key, value in loss.items():
                value = value / float(batch_size) if torch.is_tensor(value) else value
                loss_sum[key] = value if key not in loss_sum else loss_sum[key] + value
        return loss_sum

    def _forward_test_batched(self, ego_agent_data=None, other_agent_data_dict={},
                              w_label=True, return_loss=False, batch_size=1,
                              **kwargs):
        results = []
        for batch_idx in range(batch_size):
            ego_single = self._slice_agent_data(
                ego_agent_data, batch_idx, batch_size, test_mode=True)
            other_single = {
                name: self._slice_agent_data(
                    data, batch_idx, batch_size, test_mode=True)
                for name, data in other_agent_data_dict.items()
            }
            kwargs_single = self._slice_agent_data(
                kwargs, batch_idx, batch_size, test_mode=True)
            result = self._forward_test_single(
                ego_agent_data=ego_single,
                other_agent_data_dict=other_single,
                w_label=w_label,
                return_loss=return_loss,
                **kwargs_single)
            results.extend(result)
        return results

    @classmethod
    def _infer_batch_size(cls, agent_data, test_mode=False):
        if not isinstance(agent_data, dict):
            return 1
        for key in ('img', 'img_metas'):
            batch_size = cls._infer_batch_size_from_value(
                agent_data.get(key, None), test_mode=test_mode)
            if batch_size > 1:
                return batch_size
        return 1

    @classmethod
    def _infer_batch_size_from_value(cls, value, test_mode=False):
        value = cls._unwrap_data(value)
        if value is None:
            return 1
        if torch.is_tensor(value):
            return int(value.size(0)) if value.dim() > 0 else 1
        if isinstance(value, dict):
            for child in value.values():
                batch_size = cls._infer_batch_size_from_value(
                    child, test_mode=test_mode)
                if batch_size > 1:
                    return batch_size
            return 1
        if isinstance(value, (list, tuple)):
            if not value:
                return 1
            if len(value) == 1:
                inner = cls._unwrap_data(value[0])
                if torch.is_tensor(inner) and inner.dim() > 0:
                    return int(inner.size(0))
                if test_mode and isinstance(inner, (list, tuple)):
                    return len(inner)
            return len(value)
        return 1

    @classmethod
    def _slice_agent_data(cls, value, batch_idx, batch_size, test_mode=False):
        value = cls._unwrap_data(value)
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.dim() > 0 and value.size(0) == batch_size:
                return value[batch_idx:batch_idx + 1]
            return value
        if isinstance(value, dict):
            return {
                key: cls._slice_agent_data(
                    child, batch_idx, batch_size, test_mode=test_mode)
                for key, child in value.items()
            }
        if isinstance(value, tuple):
            return tuple(cls._slice_sequence(
                list(value), batch_idx, batch_size, test_mode=test_mode))
        if isinstance(value, list):
            return cls._slice_sequence(
                value, batch_idx, batch_size, test_mode=test_mode)
        return copy.deepcopy(value)

    @classmethod
    def _slice_sequence(cls, value, batch_idx, batch_size, test_mode=False):
        if len(value) == batch_size:
            return [copy.deepcopy(value[batch_idx])]
        if len(value) == 1:
            inner = cls._unwrap_data(value[0])
            if torch.is_tensor(inner) and inner.dim() > 0 and inner.size(0) == batch_size:
                return [inner[batch_idx:batch_idx + 1]]
            if test_mode and isinstance(inner, (list, tuple)) and len(inner) == batch_size:
                return [[copy.deepcopy(inner[batch_idx])]]
        return [
            cls._slice_agent_data(
                child, batch_idx, batch_size, test_mode=test_mode)
            for child in value
        ]

    @staticmethod
    def _unwrap_data(value):
        if hasattr(value, 'data'):
            return value.data
        return value

    @staticmethod
    def _has_trainable_parameters(module):
        return any(param.requires_grad for param in module.parameters())

    @classmethod
    def _first_meta(cls, value):
        value = cls._unwrap_data(value)
        if isinstance(value, dict):
            if 'physical_shift' not in value:
                frame_keys = [k for k, v in value.items() if isinstance(v, dict)]
                if frame_keys:
                    try:
                        last_key = sorted(frame_keys)[-1]
                    except TypeError:
                        last_key = frame_keys[-1]
                    return cls._first_meta(value[last_key])
            return value
        if isinstance(value, (list, tuple)) and value:
            return cls._first_meta(value[0])
        return None

    @classmethod
    def _extract_physical_shift(cls, agent_data):
        if not isinstance(agent_data, dict):
            return None
        meta = cls._first_meta(agent_data.get('img_metas', None))
        if not isinstance(meta, dict):
            return None
        return meta.get('physical_shift', None)
