
"""Compatibility shims for smoke-testing legacy UniMM-V2X on OpenMMLab 2.x.

This module is not imported automatically. Import it explicitly in diagnostic
scripts before importing projects.mmdet3d_plugin.
"""
import sys
import types


def _install_module(name, module):
    sys.modules.setdefault(name, module)
    if '.' in name:
        parent_name, child_name = name.rsplit('.', 1)
        parent = sys.modules.get(parent_name)
        if parent is not None and not hasattr(parent, child_name):
            setattr(parent, child_name, module)
    return sys.modules[name]



# NumPy 1.24 removed legacy aliases used by the original UniMM-V2X code.
try:
    import numpy as _np
    if not hasattr(_np, 'bool'):
        _np.bool = bool
    if not hasattr(_np, 'int'):
        _np.int = int
    if not hasattr(_np, 'float'):
        _np.float = float
except Exception:
    pass


# mmdet 3.x moved these wrappers to mmengine.dataset. Legacy UniMM-V2X imports
# them from mmdet.datasets.dataset_wrappers, so expose aliases without changing
# wrapper behavior.
try:
    import mmdet.datasets.dataset_wrappers as _mmdet_dataset_wrappers
    from mmengine.dataset import (
        ClassBalancedDataset as _MMENGINE_ClassBalancedDataset,
        RepeatDataset as _MMENGINE_RepeatDataset,
        ConcatDataset as _MMENGINE_ConcatDataset,
    )
    if not hasattr(_mmdet_dataset_wrappers, "ClassBalancedDataset"):
        _mmdet_dataset_wrappers.ClassBalancedDataset = _MMENGINE_ClassBalancedDataset
    if not hasattr(_mmdet_dataset_wrappers, "RepeatDataset"):
        _mmdet_dataset_wrappers.RepeatDataset = _MMENGINE_RepeatDataset
    if not hasattr(_mmdet_dataset_wrappers, "ConcatDataset"):
        _mmdet_dataset_wrappers.ConcatDataset = _MMENGINE_ConcatDataset
except Exception:
    pass




# mmcv 2.x no longer exposes FileClient at the top level. Legacy UniMM-V2X
# uses mmcv.FileClient(...); mmengine.fileio.FileClient is the moved API.
try:
    import mmcv as _mmcv_for_fileclient
    from mmengine.fileio import FileClient as _MMENGINE_FileClient
    if not hasattr(_mmcv_for_fileclient, 'FileClient'):
        _mmcv_for_fileclient.FileClient = _MMENGINE_FileClient
except Exception:
    pass


# mmcv.runner compatibility -------------------------------------------------
try:
    import mmcv
    import mmengine
    from mmengine.config import Config, DictAction
    from mmengine.utils import ProgressBar
    from mmengine.model import BaseModule
    from mmengine.dist import get_dist_info, init_dist
    from mmengine.runner.checkpoint import load_checkpoint
    from mmengine.registry import HOOKS, OPTIMIZERS, TRANSFORMS as MMENGINE_TRANSFORMS
    from mmengine.hooks import Hook
except Exception:  # pragma: no cover - best effort shim
    pass
else:
    if not hasattr(mmcv, 'dump'):
        mmcv.dump = mmengine.dump
    if not hasattr(mmcv, 'load'):
        mmcv.load = mmengine.load
    if not hasattr(mmcv, 'mkdir_or_exist'):
        mmcv.mkdir_or_exist = mmengine.mkdir_or_exist
    if not hasattr(mmcv, 'Config'):
        mmcv.Config = Config
    if not hasattr(mmcv, 'DictAction'):
        mmcv.DictAction = DictAction
    if not hasattr(mmcv, 'ProgressBar'):
        mmcv.ProgressBar = ProgressBar

    def _identity_fp_decorator(*dargs, **dkwargs):
        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return dargs[0]
        def _wrap(func):
            return func
        return _wrap

    def wrap_fp16_model(model):
        return model

    runner_mod = types.ModuleType('mmcv.runner')
    runner_mod.BaseModule = BaseModule
    runner_mod.get_dist_info = get_dist_info
    runner_mod.init_dist = init_dist
    runner_mod.load_checkpoint = load_checkpoint
    runner_mod.wrap_fp16_model = wrap_fp16_model
    runner_mod.force_fp32 = _identity_fp_decorator
    runner_mod.auto_fp16 = _identity_fp_decorator
    runner_mod.HOOKS = HOOKS
    runner_mod.OPTIMIZERS = OPTIMIZERS
    _install_module('mmcv.runner', runner_mod)

    base_module_mod = types.ModuleType('mmcv.runner.base_module')
    base_module_mod.BaseModule = BaseModule
    try:
        from torch.nn import ModuleList, Sequential
        base_module_mod.ModuleList = ModuleList
        base_module_mod.Sequential = Sequential
    except Exception:
        pass
    _install_module('mmcv.runner.base_module', base_module_mod)

    fp16_mod = types.ModuleType('mmcv.runner.fp16_utils')
    fp16_mod.force_fp32 = _identity_fp_decorator
    fp16_mod.auto_fp16 = _identity_fp_decorator
    _install_module('mmcv.runner.fp16_utils', fp16_mod)

    opt_builder_mod = types.ModuleType('mmcv.runner.optimizer.builder')
    opt_builder_mod.OPTIMIZERS = OPTIMIZERS
    _install_module('mmcv.runner.optimizer.builder', opt_builder_mod)

    hook_mod = types.ModuleType('mmcv.runner.hooks.hook')
    hook_mod.HOOKS = HOOKS
    hook_mod.Hook = Hook
    _install_module('mmcv.runner.hooks.hook', hook_mod)


# mmcv.parallel compatibility ----------------------------------------------
try:
    from torch.nn.parallel import DataParallel as MMDataParallel
    from torch.nn.parallel import DistributedDataParallel as MMDistributedDataParallel
except Exception:
    pass
else:
    class DataContainer:
        def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
            self.data = data
            self._data = data
            self.stack = stack
            self.padding_value = padding_value
            self.cpu_only = cpu_only
            self.pad_dims = pad_dims

        def __repr__(self):
            return f'DataContainer({self.data!r})'

    def collate(batch, samples_per_gpu=1):
        from collections.abc import Mapping, Sequence
        from torch.utils.data._utils.collate import default_collate
        if not batch:
            return batch
        elem = batch[0]
        if isinstance(elem, DataContainer):
            values = [sample.data for sample in batch]
            if elem.cpu_only:
                return values
            if elem.stack:
                try:
                    return default_collate(values)
                except Exception:
                    try:
                        import torch
                        return torch.stack(values, dim=0)
                    except Exception:
                        return values
            return values
        if isinstance(elem, Mapping):
            return {
                key: collate([sample[key] for sample in batch], samples_per_gpu)
                for key in elem
            }
        if isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(collate(samples, samples_per_gpu) for samples in zip(*batch)))
        if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
            try:
                if all(len(sample) == len(elem) for sample in batch):
                    return [collate(samples, samples_per_gpu) for samples in zip(*batch)]
            except Exception:
                pass
            return list(batch)
        try:
            return default_collate(batch)
        except Exception:
            return list(batch)

    parallel_mod = types.ModuleType('mmcv.parallel')
    parallel_mod.MMDataParallel = MMDataParallel
    parallel_mod.MMDistributedDataParallel = MMDistributedDataParallel
    parallel_mod.DataContainer = DataContainer
    parallel_mod.collate = collate
    _install_module('mmcv.parallel', parallel_mod)


# mmdet / mmdet3d old registry aliases --------------------------------------
try:
    from mmdet.registry import MODELS as MMDET_MODELS, DATASETS as MMDET_DATASETS, TASK_UTILS
    from mmdet3d.registry import MODELS as MMDET3D_MODELS, DATASETS as MMDET3D_DATASETS, TRANSFORMS
except Exception:
    pass
else:
    # Import the package for its OpenMMLab-2.x transform registrations.
    try:
        import mmdet3d.datasets.transforms  # noqa: F401
        from mmdet3d.datasets.transforms import MultiScaleFlipAug3D

        class LegacyMultiScaleFlipAug3D(MultiScaleFlipAug3D):
            """Match the dict-of-augmentations output expected by UniMM."""

            def transform(self, results):
                augmented = super().transform(results)
                if not augmented:
                    return {}
                return {
                    key: [data[key] for data in augmented]
                    for key in augmented[0]
                }

        MMENGINE_TRANSFORMS.register_module(
            name='MultiScaleFlipAug3D',
            module=LegacyMultiScaleFlipAug3D,
            force=True)
    except Exception:
        pass

    models_builder = types.ModuleType('mmdet.models.builder')
    models_builder.BACKBONES = MMDET3D_MODELS
    models_builder.HEADS = MMDET3D_MODELS
    models_builder.LOSSES = MMDET3D_MODELS
    models_builder.DETECTORS = MMDET3D_MODELS
    models_builder.NECKS = MMDET3D_MODELS
    models_builder.build_loss = MMDET3D_MODELS.build
    models_builder.build_head = MMDET3D_MODELS.build
    _install_module('mmdet.models.builder', models_builder)

    mmdet3d_models_builder = types.ModuleType('mmdet3d.models.builder')
    mmdet3d_models_builder.BACKBONES = MMDET3D_MODELS
    mmdet3d_models_builder.HEADS = MMDET3D_MODELS
    mmdet3d_models_builder.MIDDLE_ENCODERS = MMDET3D_MODELS
    mmdet3d_models_builder.DETECTORS = MMDET3D_MODELS
    mmdet3d_models_builder.NECKS = MMDET3D_MODELS
    mmdet3d_models_builder.build_model = MMDET3D_MODELS.build
    mmdet3d_models_builder.build_loss = MMDET3D_MODELS.build
    mmdet3d_models_builder.build_head = MMDET3D_MODELS.build
    _install_module('mmdet3d.models.builder', mmdet3d_models_builder)

    bbox_builder = types.ModuleType('mmdet.core.bbox.builder')
    bbox_builder.BBOX_ASSIGNERS = TASK_UTILS
    bbox_builder.BBOX_SAMPLERS = TASK_UTILS
    bbox_builder.BBOX_CODERS = TASK_UTILS
    bbox_builder.build_assigner = TASK_UTILS.build
    bbox_builder.build_sampler = TASK_UTILS.build
    bbox_builder.build_bbox_coder = TASK_UTILS.build
    _install_module('mmdet.core.bbox.builder', bbox_builder)

    match_builder = types.ModuleType('mmdet.core.bbox.match_costs.builder')
    match_builder.MATCH_COST = TASK_UTILS
    _install_module('mmdet.core.bbox.match_costs.builder', match_builder)

    match_costs = types.ModuleType('mmdet.core.bbox.match_costs')
    match_costs.build_match_cost = TASK_UTILS.build
    match_costs.MATCH_COST = TASK_UTILS
    _install_module('mmdet.core.bbox.match_costs', match_costs)

    datasets_builder = types.ModuleType('mmdet.datasets.builder')
    datasets_builder.DATASETS = MMDET_DATASETS
    datasets_builder.PIPELINES = MMENGINE_TRANSFORMS
    datasets_builder.build_dataset = MMDET_DATASETS.build
    datasets_builder._concat_dataset = lambda cfg, default_args=None: MMDET_DATASETS.build(cfg, default_args=default_args)
    _install_module('mmdet.datasets.builder', datasets_builder)

    try:
        import mmdet.datasets as _mmdet_datasets
        _mmdet_datasets.DATASETS = MMDET_DATASETS
        _mmdet_datasets.build_dataset = MMDET_DATASETS.build
    except Exception:
        pass

    try:
        import mmdet.datasets.dataset_wrappers as _mmdet_dataset_wrappers
        from mmengine.dataset import (
            ClassBalancedDataset as _MMENGINE_ClassBalancedDataset,
            RepeatDataset as _MMENGINE_RepeatDataset,
            ConcatDataset as _MMENGINE_ConcatDataset,
        )
        if not hasattr(_mmdet_dataset_wrappers, "ClassBalancedDataset"):
            _mmdet_dataset_wrappers.ClassBalancedDataset = _MMENGINE_ClassBalancedDataset
        if not hasattr(_mmdet_dataset_wrappers, "RepeatDataset"):
            _mmdet_dataset_wrappers.RepeatDataset = _MMENGINE_RepeatDataset
        if not hasattr(_mmdet_dataset_wrappers, "ConcatDataset"):
            _mmdet_dataset_wrappers.ConcatDataset = _MMENGINE_ConcatDataset
    except Exception:
        pass

    try:
        import mmdet3d.datasets as _mmdet3d_datasets
        _mmdet3d_datasets.DATASETS = MMDET3D_DATASETS
        _mmdet3d_datasets.PIPELINES = MMENGINE_TRANSFORMS
        _mmdet3d_datasets.build_dataset = MMDET3D_DATASETS.build
    except Exception:
        pass


# Additional old mmdet.core hierarchy aliases --------------------------------
try:
    import types
    from mmdet.utils import reduce_mean
    from mmdet.models.utils import multi_apply
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
    import mmdet.structures.mask as _mask
    from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
    from mmdet.models.task_modules.samplers import BaseSampler
    from mmdet.models.task_modules.samplers.random_sampler import RandomSampler
    from mmdet.models.task_modules.samplers.sampling_result import SamplingResult
    from mmdet.models.task_modules.coders import BaseBBoxCoder
except Exception:
    pass
else:
    core_mod = types.ModuleType('mmdet.core')
    core_mod.reduce_mean = reduce_mean
    core_mod.multi_apply = multi_apply
    core_mod.bbox_cxcywh_to_xyxy = bbox_cxcywh_to_xyxy
    core_mod.bbox_xyxy_to_cxcywh = bbox_xyxy_to_cxcywh
    core_mod.mask = _mask
    core_mod.build_assigner = TASK_UTILS.build
    _install_module('mmdet.core', core_mod)

    bbox_mod = types.ModuleType('mmdet.core.bbox')
    bbox_mod.BaseBBoxCoder = BaseBBoxCoder
    bbox_mod.bbox_cxcywh_to_xyxy = bbox_cxcywh_to_xyxy
    bbox_mod.bbox_xyxy_to_cxcywh = bbox_xyxy_to_cxcywh
    bbox_mod.demodata = types.SimpleNamespace()
    _install_module('mmdet.core.bbox', bbox_mod)

    bbox_transforms = types.ModuleType('mmdet.core.bbox.transforms')
    bbox_transforms.bbox_cxcywh_to_xyxy = bbox_cxcywh_to_xyxy
    bbox_transforms.bbox_xyxy_to_cxcywh = bbox_xyxy_to_cxcywh
    _install_module('mmdet.core.bbox.transforms', bbox_transforms)

    assigners_mod = types.ModuleType('mmdet.core.bbox.assigners')
    assigners_mod.AssignResult = AssignResult
    assigners_mod.BaseAssigner = BaseAssigner
    _install_module('mmdet.core.bbox.assigners', assigners_mod)

    assign_result_mod = types.ModuleType('mmdet.core.bbox.assigners.assign_result')
    assign_result_mod.AssignResult = AssignResult
    _install_module('mmdet.core.bbox.assigners.assign_result', assign_result_mod)

    base_assigner_mod = types.ModuleType('mmdet.core.bbox.assigners.base_assigner')
    base_assigner_mod.BaseAssigner = BaseAssigner
    _install_module('mmdet.core.bbox.assigners.base_assigner', base_assigner_mod)

    samplers_mod = types.ModuleType('mmdet.core.bbox.samplers')
    samplers_mod.BaseSampler = BaseSampler
    samplers_mod.RandomSampler = RandomSampler
    samplers_mod.SamplingResult = SamplingResult
    _install_module('mmdet.core.bbox.samplers', samplers_mod)

    base_sampler_mod = types.ModuleType('mmdet.core.bbox.samplers.base_sampler')
    base_sampler_mod.BaseSampler = BaseSampler
    _install_module('mmdet.core.bbox.samplers.base_sampler', base_sampler_mod)

    random_sampler_mod = types.ModuleType('mmdet.core.bbox.samplers.random_sampler')
    random_sampler_mod.RandomSampler = RandomSampler
    _install_module('mmdet.core.bbox.samplers.random_sampler', random_sampler_mod)

    sampling_result_mod = types.ModuleType('mmdet.core.bbox.samplers.sampling_result')
    sampling_result_mod.SamplingResult = SamplingResult
    _install_module('mmdet.core.bbox.samplers.sampling_result', sampling_result_mod)

    coders_mod = types.ModuleType('mmdet.core.bbox.coders')
    coders_mod.BaseBBoxCoder = BaseBBoxCoder
    coders_mod.build_bbox_coder = TASK_UTILS.build
    _install_module('mmdet.core.bbox.coders', coders_mod)

    _install_module('mmdet.core.mask', _mask)


# mmdet.models.utils.transformer old alias ----------------------------------
try:
    import types
    from mmdet.models.layers.transformer import inverse_sigmoid
except Exception:
    pass
else:
    transformer_mod = types.ModuleType('mmdet.models.utils.transformer')
    transformer_mod.inverse_sigmoid = inverse_sigmoid
    _install_module('mmdet.models.utils.transformer', transformer_mod)


# mmdet3d.core old hierarchy aliases ----------------------------------------
try:
    import types
    from mmdet3d.registry import TASK_UTILS as MMDET3D_TASK_UTILS
    from mmdet3d.structures import (
        xywhr2xyxyr, Box3DMode, Coord3DMode, LiDARInstance3DBoxes,
        CameraInstance3DBoxes, DepthInstance3DBoxes, BaseInstance3DBoxes,
        bbox3d2result, BasePoints)
    from mmdet3d.structures.points import get_points_type
    from mmdet3d.structures.ops.iou3d_calculator import BboxOverlaps3D
except Exception:
    pass
else:
    core3d_mod = types.ModuleType('mmdet3d.core')
    for _name, _obj in {
        'xywhr2xyxyr': xywhr2xyxyr,
        'Box3DMode': Box3DMode,
        'Coord3DMode': Coord3DMode,
        'LiDARInstance3DBoxes': LiDARInstance3DBoxes,
        'CameraInstance3DBoxes': CameraInstance3DBoxes,
        'DepthInstance3DBoxes': DepthInstance3DBoxes,
        'BaseInstance3DBoxes': BaseInstance3DBoxes,
        'bbox3d2result': bbox3d2result,
    }.items():
        setattr(core3d_mod, _name, _obj)
    _install_module('mmdet3d.core', core3d_mod)

    bbox3d_mod = types.ModuleType('mmdet3d.core.bbox')
    for _name, _obj in {
        'xywhr2xyxyr': xywhr2xyxyr,
        'Box3DMode': Box3DMode,
        'Coord3DMode': Coord3DMode,
        'LiDARInstance3DBoxes': LiDARInstance3DBoxes,
        'CameraInstance3DBoxes': CameraInstance3DBoxes,
        'DepthInstance3DBoxes': DepthInstance3DBoxes,
        'BaseInstance3DBoxes': BaseInstance3DBoxes,
    }.items():
        setattr(bbox3d_mod, _name, _obj)
    _install_module('mmdet3d.core.bbox', bbox3d_mod)

    coders3d_mod = types.ModuleType('mmdet3d.core.bbox.coders')
    coders3d_mod.build_bbox_coder = MMDET3D_TASK_UTILS.build
    _install_module('mmdet3d.core.bbox.coders', coders3d_mod)

    iou_pkg = types.ModuleType('mmdet3d.core.bbox.iou_calculators')
    iou_pkg.BboxOverlaps3D = BboxOverlaps3D
    _install_module('mmdet3d.core.bbox.iou_calculators', iou_pkg)

    iou_mod = types.ModuleType('mmdet3d.core.bbox.iou_calculators.iou3d_calculator')
    iou_mod.BboxOverlaps3D = BboxOverlaps3D
    _install_module('mmdet3d.core.bbox.iou_calculators.iou3d_calculator', iou_mod)

    points_mod = types.ModuleType('mmdet3d.core.points')
    points_mod.BasePoints = BasePoints
    points_mod.get_points_type = get_points_type
    _install_module('mmdet3d.core.points', points_mod)


# Force-compatible match cost registry for legacy duplicate names ------------
try:
    class _ForceRegistry:
        def __init__(self, registry):
            self._registry = registry
        def register_module(self, *args, **kwargs):
            kwargs.setdefault('force', True)
            return self._registry.register_module(*args, **kwargs)
        def build(self, *args, **kwargs):
            return self._registry.build(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._registry, name)

    _compat_match_cost = _ForceRegistry(TASK_UTILS)
    if 'mmdet.core.bbox.match_costs.builder' in sys.modules:
        sys.modules['mmdet.core.bbox.match_costs.builder'].MATCH_COST = _compat_match_cost
    if 'mmdet.core.bbox.match_costs' in sys.modules:
        sys.modules['mmdet.core.bbox.match_costs'].MATCH_COST = _compat_match_cost
        sys.modules['mmdet.core.bbox.match_costs'].build_match_cost = _compat_match_cost.build
except Exception:
    pass


# Legacy EvalHook stubs -------------------------------------------------------
try:
    class _CompatEvalHook(Hook):
        def __init__(self, dataloader=None, interval=1, by_epoch=True, save_best=None,
                     rule=None, test_fn=None, greater_keys=None, less_keys=None,
                     broadcast_bn_buffer=True, tmpdir=None, gpu_collect=False, **kwargs):
            self.dataloader = dataloader
            self.interval = interval
            self.by_epoch = by_epoch
            self.save_best = save_best
            self.rule = rule
            self.test_fn = test_fn
            self.broadcast_bn_buffer = broadcast_bn_buffer
            self.tmpdir = tmpdir
            self.gpu_collect = gpu_collect
        def _should_evaluate(self, runner):
            return False
        def evaluate(self, runner, results):
            return None
        def _save_ckpt(self, runner, key_score):
            return None
        def before_train_epoch(self, runner):
            return None
        def before_train_iter(self, runner, batch_idx=None, data_batch=None):
            return None

    if 'mmcv.runner' in sys.modules:
        sys.modules['mmcv.runner'].EvalHook = _CompatEvalHook
        sys.modules['mmcv.runner'].DistEvalHook = _CompatEvalHook

    eval_hooks_mod = types.ModuleType('mmdet.core.evaluation.eval_hooks')
    eval_hooks_mod.EvalHook = _CompatEvalHook
    eval_hooks_mod.DistEvalHook = _CompatEvalHook
    _install_module('mmdet.core.evaluation.eval_hooks', eval_hooks_mod)

    evaluation_mod = types.ModuleType('mmdet.core.evaluation')
    evaluation_mod.EvalHook = _CompatEvalHook
    evaluation_mod.DistEvalHook = _CompatEvalHook
    _install_module('mmdet.core.evaluation', evaluation_mod)
except Exception:
    pass


# mmdet.datasets.pipelines old alias ----------------------------------------
try:
    import types
    from mmdet.datasets.transforms.formatting import to_tensor
except Exception:
    pass
else:
    pipelines_mod = types.ModuleType('mmdet.datasets.pipelines')
    pipelines_mod.to_tensor = to_tensor
    _install_module('mmdet.datasets.pipelines', pipelines_mod)


# Legacy Whales eval aliases used by UniMM custom evaluator ------------------
try:
    import types
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.detection.data_classes import DetectionBox, DetectionConfig, DetectionMetrics
    from nuscenes.eval.tracking.data_classes import TrackingBox
    import mmdet3d.datasets as _mmdet3d_datasets
except Exception:
    pass
else:
    _mmdet3d_datasets.Whales = NuScenes
    _mmdet3d_datasets.WhalesEval = NuScenesEval
    _mmdet3d_datasets.WhalesTrackingEval = TrackingEval
    whales_eval_mod = types.ModuleType('mmdet3d.datasets.whales_eval')
    whales_eval_mod.WhalesDetectionBox = DetectionBox
    whales_eval_mod.WhalesTrackingBox = TrackingBox
    whales_eval_mod.WhalesDetectionConfig = DetectionConfig
    whales_eval_mod.WhalesDetectionMetrics = DetectionMetrics
    _install_module('mmdet3d.datasets.whales_eval', whales_eval_mod)


# mmcv.utils old registry aliases -------------------------------------------
try:
    import mmcv.utils as _mmcv_utils
    from mmengine.registry import Registry, build_from_cfg
    from mmengine.config import ConfigDict
except Exception:
    pass
else:
    _mmcv_utils.Registry = Registry
    _mmcv_utils.build_from_cfg = build_from_cfg
    if not hasattr(_mmcv_utils, 'ConfigDict'):
        _mmcv_utils.ConfigDict = ConfigDict
    if not hasattr(_mmcv_utils, 'deprecated_api_warning'):
        def deprecated_api_warning(*args, **kwargs):
            def _decorator(func):
                return func
            return _decorator
        _mmcv_utils.deprecated_api_warning = deprecated_api_warning


# mmdet.datasets.samplers legacy GroupSampler --------------------------------
try:
    import torch
    import mmdet.datasets.samplers as _mmdet_samplers
except Exception:
    pass
else:
    class GroupSampler(torch.utils.data.Sampler):
        def __init__(self, dataset, samples_per_gpu=1):
            self.dataset = dataset
            self.samples_per_gpu = samples_per_gpu
        def __iter__(self):
            return iter(range(len(self.dataset)))
        def __len__(self):
            return len(self.dataset)
    _mmdet_samplers.GroupSampler = GroupSampler


# mmcv.utils.registry old module alias ---------------------------------------
try:
    import types
    from mmengine.registry import Registry, build_from_cfg
except Exception:
    pass
else:
    registry_mod = types.ModuleType('mmcv.utils.registry')
    registry_mod.Registry = Registry
    registry_mod.build_from_cfg = build_from_cfg
    _install_module('mmcv.utils.registry', registry_mod)


# mmdet3d.core.bbox.box_np_ops old alias ------------------------------------
try:
    import mmdet3d.structures.ops.box_np_ops as _box_np_ops
except Exception:
    pass
else:
    if 'mmdet3d.core.bbox' in sys.modules:
        sys.modules['mmdet3d.core.bbox'].__path__ = []
    _install_module('mmdet3d.core.bbox.box_np_ops', _box_np_ops)


# mmdet3d.datasets.pipelines old aliases ------------------------------------
try:
    import types
    import mmdet3d.datasets.transforms as _transforms3d
    import mmdet3d.datasets.transforms.transforms_3d as _transforms_3d
    import mmdet3d.datasets.transforms.loading as _loading3d
except Exception:
    pass
else:
    _install_module('mmdet3d.datasets.pipelines', _transforms3d)
    _install_module('mmdet3d.datasets.pipelines.transforms_3d', _transforms_3d)
    _install_module('mmdet3d.datasets.pipelines.loading', _loading3d)

    formating3d_mod = types.ModuleType('mmdet3d.datasets.pipelines.formating')
    class DefaultFormatBundle3D:
        def __init__(self, class_names=None, with_gt=True, with_label=True, *args, **kwargs):
            self.class_names = class_names
            self.with_gt = with_gt
            self.with_label = with_label

        def _to_tensor(self, value):
            try:
                from mmdet.datasets.pipelines import to_tensor
                return to_tensor(value)
            except Exception:
                import torch
                return torch.as_tensor(value)

        def __call__(self, results):
            try:
                import numpy as _np
                from mmcv.parallel import DataContainer as DC
            except Exception:
                return results

            if 'img' in results:
                img = results['img']
                if isinstance(img, list):
                    img = _np.stack(img, axis=0)
                if hasattr(img, 'shape') and len(img.shape) >= 3:
                    if len(img.shape) == 3:
                        img = _np.ascontiguousarray(img.transpose(2, 0, 1))
                    elif len(img.shape) == 4 and img.shape[-1] in (1, 3):
                        img = _np.ascontiguousarray(img.transpose(0, 3, 1, 2))
                results['img'] = DC(self._to_tensor(img), stack=True)

            for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore',
                        'gt_labels', 'gt_labels_3d', 'attr_labels',
                        'pts_instance_mask', 'pts_semantic_mask',
                        'centers2d', 'depths', 'gt_sdc_label']:
                if key in results and not isinstance(results[key], DC):
                    results[key] = DC(self._to_tensor(results[key]))

            for key in ['gt_bboxes_3d', 'gt_sdc_bbox']:
                if key in results and not isinstance(results[key], DC):
                    results[key] = DC(results[key], cpu_only=True)

            return results
    formating3d_mod.DefaultFormatBundle3D = DefaultFormatBundle3D
    _install_module('mmdet3d.datasets.pipelines.formating', formating3d_mod)
    setattr(_transforms3d, 'DefaultFormatBundle3D', DefaultFormatBundle3D)
    try:
        from mmengine.registry import TRANSFORMS as _MMENGINE_TRANSFORMS
        _MMENGINE_TRANSFORMS.register_module(
            name='DefaultFormatBundle3D', module=DefaultFormatBundle3D, force=True)
    except Exception:
        pass


# mmcv.cnn moved weight init aliases -----------------------------------------
try:
    import mmcv.cnn as _mmcv_cnn
    from mmengine.model import bias_init_with_prob, constant_init, kaiming_init, normal_init, xavier_init
except Exception:
    pass
else:
    _mmcv_cnn.bias_init_with_prob = bias_init_with_prob
    _mmcv_cnn.constant_init = constant_init
    _mmcv_cnn.kaiming_init = kaiming_init
    _mmcv_cnn.normal_init = normal_init
    _mmcv_cnn.xavier_init = xavier_init


# mmcv.utils version helpers --------------------------------------------------
try:
    import torch
    import mmcv.utils as _mmcv_utils
    from mmengine.utils import digit_version
except Exception:
    pass
else:
    _mmcv_utils.TORCH_VERSION = torch.__version__
    _mmcv_utils.digit_version = digit_version


# mmdet 3.x removed the legacy test seed helper.
try:
    import mmdet.apis as _mmdet_apis
    from mmengine.runner import set_random_seed as _set_random_seed
except Exception:
    pass
else:
    if not hasattr(_mmdet_apis, 'set_random_seed'):
        _mmdet_apis.set_random_seed = _set_random_seed


# mmdet.models / mmdet3d.models old top-level registries ---------------------
try:
    import mmdet.models as _mmdet_models
    import mmdet3d.models as _mmdet3d_models
except Exception:
    pass
else:
    _mmdet_models.HEADS = MMDET3D_MODELS
    _mmdet_models.BACKBONES = MMDET3D_MODELS
    _mmdet_models.LOSSES = MMDET3D_MODELS
    _mmdet_models.DETECTORS = MMDET3D_MODELS
    _mmdet_models.NECKS = MMDET3D_MODELS
    _mmdet_models.build_loss = MMDET3D_MODELS.build
    _mmdet_models.build_head = MMDET3D_MODELS.build
    _mmdet3d_models.HEADS = MMDET3D_MODELS
    _mmdet3d_models.BACKBONES = MMDET3D_MODELS
    _mmdet3d_models.MIDDLE_ENCODERS = MMDET3D_MODELS
    _mmdet3d_models.DETECTORS = MMDET3D_MODELS
    _mmdet3d_models.NECKS = MMDET3D_MODELS
    _mmdet3d_models.build_model = MMDET3D_MODELS.build


# mmdet.core top-level sampler/bbox builders --------------------------------
try:
    if 'mmdet.core' in sys.modules:
        sys.modules['mmdet.core'].build_sampler = TASK_UTILS.build
        sys.modules['mmdet.core'].build_bbox_coder = TASK_UTILS.build
        sys.modules['mmdet.core'].build_assigner = TASK_UTILS.build
except Exception:
    pass


# Legacy transformer registries/builders -------------------------------------
try:
    import types
    import mmdet.models.utils as _mmdet_model_utils
    from mmcv.cnn.bricks.transformer import MODELS as _TRANSFORMER_MODELS
except Exception:
    pass
else:
    def build_transformer(cfg, default_args=None):
        return _TRANSFORMER_MODELS.build(cfg, default_args=default_args)

    transformer_registry_mod = types.ModuleType('mmcv.cnn.bricks.registry')
    for _name in ['ATTENTION', 'FEEDFORWARD_NETWORK', 'POSITIONAL_ENCODING',
                  'TRANSFORMER', 'TRANSFORMER_LAYER', 'TRANSFORMER_LAYER_SEQUENCE']:
        setattr(transformer_registry_mod, _name, _TRANSFORMER_MODELS)
    _install_module('mmcv.cnn.bricks.registry', transformer_registry_mod)

    mmdet_utils_builder_mod = types.ModuleType('mmdet.models.utils.builder')
    mmdet_utils_builder_mod.TRANSFORMER = _TRANSFORMER_MODELS
    mmdet_utils_builder_mod.build_transformer = build_transformer
    _install_module('mmdet.models.utils.builder', mmdet_utils_builder_mod)
    _mmdet_model_utils.TRANSFORMER = _TRANSFORMER_MODELS
    _mmdet_model_utils.build_transformer = build_transformer


# Legacy mmdet.models.utils.Transformer base ---------------------------------
try:
    import torch.nn as nn
    from mmengine.model import BaseModule
    from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
    import mmdet.models.utils as _mmdet_model_utils
except Exception:
    pass
else:
    class Transformer(BaseModule):
        def __init__(self, encoder=None, decoder=None, init_cfg=None, **kwargs):
            super().__init__(init_cfg=init_cfg)
            self.encoder = build_transformer_layer_sequence(encoder) if encoder is not None else None
            self.decoder = build_transformer_layer_sequence(decoder) if decoder is not None else None
            self.embed_dims = getattr(self.encoder, 'embed_dims', getattr(self.decoder, 'embed_dims', None))
        def init_weights(self):
            pass
        def forward(self, *args, **kwargs):
            raise NotImplementedError
    _mmdet_model_utils.Transformer = Transformer
    if 'mmdet.models.utils.transformer' in sys.modules:
        sys.modules['mmdet.models.utils.transformer'].Transformer = Transformer


# More mmdet3d iou calculator aliases ---------------------------------------
try:
    from mmdet3d.structures.ops.iou3d_calculator import (
        bbox_overlaps_nearest_3d, bbox_overlaps_3d, axis_aligned_bbox_overlaps_3d)
except Exception:
    pass
else:
    if 'mmdet3d.core.bbox.iou_calculators.iou3d_calculator' in sys.modules:
        _iou_mod = sys.modules['mmdet3d.core.bbox.iou_calculators.iou3d_calculator']
        _iou_mod.bbox_overlaps_nearest_3d = bbox_overlaps_nearest_3d
        _iou_mod.bbox_overlaps_3d = bbox_overlaps_3d
        _iou_mod.axis_aligned_bbox_overlaps_3d = axis_aligned_bbox_overlaps_3d
    if 'mmdet3d.core.bbox.iou_calculators' in sys.modules:
        _iou_pkg = sys.modules['mmdet3d.core.bbox.iou_calculators']
        _iou_pkg.bbox_overlaps_nearest_3d = bbox_overlaps_nearest_3d
        _iou_pkg.bbox_overlaps_3d = bbox_overlaps_3d
        _iou_pkg.axis_aligned_bbox_overlaps_3d = axis_aligned_bbox_overlaps_3d


# mmcv.utils tuple helper alias ---------------------------------------------
try:
    import mmcv.utils as _mmcv_utils
    from mmengine.utils import to_2tuple
except Exception:
    pass
else:
    _mmcv_utils.to_2tuple = to_2tuple


# mmcv top-level Config aliases ---------------------------------------------
try:
    import mmcv as _mmcv
    from mmengine.config import Config, ConfigDict
except Exception:
    pass
else:
    _mmcv.Config = Config
    _mmcv.ConfigDict = ConfigDict


# pytorch_lightning.metrics.metric legacy alias ------------------------------
try:
    import types
    from torchmetrics import Metric
except Exception:
    pass
else:
    pl_mod = types.ModuleType('pytorch_lightning')
    pl_metrics_mod = types.ModuleType('pytorch_lightning.metrics')
    pl_metric_mod = types.ModuleType('pytorch_lightning.metrics.metric')
    pl_metric_mod.Metric = Metric
    _install_module('pytorch_lightning', pl_mod)
    _install_module('pytorch_lightning.metrics', pl_metrics_mod)
    _install_module('pytorch_lightning.metrics.metric', pl_metric_mod)


# pytorch_lightning old metric functional aliases ----------------------------
try:
    import types
    import torch
except Exception:
    pass
else:
    def stat_scores_multiple_classes(prediction, target, num_classes):
        pred = prediction.reshape(-1).long()
        tgt = target.reshape(-1).long()
        device = pred.device
        tps = torch.zeros(num_classes, device=device, dtype=torch.long)
        fps = torch.zeros(num_classes, device=device, dtype=torch.long)
        tns = torch.zeros(num_classes, device=device, dtype=torch.long)
        fns = torch.zeros(num_classes, device=device, dtype=torch.long)
        sups = torch.zeros(num_classes, device=device, dtype=torch.long)
        for cls in range(num_classes):
            pred_c = pred == cls
            tgt_c = tgt == cls
            tps[cls] = (pred_c & tgt_c).sum()
            fps[cls] = (pred_c & ~tgt_c).sum()
            tns[cls] = (~pred_c & ~tgt_c).sum()
            fns[cls] = (~pred_c & tgt_c).sum()
            sups[cls] = tgt_c.sum()
        return tps, fps, tns, fns, sups

    def reduce(x, reduction='none'):
        if reduction in (None, 'none'):
            return x
        if reduction in ('mean', 'elementwise_mean'):
            return x.mean()
        if reduction == 'sum':
            return x.sum()
        raise ValueError(f'Unsupported reduction: {reduction}')

    pl_functional_mod = types.ModuleType('pytorch_lightning.metrics.functional')
    pl_classification_mod = types.ModuleType('pytorch_lightning.metrics.functional.classification')
    pl_reduction_mod = types.ModuleType('pytorch_lightning.metrics.functional.reduction')
    pl_classification_mod.stat_scores_multiple_classes = stat_scores_multiple_classes
    pl_reduction_mod.reduce = reduce
    _install_module('pytorch_lightning.metrics.functional', pl_functional_mod)
    _install_module('pytorch_lightning.metrics.functional.classification', pl_classification_mod)
    _install_module('pytorch_lightning.metrics.functional.reduction', pl_reduction_mod)


# Additional old mmdet model registry names ----------------------------------
try:
    import mmdet.models as _mmdet_models
except Exception:
    pass
else:
    _mmdet_models.DETECTORS = MMDET3D_MODELS
    _mmdet_models.NECKS = MMDET3D_MODELS
    _mmdet_models.ROI_EXTRACTORS = MMDET3D_MODELS
    _mmdet_models.SHARED_HEADS = MMDET3D_MODELS


# mmdet3d.ops old sparse aliases ---------------------------------------------
try:
    import types
    from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule
except Exception:
    pass
else:
    ops3d_mod = types.ModuleType('mmdet3d.ops')
    ops3d_mod.SparseBasicBlock = SparseBasicBlock
    ops3d_mod.make_sparse_convmodule = make_sparse_convmodule
    _install_module('mmdet3d.ops', ops3d_mod)


# mmdet3d.ops.spconv old module alias ----------------------------------------
try:
    import spconv.pytorch as _spconv_torch
except Exception:
    pass
else:
    if 'mmdet3d.ops' in sys.modules:
        sys.modules['mmdet3d.ops'].spconv = _spconv_torch
    _install_module('mmdet3d.ops.spconv', _spconv_torch)


# Prefer mmdet3d TASK_UTILS for custom 3D bbox coders/assigners --------------
try:
    from mmdet3d.registry import TASK_UTILS as _MMDET3D_TASK_UTILS
except Exception:
    pass
else:
    if 'mmdet.core.bbox.builder' in sys.modules:
        _bb = sys.modules['mmdet.core.bbox.builder']
        _bb.BBOX_ASSIGNERS = _MMDET3D_TASK_UTILS
        _bb.BBOX_SAMPLERS = _MMDET3D_TASK_UTILS
        _bb.BBOX_CODERS = _MMDET3D_TASK_UTILS
        _bb.build_assigner = _MMDET3D_TASK_UTILS.build
        _bb.build_sampler = _MMDET3D_TASK_UTILS.build
        _bb.build_bbox_coder = _MMDET3D_TASK_UTILS.build
    if 'mmdet.core.bbox.coders' in sys.modules:
        sys.modules['mmdet.core.bbox.coders'].build_bbox_coder = _MMDET3D_TASK_UTILS.build
    if 'mmdet3d.core.bbox.coders' in sys.modules:
        sys.modules['mmdet3d.core.bbox.coders'].build_bbox_coder = _MMDET3D_TASK_UTILS.build
    if 'mmdet.core' in sys.modules:
        sys.modules['mmdet.core'].build_sampler = _MMDET3D_TASK_UTILS.build
        sys.modules['mmdet.core'].build_bbox_coder = _MMDET3D_TASK_UTILS.build
        sys.modules['mmdet.core'].build_assigner = _MMDET3D_TASK_UTILS.build


# Smoke-only old DETRHead init-contract compatibility ------------------------
# This is diagnostic only.  It provides the attributes UniMM custom heads expect
# from MMDetection 2.x DETRHead, but it is not a validated training port.
try:
    import torch.nn as nn
    import mmdet.models.dense_heads as _dense_heads
    import mmdet.models.dense_heads.detr_head as _detr_head_mod
    from mmengine.model import BaseModule
    from mmcv.cnn.bricks.transformer import build_positional_encoding
except Exception:
    pass
else:
    class _LegacyDETRHeadCompat(BaseModule):
        def __init__(self,
                     num_classes,
                     in_channels=None,
                     num_query=100,
                     num_reg_fcs=2,
                     transformer=None,
                     sync_cls_avg_factor=False,
                     positional_encoding=None,
                     loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                     loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                     loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                     train_cfg=None,
                     test_cfg=None,
                     init_cfg=None,
                     **kwargs):
            super().__init__(init_cfg=init_cfg)
            self.num_classes = num_classes
            self.in_channels = in_channels
            self.num_query = num_query
            self.num_reg_fcs = num_reg_fcs
            self.train_cfg = train_cfg
            self.test_cfg = test_cfg
            self.fp16_enabled = False
            self.sync_cls_avg_factor = sync_cls_avg_factor
            self.bg_cls_weight = 0

            def _build_model_compat(cfg):
                try:
                    return MMDET3D_MODELS.build(cfg)
                except KeyError:
                    return MMDET_MODELS.build(cfg)
            self.loss_cls = _build_model_compat(loss_cls)
            self.loss_bbox = _build_model_compat(loss_bbox)
            self.loss_iou = _build_model_compat(loss_iou)
            self.cls_out_channels = num_classes
            if getattr(self.loss_cls, 'use_sigmoid', False):
                self.cls_out_channels = num_classes
            else:
                self.cls_out_channels = num_classes + 1
            class_weight = getattr(self.loss_cls, 'class_weight', None)
            if class_weight is not None:
                try:
                    self.bg_cls_weight = class_weight[-1]
                except Exception:
                    pass

            self.positional_encoding = _build_model_compat(positional_encoding) if positional_encoding is not None else None
            self.transformer = build_transformer(transformer) if transformer is not None else None
            self.embed_dims = getattr(self.transformer, 'embed_dims', None) or kwargs.get('embed_dims', 256)

            if train_cfg is not None:
                assigner_cfg = train_cfg.get('assigner') if hasattr(train_cfg, 'get') else None
                if assigner_cfg is not None:
                    self.assigner = _MMDET3D_TASK_UTILS.build(assigner_cfg)
                    self.sampler = _MMDET3D_TASK_UTILS.build(dict(type='PseudoSampler'))
            self._init_layers()

        def _init_layers(self):
            pass

        def init_weights(self):
            if hasattr(self.transformer, 'init_weights'):
                self.transformer.init_weights()

    _dense_heads.DETRHead = _LegacyDETRHeadCompat
    _detr_head_mod.DETRHead = _LegacyDETRHeadCompat


# Expose mmdet 2D modules through mmdet3d MODELS for MVX builders ------------
try:
    import mmdet.models  # ensure registration
    from mmdet.registry import MODELS as _MMDET_MODELS_REAL
    from mmdet3d.registry import MODELS as _MMDET3D_MODELS_REAL
except Exception:
    pass
else:
    for _name in ['ResNet', 'FPN', 'FocalLoss', 'L1Loss', 'GIoULoss', 'CrossEntropyLoss', 'LearnedPositionalEncoding']:
        if _name in _MMDET_MODELS_REAL.module_dict and _name not in _MMDET3D_MODELS_REAL.module_dict:
            _MMDET3D_MODELS_REAL.register_module(name=_name, module=_MMDET_MODELS_REAL.module_dict[_name], force=True)


# Task util fallback: custom 3D first, upstream mmdet second -----------------
try:
    from mmdet.registry import TASK_UTILS as _MMDET_TASK_UTILS_REAL
    from mmdet3d.registry import TASK_UTILS as _MMDET3D_TASK_UTILS_REAL
except Exception:
    pass
else:
    def _convert_legacy_hungarian_assigner_cfg(cfg):
        if isinstance(cfg, dict) and cfg.get('type') == 'HungarianAssigner':
            legacy_keys = ['cls_cost', 'reg_cost', 'iou_cost']
            if any(k in cfg for k in legacy_keys) and 'match_costs' not in cfg:
                new_cfg = cfg.copy()
                match_costs = []
                for key in legacy_keys:
                    if key in new_cfg:
                        match_costs.append(new_cfg.pop(key))
                new_cfg['match_costs'] = match_costs
                return new_cfg
        return cfg

    def _registry_build_with_default_args(registry, cfg, *args, **kwargs):
        default_args = kwargs.pop('default_args', None)
        if kwargs:
            merged_default_args = dict(default_args or {})
            merged_default_args.update(kwargs)
            default_args = merged_default_args
        if default_args is not None:
            return registry.build(cfg, *args, default_args=default_args)
        return registry.build(cfg, *args)

    def _build_task_util_compat(cfg, *args, **kwargs):
        try:
            return _registry_build_with_default_args(_MMDET3D_TASK_UTILS_REAL, cfg, *args, **kwargs)
        except KeyError:
            return _registry_build_with_default_args(_MMDET_TASK_UTILS_REAL, _convert_legacy_hungarian_assigner_cfg(cfg), *args, **kwargs)
        except TypeError:
            if isinstance(cfg, dict) and cfg.get('type') == 'HungarianAssigner':
                return _registry_build_with_default_args(_MMDET_TASK_UTILS_REAL, _convert_legacy_hungarian_assigner_cfg(cfg), *args, **kwargs)
            raise

    for _mod_name in ['mmdet.core', 'mmdet.core.bbox.builder', 'mmdet.core.bbox.coders', 'mmdet3d.core.bbox.coders']:
        _mod = sys.modules.get(_mod_name)
        if _mod is not None:
            if hasattr(_mod, 'build_assigner'):
                _mod.build_assigner = _build_task_util_compat
            if hasattr(_mod, 'build_sampler'):
                _mod.build_sampler = _build_task_util_compat
            if hasattr(_mod, 'build_bbox_coder'):
                _mod.build_bbox_coder = _build_task_util_compat
    if 'mmdet.core.bbox.match_costs' in sys.modules:
        sys.modules['mmdet.core.bbox.match_costs'].build_match_cost = _build_task_util_compat


# Positional encoding registry bridge for mmcv transformer builders -----------
try:
    import mmdet.models  # ensure registration
    import mmcv.cnn.bricks.transformer as _mmcv_transformer
    from mmdet.registry import MODELS as _MMDET_MODELS_FOR_POS
except Exception:
    pass
else:
    for _name in ['SinePositionalEncoding', 'LearnedPositionalEncoding']:
        if _name in _MMDET_MODELS_FOR_POS.module_dict:
            _mmcv_transformer.MODELS.register_module(
                name=_name,
                module=_MMDET_MODELS_FOR_POS.module_dict[_name],
                force=True,
            )


# Old mmdet DETR transformer names used by UniMM configs --------------------
try:
    import torch as _torch
    from mmcv.cnn.bricks.transformer import (
        BaseTransformerLayer as _BaseTransformerLayer,
        TransformerLayerSequence as _TransformerLayerSequence,
        MODELS as _TRANSFORMER_SEQUENCE_MODELS,
    )
except Exception:
    pass
else:
    def _inverse_sigmoid_compat(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return _torch.log(x1 / x2)

    class _DetrTransformerEncoderCompat(_TransformerLayerSequence):
        pass

    class _DetrTransformerDecoderLayerCompat(_BaseTransformerLayer):
        pass

    class _DetrTransformerDecoderCompat(_TransformerLayerSequence):
        def __init__(self, *args, return_intermediate=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.return_intermediate = return_intermediate

        def forward(self, query, *args, **kwargs):
            output = query
            intermediate = []
            for layer in self.layers:
                output = layer(output, *args, **kwargs)
                if self.return_intermediate:
                    intermediate.append(output)
            if self.return_intermediate:
                return _torch.stack(intermediate)
            return output

    class _DeformableDetrTransformerDecoderCompat(_TransformerLayerSequence):
        def __init__(self, *args, return_intermediate=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.return_intermediate = return_intermediate

        def forward(self,
                    query,
                    *args,
                    reference_points=None,
                    reg_branches=None,
                    key_padding_mask=None,
                    **kwargs):
            output = query
            intermediate = []
            intermediate_reference_points = []
            valid_ratios = kwargs.get('valid_ratios')

            for lid, layer in enumerate(self.layers):
                if reference_points is None:
                    reference_points_input = None
                elif reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] * _torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
                else:
                    reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

                output = layer(
                    output,
                    *args,
                    reference_points=reference_points_input,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                output = output.permute(1, 0, 2)

                if reg_branches is not None:
                    tmp = reg_branches[lid](output)
                    if reference_points.shape[-1] == 4:
                        new_reference_points = tmp + _inverse_sigmoid_compat(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        new_reference_points = tmp
                        new_reference_points[..., :2] = (
                            tmp[..., :2] + _inverse_sigmoid_compat(reference_points))
                        new_reference_points = new_reference_points[..., :2].sigmoid()
                    reference_points = new_reference_points.detach()

                output = output.permute(1, 0, 2)
                if self.return_intermediate:
                    intermediate.append(output)
                    intermediate_reference_points.append(reference_points)

            if self.return_intermediate:
                return _torch.stack(intermediate), _torch.stack(intermediate_reference_points)
            return output, reference_points

    for _name, _module in {
        'DetrTransformerEncoder': _DetrTransformerEncoderCompat,
        'DetrTransformerDecoder': _DetrTransformerDecoderCompat,
        'DetrTransformerDecoderLayer': _DetrTransformerDecoderLayerCompat,
        'DeformableDetrTransformerDecoder': _DeformableDetrTransformerDecoderCompat,
        'DeformableDetrTransformerDecoderLayer': _DetrTransformerDecoderLayerCompat,
    }.items():
        _TRANSFORMER_SEQUENCE_MODELS.register_module(
            name=_name,
            module=_module,
            force=True,
        )
