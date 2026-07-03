import numpy as np
import mmcv
import hashlib
from mmdet.datasets.builder import PIPELINES
from einops import rearrange
from mmdet3d.datasets.pipelines import LoadAnnotations3D
from mmdet3d.datasets.pipelines import LoadPointsFromFile
from mmdet3d.core.points import BasePoints, get_points_type
import os


@PIPELINES.register_module()
class LoadPointsFromFile_E2E(LoadPointsFromFile):
    def __init__(self, coord_type, load_dim=6, use_dim=..., shift_height=False, use_color=False, file_client_args=dict(backend='disk'),pts_root=''):

        super().__init__(coord_type, load_dim, use_dim, shift_height, use_color, file_client_args)

        self.pts_root = pts_root

    @staticmethod
    def _stable_rng(stable_seed, tag):
        key = '{}|{}'.format(stable_seed, tag)
        digest = hashlib.sha256(key.encode('utf-8')).digest()
        seed = int.from_bytes(digest[:4], byteorder='little', signed=False)
        return np.random.RandomState(seed)

    def _apply_single_physical_shift(self, points, results, shift):
        name = shift.get('name', '')
        if name not in ['lidar_sparsity', 'point_dropout']:
            return points
        keep_ratio = float(np.clip(shift.get('keep_ratio', 1.0), 0.0, 1.0))
        if keep_ratio >= 1.0 or points.shape[0] == 0:
            return points
        stable_seed = int(shift.get('stable_seed', shift.get('seed', 0)))
        rng = self._stable_rng(stable_seed, name)
        keep_mask = rng.rand(points.shape[0]) < keep_ratio
        if not keep_mask.any():
            keep_mask[int(rng.randint(0, points.shape[0]))] = True
        shifted = points[keep_mask]
        results.setdefault('physical_shift', {}).update(
            dict(applied_to='points', point_keep_ratio=keep_ratio,
                 points_before=int(points.shape[0]), points_after=int(shifted.shape[0])))
        return shifted

    def _apply_physical_shift(self, points, results):
        shift = results.get('physical_shift_config', None)
        if not shift:
            return points
        if shift.get('name', '') in [
                'compound', 'missing_modality', 'modality_dropout',
                'unseen_sensor_setup']:
            shifted = points
            for subshift in shift.get('shifts', []):
                shifted = self._apply_single_physical_shift(shifted, results, subshift)
            return shifted
        return self._apply_single_physical_shift(points, results, shift)

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """       
        pts_filename = os.path.join(self.pts_root,results['pts_filename'])
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        points = self._apply_physical_shift(points, results)
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

@PIPELINES.register_module()
class LoadInfTrackQueryFile(object):
    """Load inf track query file.

    Args:
        inf_track_query_file (str): inf track query file path. 
            Defaults to ''.  file: {'sample_token_inf': Instances}

    """    
    def __init__(self, inf_track_query_file=''):
        self.inf_track_query_file = inf_track_query_file
        
        if os.path.exists(self.inf_track_query_file):
            self.inf_track_query_infos = mmcv.load(self.inf_track_query_file)

    def __call__(self, results):
        if self.inf_track_query_infos:
            sample_token_inf = results['sample_idx_inf']
            results['inf_track_query'] = self.inf_track_query_infos[sample_token_inf]

        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(inf_track_query_file={self.inf_track_query_file}, '
        return repr_str        
   

@PIPELINES.register_module()
class LoadMultiViewImageFromFilesInCeph(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged', file_client_args=dict(backend='disk'), img_root=''):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = mmcv.FileClient(**self.file_client_args)
        self.img_root = img_root

    @staticmethod
    def _stable_rng(stable_seed, tag, view_idx):
        key = '{}|{}|{}'.format(stable_seed, tag, view_idx)
        digest = hashlib.sha256(key.encode('utf-8')).digest()
        seed = int.from_bytes(digest[:4], byteorder='little', signed=False)
        return np.random.RandomState(seed)

    @staticmethod
    def _apply_fov_mask(img, keep_ratio, axis='horizontal'):
        keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
        if keep_ratio >= 1.0:
            return img
        shifted = img.copy()
        h, w = shifted.shape[:2]
        if axis == 'vertical':
            keep_h = max(1, int(round(h * keep_ratio)))
            top = max(0, (h - keep_h) // 2)
            bottom = min(h, top + keep_h)
            shifted[:top, ...] = 0
            shifted[bottom:, ...] = 0
        else:
            keep_w = max(1, int(round(w * keep_ratio)))
            left = max(0, (w - keep_w) // 2)
            right = min(w, left + keep_w)
            shifted[:, :left, ...] = 0
            shifted[:, right:, ...] = 0
        return shifted

    def _apply_single_physical_shift(self, images, results, shift):
        name = shift.get('name', '')
        stable_seed = int(shift.get('stable_seed', shift.get('seed', 0)))
        if name == 'fov_mask':
            keep_ratio = shift.get('keep_ratio', 1.0)
            axis = shift.get('mask_axis', 'horizontal')
            shifted = [
                self._apply_fov_mask(img, keep_ratio, axis=axis)
                for img in images
            ]
            results.setdefault('physical_shift', {}).update(
                dict(applied_to='image', num_views=len(shifted)))
            return shifted

        if name in ['missing_camera', 'camera_dropout']:
            p = float(np.clip(shift.get('drop_probability', 0.0), 0.0, 1.0))
            ensure_one_view = bool(shift.get('ensure_one_view', False))
            drop_views = []
            shifted = []
            for view_idx, img in enumerate(images):
                rng = self._stable_rng(stable_seed, name, view_idx)
                should_drop = bool(rng.rand() < p)
                if should_drop:
                    drop_views.append(view_idx)
                    shifted.append(np.zeros_like(img))
                else:
                    shifted.append(img)
            if ensure_one_view and len(drop_views) == len(images) and images:
                rng = self._stable_rng(stable_seed, '{}_keep_one'.format(name), 0)
                keep_idx = int(rng.randint(0, len(images)))
                shifted[keep_idx] = images[keep_idx]
                drop_views.remove(keep_idx)
            results.setdefault('physical_shift', {}).update(
                dict(applied_to='image', dropped_views=drop_views,
                     num_views=len(shifted)))
            return shifted

        return images

    def _apply_physical_shift(self, images, results):
        shift = results.get('physical_shift_config', None)
        if not shift:
            return images

        if shift.get('name', '') in [
                'compound', 'missing_modality', 'modality_dropout',
                'unseen_sensor_setup']:
            shifted = images
            applied = []
            for subshift in shift.get('shifts', []):
                before = shifted
                shifted = self._apply_single_physical_shift(shifted, results, subshift)
                if shifted is not before:
                    applied.append(subshift.get('name', ''))
            results.setdefault('physical_shift', {}).update(
                dict(applied_to='image', image_shift_sequence=applied,
                     num_views=len(shifted)))
            return shifted

        return self._apply_single_physical_shift(images, results, shift)

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (list of str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        images_multiView = []
        filename = results['img_filename']
        for img_path in filename:
            img_path = os.path.join(self.img_root, img_path)
            if self.file_client_args['backend'] == 'petrel':
                img_bytes = self.file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes)
            elif self.file_client_args['backend'] == 'disk':
                img = mmcv.imread(img_path, self.color_type)
            images_multiView.append(img)
        images_multiView = self._apply_physical_shift(images_multiView, results)
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            #[mmcv.imread(name, self.color_type) for name in filename], axis=-1)
            images_multiView, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3D_E2E(LoadAnnotations3D):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """
    def __init__(self,
                 with_future_anns=False,
                 with_ins_inds_3d=False,
                 ins_inds_add_1=False,  # NOTE: make ins_inds start from 1, not 0
                 **kwargs):
        super().__init__(**kwargs)
        self.with_future_anns = with_future_anns
        self.with_ins_inds_3d = with_ins_inds_3d

        self.ins_inds_add_1 = ins_inds_add_1
    
    def _load_future_anns(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        gt_bboxes_3d = []
        gt_labels_3d = []
        gt_inds_3d = []
        # gt_valid_flags = []
        gt_vis_tokens  = []

        for ann_info in results['occ_future_ann_infos']:
            if ann_info is not None:
                gt_bboxes_3d.append(ann_info['gt_bboxes_3d'])
                gt_labels_3d.append(ann_info['gt_labels_3d'])
                
                ann_gt_inds = ann_info['gt_inds']
                if self.ins_inds_add_1:
                    ann_gt_inds += 1
                    # NOTE: sdc query is changed from -10 -> -9
                gt_inds_3d.append(ann_gt_inds)

                # gt_valid_flags.append(ann_info['gt_valid_flag'])
                gt_vis_tokens.append(ann_info['gt_vis_tokens'])
            else:
                # invalid frame
                gt_bboxes_3d.append(None)
                gt_labels_3d.append(None)
                gt_inds_3d.append(None)
                # gt_valid_flags.append(None)
                gt_vis_tokens.append(None)

        results['future_gt_bboxes_3d'] = gt_bboxes_3d
        # results['future_bbox3d_fields'].append('gt_bboxes_3d')  # Field is used for augmentations, not needed here
        results['future_gt_labels_3d'] = gt_labels_3d
        results['future_gt_inds'] = gt_inds_3d
        # results['future_gt_valid_flag'] = gt_valid_flags
        results['future_gt_vis_tokens'] = gt_vis_tokens

        return results 
  
    def _load_ins_inds_3d(self, results):
        ann_gt_inds = results['ann_info']['gt_inds'].copy() # TODO: note here

        # NOTE: Avoid gt_inds generated twice
        results['ann_info'].pop('gt_inds')
        
        if self.ins_inds_add_1:
            ann_gt_inds += 1
        results['gt_inds'] = ann_gt_inds
        return results

    def __call__(self, results):
        results = super().__call__(results)
        
        if self.with_future_anns:
            results = self._load_future_anns(results)
        if self.with_ins_inds_3d:
            results = self._load_ins_inds_3d(results)
        
        # Generate ann for plan
        if 'occ_future_ann_infos_for_plan' in results.keys():
            results = self._load_future_anns_plan(results)
        
        return results

    def __repr__(self):
        repr_str = super().__repr__()
        indent_str = '    '
        repr_str += f'{indent_str}with_future_anns={self.with_future_anns}, '
        repr_str += f'{indent_str}with_ins_inds_3d={self.with_ins_inds_3d}, '
        
        return repr_str
