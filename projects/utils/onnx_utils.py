import torch.onnx
from torch import nn
import onnx
import onnxruntime
import numpy as np
import mmcv
import os
from pathlib import Path
import copy
import math
from torchvision.transforms.functional import rotate


class OnnxWrapper(nn.Module):
        def __init__(self, model, meta_data):
            super().__init__()
            model.eval()
            model.cuda()
            self.model = model

            self.meta_data = meta_data

        def forward(self, input):
            forward_input = {'img': input,'img_metas': self.meta_data}
            return self.model.forward(return_loss=False, **forward_input)
        
        def post_processing(onnx_output):
            # Modify data format from onnx from to torch form
            torch_output = {}
            # TODO
            return torch_output
        

def to_numpy(x):
    # Transfer tensor to numpy.
    # If list/dict of tensor included, iteratively transfer elements to numpy.
    if isinstance(x, torch.Tensor):
        if x.requires_grad:
            x = x.detach()
        x = x.cpu().numpy()
    elif isinstance(x, dict):
        for k in x:
            x[k] = to_numpy(x[k])
    elif isinstance(x, list):
        for e in x:
            e = to_numpy(e)
    else:
        AssertionError() # TODO
    return x

def to_tensor(x):
    # Transfer numpy to tensor.
    # If list/dict of numpy included, iteratively transfer elements to tensor.
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        x = x.cuda()
    elif isinstance(x, dict):
        for k in x:
            x[k] = to_tensor(x[k])
    elif isinstance(x, list):
        for e in x:
            e = to_tensor(e)
    else:
        AssertionError() # TODO
    return x

# deprecated  
def add_value(dict_out, key_list, value):
    """
        Add value to the dict with key_list.
        The index of key_list become depth of dict.
        e.x) key_list = ['a', 'b', 'c'], value = 3
            org_dict = {'a' : {'b' : {'c' : 3}}}   
    """
    key = key_list[0]
    if len(key_list) == 1:
        dict_out[key] = value
        return dict_out
    
    if key in dict_out.keys():
        add_value(dict_out[key], key_list[1:], value)
    else:
        dict_out[key] = {}
        add_value(dict_out[key], key_list[1:], value)

# deprecated
def get_tensor_input(inputs, tensor_inputs = {}, non_tensor_inputs = {},
                    tensor_inputs_ort_form = {}, name = [], device = 'cuda'):
    # TODO : reference
    it = inputs.items() if isinstance(inputs, dict) else enumerate(inputs)
    for k1, d1 in it:
        if isinstance(d1, torch.Tensor):
            name.append(k1)
            data_key = '.'.join(str(n) for n in name)
            tensor_inputs_ort_form[data_key] = d1
            add_value(tensor_inputs, name, d1.to(device))
            name.pop()
        elif isinstance(d1, dict) or isinstance(d1, list):
            name.append(k1)
            get_tensor_input(d1, tensor_inputs, non_tensor_inputs,
                            tensor_inputs_ort_form, name, device)

        elif isinstance(d1, mmcv.parallel.data_container.DataContainer):
            get_tensor_input(d1.data, tensor_inputs, non_tensor_inputs,
                            tensor_inputs_ort_form, name, device)
        else:
            name.append(k1)
            add_value(non_tensor_inputs, name, d1)
            name.pop()

    if len(name) != 0:
        name.pop()

    return tensor_inputs, non_tensor_inputs, tensor_inputs_ort_form

def get_continer_data(data):
    it = data.items() if isinstance(data, dict) else enumerate(data)
    for k1, d1 in it:
        if isinstance(d1, mmcv.parallel.data_container.DataContainer):
            return d1.data
        else:
            get_continer_data(data[k1])

def load_gpu(data):
    it = data.items() if isinstance(data, dict) else enumerate(data)
    for k1, d1 in it:
        if isinstance(d1, dict) or isinstance(d1, list):
            load_gpu(data[k1])
        elif isinstance(d1, torch.Tensor):
            data[k1] = d1.to('cuda')
        elif isinstance(d1, mmcv.parallel.data_container.DataContainer):
            load_gpu(data[k1].data)
        # else: ignore other dtypes

    return data

def get_ort_inputs(inputs, ort_inputs = {}, name = []):
    # TODO : reference
    it = inputs.items() if isinstance(inputs, dict) else enumerate(inputs)
    for k1, d1 in it:
        if isinstance(d1, torch.Tensor):
            name.append(k1)
            data_key = '.'.join(str(n) for n in name)
            ort_inputs[data_key] = d1
            name.pop()
        # elif isinstance(d1, np.ndarry):
        #     name.append(k1)
        #     data_key = '.'.join(str(n) for n in name)
        #     ort_inputs[data_key] = to_tensor(d1)
        #     name.pop()
        elif isinstance(d1, dict) or isinstance(d1, list):
            name.append(k1)
            get_ort_inputs(d1, ort_inputs, name)
        elif isinstance(d1, mmcv.parallel.data_container.DataContainer):
            get_ort_inputs(d1.data, ort_inputs, name)
        # else: ignore the other dtypes

    if len(name) != 0:
        name.pop()

    return ort_inputs

def onnx_export(model, dataloader, save_dir, chk_pth, device, logger=None):
    save_dir = Path(save_dir)
    chk_pth = Path(chk_pth)
    if not save_dir.is_dir():
        save_dir.mkdir()
    onnx_save_pth = save_dir / chk_pth.stem

    batch = next(iter(dataloader))
    batch = load_gpu(batch)
    ort_inputs = get_ort_inputs(batch)
    # data['img'][0].data[0].shape = [1, 6, 3, 736, 1280]
    # data['img_metas'][0].data[0][0].keys() =
    #  ['filename', 'ori_shape', 'img_shape', 'lidar2img', 'lidar2cam',
    #  'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
    #  'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
    #  'sample_idx', 'prev_idx', 'next_idx', 'pcd_scale_factor', 'pts_filename',
    #  'scene_token', 'can_bus']
    input = get_continer_data(batch['img'])
    meta_data = get_continer_data(batch['img_metas'])
    with torch.no_grad():
        onnx_wrapper = OnnxWrapper(model, meta_data)
        # if you want to re-use input, you have to copy it.
        # it modified while forward()
        output = onnx_wrapper(copy.deepcopy(input))

        torch.onnx.export(
            onnx_wrapper,
            args = (input, {}),
            f = onnx_save_pth,
            input_names = list(ort_inputs.keys()),
            opset_version = 13,
            verbose = True,
        )
    print("Onnx Export Suceed")

    onnx_model = onnx.load(onnx_save_pth)
    onnx.checker.check_model (onnx_model)

    return

from mmdet3d.core import bbox3d2result

class TestWrapper(nn.Module):
    def __init__(self, model, meta_data):
        super().__init__()
        model.eval()
        model.cuda()
        self.model = model
        self.head = self.model.pts_bbox_head
        self.transformer = self.model.pts_bbox_head.transformer

        self.meta_data = meta_data

    def preprocess(self, img=None):
        img_metas = self.meta_data
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.model.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.model.prev_frame_info['prev_bev'] = None
        # update idx
        self.model.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.model.video_test_mode:
            self.model.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.model.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.model.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.model.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        img_feats = self.model.extract_feat(img=img[0], img_metas=img_metas[0])

        bs, num_cam, _, _, _ = img_feats[0].shape
        dtype = img_feats[0].dtype
        object_query_embeds = self.head.query_embedding.weight.to(dtype)
        bev_queries = self.head.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.head.bev_h, self.head.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.head.positional_encoding(bev_mask).to(dtype)

        # get_bev_features
        bev_h = self.head.bev_h
        bev_w = self.head.bev_w
        grid_length=(self.head.real_h / self.head.bev_h,
                        self.head.real_w / self.head.bev_w)
        prev_bev = self.model.prev_frame_info['prev_bev']
        img_metas = self.meta_data

        bs = img_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        
        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in img_metas[0]])
        delta_y = np.array([each['can_bus'][1]
                           for each in img_metas[0]])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in img_metas[0]])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / math.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * math.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * math.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.transformer.use_shift
        shift_x = shift_x * self.transformer.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.transformer.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = img_metas[0][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.transformer.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in img_metas[0]])  # [:, :]
        can_bus = self.transformer.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.transformer.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(img_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.transformer.use_cams_embeds:
                feat = feat + self.transformer.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.transformer.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        return feat_flatten, bev_queries, spatial_shapes, level_start_index, bev_pos

    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        
        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / math.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / math.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * math.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * math.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.transformer.use_shift
        shift_x = shift_x * self.transformer.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.transformer.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.transformer.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.transformer.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.transformer.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.transformer.use_cams_embeds:
                feat = feat + self.transformer.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.transformer.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.transformer.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )

        return bev_embed
    

    def forward(self, img, feat_flatten, bev_queries, spatial_shapes, level_start_index, bev_pos):
        bev_h = self.head.bev_h
        bev_w = self.head.bev_w
        bev_embed = self.transformer.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev = self.model.prev_frame_info['prev_bev'],
            shift=shift,
            **kwargs
        )
        return bev_embed

def test_export(model, dataloader, save_dir, chk_pth, device, logger=None):
    save_dir = Path(save_dir)
    chk_pth = Path(chk_pth)
    if not save_dir.is_dir():
        save_dir.mkdir()
    onnx_save_pth = save_dir / chk_pth.stem

    batch = next(iter(dataloader))
    batch = load_gpu(batch)
    ort_inputs = get_ort_inputs(batch)
    
    input = get_continer_data(batch['img'])
    meta_data = get_continer_data(batch['img_metas']) 
    with torch.no_grad():
        onnx_wrapper = TestWrapper(model, meta_data)
        img_feats, bev_queries, bev_pos = onnx_wrapper.preprocess(copy.deepcopy(input))

        torch.onnx.export(
            onnx_wrapper,
            args = (input, img_feats, bev_queries, bev_pos, {}),
            f = onnx_save_pth,
            # input_names = list(ort_inputs.keys()),
            # opset_version = 16,
            verbose = True,
        )
    print("Onnx Export Suceed")

    onnx_model = onnx.load(onnx_save_pth)
    onnx.checker.check_model (onnx_model)

    return


