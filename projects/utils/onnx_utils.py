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
import pickle
from projects.mmdet3d_plugin.bevformer.modules.decoder import inverse_sigmoid
from mmdet3d.core import bbox3d2result
from projects.utils.onnx_symbolic_funtion import register_custom_symbolic_functions


class OnnxWrapper(nn.Module):
        def __init__(self, model, meta_data):
            super().__init__()
            model.eval()
            model.cuda()
            self.model = model

            self.meta_data = meta_data

        def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
            """Test function"""
            outs = self.model.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

            bbox_list = self.model.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
            # bbox_results = [
            #     bbox3d2result(bboxes, scores, labels)
            #     for bboxes, scores, labels in bbox_list
            # ]
            return outs, bbox_list # outs['bev_embed'], bbox_results
        
        def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
            """Test function without augmentaiton."""
            img_feats = self.model.extract_feat(img=img, img_metas=img_metas)

            bbox_list = [dict() for i in range(len(img_metas))]
            new_prev_bev, bbox_pts = self.model.simple_test_pts(
                img_feats, img_metas, prev_bev, rescale=rescale)
            # for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            #     result_dict['pts_bbox'] = pts_bbox
            return new_prev_bev, bbox_pts
            
        def forward_test(self, img_metas, img=None, **kwargs):
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

            new_prev_bev, bbox_results = self.model.simple_test(
                img_metas[0], img[0], prev_bev=self.model.prev_frame_info['prev_bev'], **kwargs)
            # During inference, we save the BEV features and ego motion of each timestamp.
            self.model.prev_frame_info['prev_pos'] = tmp_pos
            self.model.prev_frame_info['prev_angle'] = tmp_angle
            self.model.prev_frame_info['prev_bev'] = new_prev_bev
            return bbox_results

        def forward(self, input):
            self.model.simple_test_pts = self.simple_test_pts
            self.model.simple_test = self.simple_test
            self.model.forward_test = self.forward_test
            forward_input = {'img': input,'img_metas': self.meta_data}
            return self.model.forward(return_loss=False, **forward_input)

        # def forward(self, input):
        #     forward_input = {'img': input,'img_metas': self.meta_data}
        #     return self.model.forward(return_loss=False, **forward_input)
        
        # def post_processing(onnx_output):
        #     # Modify data format from onnx from to torch form
        #     torch_output = {}
        #     # TODO
        #     return torch_output
        

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
    # else:
    #     AssertionError() # TODO
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
    # else:
    #     AssertionError() # TODO
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
        elif isinstance(d1, dict) or isinstance(d1, list):
            name.append(k1)
            get_ort_inputs(d1, ort_inputs, name)
        elif isinstance(d1, mmcv.parallel.data_container.DataContainer):
            get_ort_inputs(d1.data, ort_inputs, name)
        # else: ignore the other dtypes

    if len(name) != 0:
        name.pop()

    return ort_inputs

def save_pkl_data(save_dir, data_dict):
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        save_dir.mkdir()

    for k, v in data_dict.items():
        if isinstance(v, dict):
            save_pkl_data(save_dir, v)
        else:
            save_pth = save_dir / f"{k}.pickle"
            if isinstance(v, torch.Tensor):
                # v.requires_grad = False
                v = v.detach()
            elif isinstance(v, nn.ModuleList):
                for p in v.parameters():
                    p.requires_grad = False
            with open(save_pth, 'wb') as f:
                pickle.dump(v, f)
    return

def load_pkl_data(save_pth):
    data_dict = {}
    save_pth = Path(save_pth)
    data_pth_list = list(save_pth.glob('**/*.pickle'))
    for data_pth in data_pth_list:
        with open(data_pth, 'rb') as f:
            data_dict[data_pth.stem] = pickle.load(f)

    return data_dict

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

    register_custom_symbolic_functions(13)

    with torch.no_grad():
        onnx_wrapper = OnnxWrapper(model, meta_data)
        # if you want to re-use input, you have to copy it.
        # it modified while forward()
        # output = onnx_wrapper(copy.deepcopy(input))

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


class TestWrapper(nn.Module):
    def __init__(self, model, metadata, batch):
        super().__init__()
        model.eval()
        model.cuda()
        self.model = model
        self.meta_data = metadata
        # self.meta_data = {'img_metas': metadata}

        # org_batch = copy.deepcopy(batch)
        # for k, v in org_batch.items():
        #     if not isinstance(v, torch.Tensor):
        #         batch.pop(k)
        #         self.meta_data[k] = v
    
    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        # bbox_list = self.pts_bbox_head.get_bboxes(
        #     outs, img_metas, rescale=rescale)
        # bbox_results = [
        #     bbox3d2result(bboxes, scores, labels)
        #     for bboxes, scores, labels in bbox_list
        # ]
        return outs, None #outs['bev_embed'], bbox_results
    
    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        # for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        #     result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
    
    def forward_test(self, img_metas, img=None, **kwargs):
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

        new_prev_bev, bbox_results = self.model.simple_test(
            img_metas[0], img[0], prev_bev=self.model.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.model.prev_frame_info['prev_pos'] = tmp_pos
        self.model.prev_frame_info['prev_angle'] = tmp_angle
        self.model.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def forward(self, batch):
        self.model.simple_test_pts = self.simple_test_pts
        self.model.simple_test = self.simple_test
        self.model.forward_test = self.forward_test
        forward_input = {'img': batch,'img_metas': self.meta_data}
        return self.model.forward(return_loss=False, **forward_input)

def test_export(model, dataloader, save_dir, chk_pth, device, logger=None):
    # set path to save test.onnx
    save_dir = Path(save_dir)
    chk_pth = Path(chk_pth)
    if not save_dir.is_dir():
        save_dir.mkdir()
    onnx_save_pth = save_dir / chk_pth.stem

    # save numpy data used while debugging
    batch = next(iter(dataloader))
    batch = load_gpu(batch)
    input = get_continer_data(batch['img'])
    meta_data = get_continer_data(batch['img_metas'])

    with torch.no_grad():
        onnx_wrapper = OnnxWrapper(model, meta_data)
        # if you want to re-use input, you have to copy it.
        # it modified while forward()
        _ = onnx_wrapper(copy.deepcopy(input))

    # load numpy data
    batch = load_pkl_data('./debug/decoder/')
    batch = load_gpu(batch)

    # export onnx for specified part of model to debug
    with torch.no_grad():
        onnx_wrapper = TestWrapper(model, meta_data, batch)
        batch = to_tensor(batch)

        # test forward
        onnx_output = onnx_wrapper(batch)

        torch.onnx.export(
            onnx_wrapper,
            args = (batch, {}),
            f = onnx_save_pth,
            opset_version = 13,
            verbose = True,
        )
    print("Onnx Export Suceed")

    onnx_model = onnx.load(onnx_save_pth)
    onnx.checker.check_model (onnx_model)

    return


