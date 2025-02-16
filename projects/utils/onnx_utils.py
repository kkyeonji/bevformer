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


class TestWrapper(nn.Module):
    def __init__(self, model, batch):
        super().__init__()
        model.eval()
        model.cuda()
        self.model = model
        self.metadata = {}

        org_batch = copy.deepcopy(batch)
        for k, v in org_batch.items():
            if not isinstance(v, torch.Tensor):
                batch.pop(k)
                self.metadata[k] = v

    def forward(self, batch):
        # inter_states, inter_references = self.model.pts_bbox_head.transformer.decoder(
        # **batch,
        # **self.metadata)
        query =  batch['query']
        reference_points = batch['reference_points']
        reg_branches = self.metadata['reg_branches']
        key_padding_mask = None

        args = ()
        kwargs = {
            'spatial_shapes': batch['spatial_shapes'],
            'value': batch['value'],
            'level_start_index': batch['level_start_index'],
            'query_pos': batch['query_pos'],
            'img_metas': self.metadata['img_metas'],
            'cls_branches': self.metadata['cls_branches'],
            'key': self.metadata['key'],
        }

        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.model.pts_bbox_head.transformer.decoder.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            # output = output.permute(1, 0, 2)

        #     if reg_branches is not None:
        #         tmp = reg_branches[lid](output)

        #         assert reference_points.shape[-1] == 3

        #         new_reference_points = torch.zeros_like(reference_points)
        #         new_reference_points[..., :2] = tmp[
        #             ..., :2] + inverse_sigmoid(reference_points[..., :2])
        #         new_reference_points[..., 2:3] = tmp[
        #             ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

        #         new_reference_points = new_reference_points.sigmoid()

        #         reference_points = new_reference_points.detach()

            # output = output.permute(1, 0, 2)
        #     if self.model.pts_bbox_head.transformer.decoder.return_intermediate:
        #         intermediate.append(output)
        #         intermediate_reference_points.append(reference_points)

        # if self.model.pts_bbox_head.transformer.decoder.return_intermediate:
        #     return torch.stack(intermediate), torch.stack(
        #         intermediate_reference_points)

        return output, reference_points

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
    batch = load_pkl_data('./debug/decode/')
    batch = load_gpu(batch)

    # export onnx for specified part of model to debug
    with torch.no_grad():
        onnx_wrapper = TestWrapper(model, batch)
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


