import torch.onnx
from torch import nn
import onnx
import onnxruntime
import numpy as np
import mmcv
import os
from pathlib import Path
import copy

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
    meta_data[0][0].pop('box_type_3d', None)
    with torch.no_grad():
        onnx_wrapper = OnnxWrapper(model, meta_data)
        output = onnx_wrapper(copy.deepcopy(input))

        torch.onnx.export(
            onnx_wrapper,
            args = (input, {}),
            f = onnx_save_pth,
            input_names = list(ort_inputs.keys()),
            # opset_version = 16,
            verbose = True,
        )
    print("Onnx Export Suceed")

    onnx_model = onnx.load(onnx_save_pth)
    onnx.checker.check_model (onnx_model)

    return

