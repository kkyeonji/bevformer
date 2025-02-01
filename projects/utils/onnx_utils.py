import torch.onnx
from torch import nn
import onnx
import onnxruntime
import numpy as np


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

def get_tensor_input(inputs, tensor_inputs = {}, tensor_inputs_ort_form = {}, name = []):
    # TODO : reference
    it = inputs.item() if isinstance(inputs, dict) else enumerate(inputs)
    for k1, d1 in it:
        if isinstance(d1, torch.Tensor):
            name.append(str(k1))
            tensor_inputs_ort_form['.'.join(name)] = d1
            add_value(tensor_inputs, name, d1)
            name.pop()
        elif isinstance(d1, dict): # or list?
            name.append(k1)
            get_tensor_input(d1, tensor_inputs, tensor_inputs_ort_form, name)
        # TODO : else
    if len(name) != 0:
        name.pop()

    return tensor_inputs, tensor_inputs_ort_form

def onnx_export(model, dataloader, onnx_file_path, logger=None):
    data = dataloader.collate_fn(dataset[demo_idx])
    tensor_input = get_tensor_input(data)

    torch.onnx.export(
        model,
        args = (tensor_input, {}),
        f = onnx_file_path,
        input_names = list(tensor_input.keys()),
        opset_version = 16,
        verbose = True,
    )

    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model (onnx_model)

    return