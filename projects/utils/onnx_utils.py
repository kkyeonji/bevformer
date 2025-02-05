import torch.onnx
from torch import nn
import onnx
import onnxruntime
import numpy as np
import mmcv


class OnnxWrapper(nn.Module):
        def __init__(self, model, non_tensor_inputs):
            super().__init__()
            model.eval()
            model.cuda()
            self.model = model

            # TODO : load non tenosrs as class var
            self.non_tensor_inputs = non_tensor_inputs
            # for key, val in non_tensor_inputs.items():
            #     setattr(self, f'key', val)

        def forward(self, tensor_inputs):
            tensor_inputs.update(self.non_tensor_inputs)
            return self.model.forward(return_loss=False, **tensor_inputs)
        
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

def get_tensor_input(inputs, tensor_inputs = {}, non_tensor_inputs = {},
                    tensor_inputs_ort_form = {}, name = []):
    # TODO : reference
    it = inputs.items() if isinstance(inputs, dict) else enumerate(inputs)
    for k1, d1 in it:
        if isinstance(d1, torch.Tensor):
            name.append(str(k1))
            tensor_inputs_ort_form['.'.join(name)] = d1
            add_value(tensor_inputs, name, d1)
            name.pop()
        elif isinstance(d1, dict) or isinstance(d1, list):
            name.append(k1)
            get_tensor_input(d1, tensor_inputs, non_tensor_inputs,
                            tensor_inputs_ort_form, name)
        elif isinstance(d1, mmcv.parallel.data_container.DataContainer):
            get_tensor_input(d1.data, tensor_inputs, non_tensor_inputs,
                            tensor_inputs_ort_form, name)
        else:
            name.append(str(k1))
            add_value(non_tensor_inputs, name, d1)
            name.pop()

    if len(name) != 0:
        name.pop()

    return tensor_inputs, non_tensor_inputs, tensor_inputs_ort_form

def onnx_export(model, dataloader, onnx_file_path, logger=None):
    data = next(iter(dataloader))
    tensor_inputs, non_tensor_inputs, _ = get_tensor_input(data)
    
    onnx_wrapper = OnnxWrapper(model, non_tensor_inputs)

    torch.onnx.export(
        onnx_wrapper,
        args = (tensor_inputs, {}),
        f = onnx_file_path,
        input_names = list(tensor_inputs.keys()),
        opset_version = 16,
        verbose = True,
    )

    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model (onnx_model)

    return

