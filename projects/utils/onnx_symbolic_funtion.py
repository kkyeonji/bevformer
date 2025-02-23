import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_registry as sym_registry

def atan2_symbolic_override(g, input1, input2):
    return g.op("Atan", g.op("Div", input1, input2))

def all_symbolic_override(g, input, dim, keepdim):
    axes = sym_help._maybe_get_const(dim, "i")
    keepdim = sym_help._maybe_get_const(keepdim, "i")
    return g.op("ReduceMin", input, axes_i=[axes], keepdims_i=keepdim)

def and_symbolic_override(g, self, other):
    return g.op("And", self, other)

def register_custom_symbolic_functions(opset_version):
    sym_registry.register_op("atan2", atan2_symbolic_override, "", opset_version)
    sym_registry.register_op("all", all_symbolic_override, "", opset_version)
    sym_registry.register_op("__iand_", and_symbolic_override, "", opset_version)


