_base_ = [
    '../bevformer/bevformer_small.py',
]

opsest_version = 13
onnx_file_dir = "./onnx_output"
seed = 0
checkpoint = "/home/ubuntu/workspace/bevformer/ckpts/bevformer_small_epoch_24.pth"
fuse_conv_bn = False
show = True
show_dir = "./onnx_output"
device = 'cuda'