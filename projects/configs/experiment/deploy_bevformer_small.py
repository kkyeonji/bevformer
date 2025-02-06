_base_ = [
    '../bevformer/bevformer_small.py',
]

onnx_file_path = "./onnx_output"
seed = 0
checkpoint = "/home/ubuntu/workspace/BEVFormer/ckpts/r101_dcn_fcos3d_pretrain.pth"
fuse_conv_bn = False
show = True
show_dir = "./onnx_output"
device = 'cuda'