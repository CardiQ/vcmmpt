import torch

from mmpt.models import MMPTModel


model, tokenizer, aligner = MMPTModel.from_pretrained(
    "D:/mycodepy/videoCLIP/fairseq/examples/MMPT/projects/retri/videoclip/how2.yaml")

# train_on_gpu = torch.cuda.is_available()
# print(torch.__version__)
# if not train_on_gpu:
#     print('CUDA is not available.')
# else:
#     print('CUDA is available!')
#
#
# # 启用GPU-model
# gpuok = torch.cuda.is_available()
# if gpuok:
#     model.cuda()

model.eval()

# B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
video_frames = torch.randn(1, 2, 30, 224, 224, 3)
caps, cmasks = aligner._build_text_seq(
    tokenizer("some text", add_special_tokens=False)["input_ids"]
)

caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

with torch.no_grad():
    output = model(video_frames, caps, cmasks, return_score=True)
print(output["score"])  # dot-product