import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.device_count())  # 显卡数量
print(torch.__version__)          # 应 ≥ 2.6
print(torch.cuda.is_available())  # GPU 用户应为 True