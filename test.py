import time
import torch

print("=" * 60)
print("PyTorch CUDA Test")
print("=" * 60)

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("CUDA version    :", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用")

device = torch.device("cuda:0")

print("GPU name        :", torch.cuda.get_device_name(0))
print("GPU count       :", torch.cuda.device_count())

properties = torch.cuda.get_device_properties(0)

print(f"GPU memory      : {properties.total_memory / 1024**3:.2f} GB")

print("\n开始 GPU 矩阵计算测试...")

# 创建两个矩阵，直接放到 RTX 4060
a = torch.randn(5000, 5000, device=device)
b = torch.randn(5000, 5000, device=device)

# CUDA 是异步执行的，所以计时前同步
torch.cuda.synchronize()

start = time.time()

for _ in range(10):
    c = torch.mm(a, b)

torch.cuda.synchronize()

elapsed = time.time() - start

print("\n计算完成")
print(f"矩阵设备       : {c.device}")
print(f"计算耗时       : {elapsed:.3f} 秒")
print(
    f"已分配显存     : "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)
print(
    f"峰值显存       : "
    f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
)

print("\n✅ RTX 4060 CUDA 测试成功")