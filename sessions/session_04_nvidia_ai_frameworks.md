# Session 4: NVIDIA AI Frameworks

## Learning Objectives

By the end of this session you will be able to:

- Describe the purpose and capabilities of cuDNN, TensorRT, RAPIDS, and NCCL
- Use TensorRT to optimise and deploy a trained model for low-latency inference
- Accelerate a pandas/scikit-learn workflow with RAPIDS cuDF and cuML
- Set up multi-GPU training with NCCL through PyTorch DDP

---

## 1. Overview of NVIDIA AI Libraries

| Library | Purpose | Layer in Stack |
|---------|---------|---------------|
| **cuBLAS** | GPU-accelerated BLAS (dense linear algebra) | Primitive |
| **cuDNN** | Deep neural network primitives (convolution, attention, etc.) | Primitive |
| **NCCL** | Multi-GPU collective communications | Primitive |
| **TensorRT** | High-performance inference engine | Deployment |
| **RAPIDS** | GPU-accelerated data science (cuDF, cuML, cuGraph) | Data Science |
| **Triton Inference Server** | Production model serving | Deployment |

---

## 2. cuDNN

cuDNN provides highly optimised implementations of common DNN operations. Deep learning frameworks such as PyTorch and TensorFlow call cuDNN automatically, so most users never write cuDNN code directly.

Key operations:
- Convolution (forward, backward data, backward filter)
- Batch normalisation
- Pooling
- Activation functions (ReLU, sigmoid, tanh)
- Attention (multi-head self-attention for Transformer models)

### Checking the cuDNN Version

```python
import torch
print(torch.backends.cudnn.version())  # e.g. 8906
```

### Enabling cuDNN Benchmark Mode

```python
import torch
torch.backends.cudnn.benchmark = True  # auto-tune conv algorithms for fixed input sizes
```

---

## 3. TensorRT

TensorRT is NVIDIA's SDK for high-performance inference. It takes a trained model, applies a suite of optimisations, and produces a highly efficient inference engine.

### Optimisation Techniques

| Technique | Description |
|-----------|-------------|
| Layer fusion | Merges adjacent layers (e.g., conv + BN + ReLU → single kernel) |
| Precision calibration | Reduces weights/activations from FP32 → FP16 or INT8 |
| Kernel auto-tuning | Selects the fastest CUDA kernel for each op on the target GPU |
| Dynamic shapes | Handles variable-length inputs via optimisation profiles |

### Workflow

```
Trained Model (ONNX / TF / PyTorch)
           │
   TensorRT Builder
           │
     Serialised Engine (.trt / .plan)
           │
   TensorRT Runtime (inference)
```

### Example: Export PyTorch → ONNX → TensorRT

```python
import torch
import torch.onnx

# 1. Export to ONNX
model = torch.hub.load("pytorch/vision", "resnet50", pretrained=True).eval().cuda()
dummy = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(model, dummy, "resnet50.onnx", opset_version=17, input_names=["input"], output_names=["output"])
```

```bash
# 2. Build TensorRT engine with FP16 precision
trtexec --onnx=resnet50.onnx --saveEngine=resnet50_fp16.trt --fp16

# 3. Run benchmark
trtexec --loadEngine=resnet50_fp16.trt --batch=32
```

---

## 4. RAPIDS – GPU-Accelerated Data Science

RAPIDS brings the familiar pandas/scikit-learn API to the GPU, enabling data science pipelines to run orders of magnitude faster.

### Key Libraries

| Library | CPU Equivalent | Description |
|---------|---------------|-------------|
| cuDF | pandas | GPU DataFrame operations |
| cuML | scikit-learn | GPU ML algorithms (linear models, clustering, random forests, etc.) |
| cuGraph | NetworkX | GPU graph analytics |
| cuSpatial | GeoPandas | GPU geospatial operations |

### Example: cuDF vs pandas

```python
import cudf
import pandas as pd

# Read a 1 GB CSV
df_gpu = cudf.read_csv("large_dataset.csv")   # runs on GPU
df_cpu = pd.read_csv("large_dataset.csv")     # runs on CPU

# GroupBy aggregation
result_gpu = df_gpu.groupby("category")["value"].mean()
result_cpu = df_cpu.groupby("category")["value"].mean()
```

### Example: cuML for Machine Learning

```python
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score
import cudf

X = cudf.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0]})
y = cudf.Series([0, 0, 1, 1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test))}")
```

---

## 5. NCCL and Multi-GPU Training

**NCCL** (NVIDIA Collective Communications Library) provides optimised primitives for multi-GPU communication (AllReduce, Broadcast, Scatter, etc.). PyTorch's `DistributedDataParallel` uses NCCL automatically.

### PyTorch DDP (DistributedDataParallel)

```python
# train_ddp.py
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    for _ in range(100):
        outputs = ddp_model(torch.randn(20, 10).to(rank))
        loss = loss_fn(outputs, torch.randn(20, 10).to(rank))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
```

```bash
# Launch on 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py
```

---

## Hands-On Exercise

1. **TensorRT**: Export the ResNet-50 model to ONNX and build an FP16 engine with `trtexec`. Compare throughput (images/sec) between the FP32 PyTorch baseline and the FP16 TensorRT engine.
2. **RAPIDS**: Load a CSV with at least 1 million rows using both `pandas` and `cudf`. Measure the time for a groupby aggregation with `%%timeit` in a Jupyter notebook.

---

## Key Takeaways

- cuDNN is the GPU primitive library that powers all major deep learning frameworks.
- TensorRT dramatically improves inference throughput and latency through layer fusion, precision reduction, and kernel tuning.
- RAPIDS replaces CPU-based data science libraries with GPU-native equivalents, using the same familiar API.
- NCCL enables efficient multi-GPU training; PyTorch DDP handles the boilerplate.

---

## Additional Resources

- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [RAPIDS Docs](https://docs.rapids.ai/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [PyTorch Distributed Training Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

---

**Previous:** [Session 3 – CUDA Programming Basics](session_03_cuda_programming_basics.md)  
**Next:** [Session 5 – MLOps and Model Deployment](session_05_mlops_and_deployment.md)
