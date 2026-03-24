# Session 5: MLOps and Model Deployment with NVIDIA Tools

## Learning Objectives

By the end of this session you will be able to:

- Describe the MLOps lifecycle and where NVIDIA tools fit in
- Deploy a model with NVIDIA Triton Inference Server
- Monitor GPU metrics in production
- Containerise an NVIDIA AI workload with Docker and NGC
- Apply basic security and versioning best practices for deployed models

---

## 1. The MLOps Lifecycle

```
  Data Collection & Preparation
           │
  Model Training & Experimentation
           │
  Model Evaluation & Validation
           │
  Model Packaging & Versioning
           │
  Deployment & Serving
           │
  Monitoring & Feedback Loop
           │
  (back to Training)
```

NVIDIA tools accelerate multiple stages of this loop:

| Stage | NVIDIA Tool |
|-------|------------|
| Training | DGX systems, NeMo, cuML |
| Optimisation | TensorRT, TensorRT-LLM |
| Serving | Triton Inference Server |
| Monitoring | Prometheus + DCGM Exporter |
| Containerisation | NGC containers |

---

## 2. NVIDIA Triton Inference Server

Triton is an open-source inference serving solution that supports multiple model frameworks (TensorRT, ONNX Runtime, PyTorch, TensorFlow, Python backends) through a unified HTTP/gRPC API.

### Key Features

- **Multi-framework**: Serve TensorRT, ONNX, PyTorch, TensorFlow, and custom Python models from the same server.
- **Dynamic batching**: Automatically groups individual requests into batches for higher GPU utilisation.
- **Concurrent model execution**: Multiple model instances run in parallel on the same GPU.
- **Model pipeline (ensemble)**: Chain models together as a DAG.
- **Health & metrics endpoints**: Built-in Prometheus metrics.

### Model Repository Layout

```
model_repository/
├── resnet50/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan          # TensorRT engine
└── bert_base/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

### Example `config.pbtxt`

```protobuf
name: "resnet50"
platform: "tensorrt_plan"
max_batch_size: 32

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [1000]
  }
]

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 1000
}
```

### Starting Triton

```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### Sending a Request

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = httpclient.InferInput("input", [1, 3, 224, 224], "FP32")
inputs.set_data_from_numpy(np.random.rand(1, 3, 224, 224).astype(np.float32))

outputs = httpclient.InferRequestedOutput("output")
response = client.infer("resnet50", [inputs], outputs=[outputs])
result = response.as_numpy("output")
print(result.shape)  # (1, 1000)
```

---

## 3. GPU Monitoring in Production

### DCGM (Data Centre GPU Manager)

DCGM provides health monitoring, diagnostics, and policy management for NVIDIA GPUs in cluster environments.

```bash
# Install DCGM exporter (Prometheus format)
docker run -d --gpus all \
  -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04 \
  -f /etc/dcgm-exporter/dcp-metrics-included.csv

# Query metrics
curl http://localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL
```

### Key Metrics to Track

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `GPU_UTIL` | SM utilisation | > 80% during inference |
| `FB_USED` | Framebuffer (GPU memory) used | < 90% of total |
| `POWER_USAGE` | GPU power draw | Within TDP |
| `GPU_TEMP` | GPU temperature | < 85 °C |
| `NVLINK_BANDWIDTH` | NVLink data rate | Scales with batch size |

---

## 4. Containerisation Best Practices

Use NGC base containers to guarantee a reproducible, optimised environment:

```dockerfile
# Example Dockerfile for a PyTorch inference service
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ ./model/
COPY serve.py .

EXPOSE 8080
CMD ["python", "serve.py"]
```

### Best Practices

1. **Pin the NGC container tag** (e.g., `24.01-py3`) to ensure reproducibility.
2. **Do not include training code or raw datasets** in production images.
3. **Scan images** with `docker scout` or `grype` before deployment.
4. **Use multi-stage builds** to keep the final image lean.

---

## 5. Model Versioning and Experiment Tracking

| Tool | Purpose |
|------|---------|
| MLflow | Experiment tracking, model registry, serving |
| Weights & Biases (W&B) | Experiment tracking, hyperparameter sweeps |
| DVC | Data and model versioning with Git |
| NVIDIA NeMo | End-to-end framework for LLM and speech; built-in experiment management |

### Minimal MLflow Example

```python
import mlflow
import mlflow.pytorch

with mlflow.start_run():
    mlflow.log_param("learning_rate", 1e-3)
    mlflow.log_param("batch_size", 128)
    # ... training loop ...
    mlflow.log_metric("val_accuracy", 0.934)
    mlflow.pytorch.log_model(model, "model")
```

---

## 6. Hands-On Exercise

1. Pull the Triton Inference Server container.
2. Build a TensorRT engine for ResNet-50 (from Session 4).
3. Create a `model_repository` with the engine and a `config.pbtxt`.
4. Start Triton and use the Python client to send an inference request.
5. Check the Triton metrics endpoint: `curl http://localhost:8002/metrics`.

---

## Key Takeaways

- Triton Inference Server provides a production-grade, multi-framework serving solution with dynamic batching and built-in metrics.
- DCGM and Prometheus give you deep visibility into GPU health and utilisation in production clusters.
- Containerising workloads with pinned NGC base images ensures reproducible, optimised deployments.
- Experiment tracking and model versioning are essential for maintaining a healthy MLOps practice.

---

## Additional Resources

- [Triton Inference Server GitHub](https://github.com/triton-inference-server/server)
- [DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [NVIDIA Kubernetes Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/)

---

**Previous:** [Session 4 – NVIDIA AI Frameworks](session_04_nvidia_ai_frameworks.md)  
**Back to start:** [README](../README.md)
