# Session 1: Introduction to NVIDIA AI & GPU Computing

## Learning Objectives

By the end of this session you will be able to:

- Describe NVIDIA's role in the AI ecosystem
- Explain the difference between CPU and GPU computing
- Identify key NVIDIA hardware platforms used for AI workloads
- Navigate the NVIDIA software stack at a high level

---

## 1. Why NVIDIA for AI?

NVIDIA GPUs are the dominant platform for AI training and inference because:

- **Massive parallelism** – A modern GPU contains thousands of cores designed to execute many operations simultaneously, which maps perfectly to the matrix and tensor operations that underlie deep learning.
- **High memory bandwidth** – GPUs move data between memory and compute units far faster than CPUs.
- **Mature software ecosystem** – CUDA, cuDNN, TensorRT, and the broader NVIDIA stack have been optimised over many years for AI workloads.

---

## 2. CPU vs GPU Computing

| Feature | CPU | GPU |
|---------|-----|-----|
| Core count | Tens (high-end) | Thousands |
| Optimised for | Sequential, branch-heavy tasks | Parallel, data-heavy tasks |
| Memory bandwidth | ~100 GB/s | ~1–3 TB/s (A100/H100) |
| Use case | Orchestration, preprocessing | Training, inference, simulation |

Deep learning training involves billions of floating-point multiply-add operations arranged in regular patterns. GPUs excel at exactly this kind of work.

---

## 3. Key NVIDIA Hardware Platforms

### Data-Centre GPUs

| GPU | Architecture | Use Case |
|-----|-------------|----------|
| H100 | Hopper | Large-scale LLM training, HPC |
| A100 | Ampere | General AI training & inference |
| L40S | Ada Lovelace | Inference, visualisation |

### Workstation / Edge

| Platform | Description |
|----------|-------------|
| RTX 4090 | High-end workstation GPU for research & development |
| Jetson Orin | Edge AI platform for robotics and IoT |

### DGX Systems

NVIDIA DGX systems are purpose-built AI servers that bundle multiple high-end GPUs, NVLink high-speed GPU interconnect, and pre-installed NVIDIA AI software into a single appliance.

---

## 4. The NVIDIA Software Stack

```
┌────────────────────────────────────────────────────────┐
│          Applications & Frameworks                     │
│   PyTorch  |  TensorFlow  |  JAX  |  cuML  |  ...     │
├────────────────────────────────────────────────────────┤
│          NVIDIA Optimisation Libraries                 │
│   cuDNN  |  TensorRT  |  cuBLAS  |  NCCL  |  RAPIDS   │
├────────────────────────────────────────────────────────┤
│                     CUDA Runtime                       │
├────────────────────────────────────────────────────────┤
│                  NVIDIA GPU Driver                     │
├────────────────────────────────────────────────────────┤
│                  NVIDIA GPU Hardware                   │
└────────────────────────────────────────────────────────┘
```

- **CUDA** – The foundational programming model for NVIDIA GPUs.
- **cuDNN** – GPU-accelerated primitives for deep neural networks (convolutions, activations, pooling, etc.).
- **TensorRT** – A high-performance inference SDK for deploying trained models.
- **NCCL** – Collective communications library for multi-GPU and multi-node training.
- **RAPIDS** – GPU-accelerated data science (pandas, scikit-learn style APIs on the GPU).

---

## 5. NVIDIA NGC – The AI Model Hub

[NVIDIA NGC](https://ngc.nvidia.com/) provides:

- Pre-built Docker containers optimised for NVIDIA GPUs
- Pre-trained models and model scripts
- SDKs and HPC applications

Using NGC containers is the recommended way to get started quickly without worrying about dependency management.

---

## Hands-On Exercise

1. Run `nvidia-smi` to inspect the GPU(s) on your machine. Note the GPU model, driver version, CUDA version, and current memory usage.
2. Pull and run the latest NGC PyTorch container:
   ```bash
   docker pull nvcr.io/nvidia/pytorch:24.01-py3
   docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:24.01-py3 python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```
3. Confirm that PyTorch can see the GPU and print the device name.

---

## Key Takeaways

- NVIDIA GPUs accelerate AI workloads through massive parallelism and high memory bandwidth.
- The NVIDIA software stack (CUDA → cuDNN → frameworks) provides a complete, optimised environment for AI.
- NGC containers are the fastest way to get a reproducible, GPU-ready environment.

---

## Additional Resources

- [NVIDIA GPU Architecture Overview](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA NGC Getting Started](https://docs.ngc.nvidia.com/catalog/getting-started)

---

**Next:** [Session 2 – Deep Learning Fundamentals](session_02_deep_learning_fundamentals.md)
