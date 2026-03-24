# Session 2: Deep Learning Fundamentals

## Learning Objectives

By the end of this session you will be able to:

- Explain the building blocks of neural networks
- Describe how a model learns through forward pass, loss, and backpropagation
- Implement a simple neural network in PyTorch on an NVIDIA GPU
- Recognise common model architectures and when to use them

---

## 1. What Is Deep Learning?

Deep learning is a subset of machine learning that uses multi-layered artificial neural networks to learn representations of data. It has driven breakthroughs in:

- Computer vision (image classification, object detection, segmentation)
- Natural language processing (translation, summarisation, question answering)
- Generative AI (image synthesis, large language models)
- Robotics and autonomous systems

---

## 2. Building Blocks of Neural Networks

### Neuron / Perceptron

A neuron computes a weighted sum of its inputs and applies a non-linear **activation function**:

```
output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

Where `w` are learnable weights, `x` are inputs, and `b` is a bias term.

### Layers

| Layer Type | Description |
|------------|-------------|
| Linear / Dense | Fully connected; every input connects to every output |
| Convolutional (Conv2D) | Applies learnable filters across spatial dimensions; used in vision |
| Recurrent (RNN/LSTM/GRU) | Processes sequential data with hidden state |
| Attention / Transformer | Computes pairwise relationships; backbone of modern LLMs |
| Normalization (BatchNorm, LayerNorm) | Stabilises training by normalising activations |
| Dropout | Randomly zeroes activations during training to reduce overfitting |

### Activation Functions

| Activation | Formula | Typical Use |
|-----------|---------|-------------|
| ReLU | `max(0, x)` | Hidden layers (default choice) |
| Sigmoid | `1 / (1 + e⁻ˣ)` | Binary classification output |
| Softmax | `eˣⁱ / Σeˣʲ` | Multi-class classification output |
| GELU | `x · Φ(x)` | Transformer models |

---

## 3. The Training Loop

```
for each mini-batch:
    1. Forward pass  → compute predictions
    2. Compute loss  → measure prediction error
    3. Backward pass → compute gradients (backpropagation)
    4. Optimiser step → update weights
```

### Loss Functions

| Task | Common Loss |
|------|------------|
| Regression | Mean Squared Error (MSE) |
| Binary classification | Binary Cross-Entropy |
| Multi-class classification | Cross-Entropy |

### Optimisers

| Optimiser | Notes |
|----------|-------|
| SGD | Simple, with optional momentum |
| Adam | Adaptive learning rates; most popular default |
| AdamW | Adam + weight decay; preferred for transformers |

---

## 4. Key Concepts

### Overfitting vs Underfitting

- **Underfitting** – Model is too simple; high training and validation loss.
- **Overfitting** – Model memorises training data; low training loss but high validation loss.
- **Remedies:** More data, regularisation (dropout, weight decay), early stopping, data augmentation.

### Learning Rate

The learning rate controls how large each weight update is. Common strategies:

- **Learning rate schedule** – Reduce LR on plateau or follow a cosine annealing schedule.
- **Warm-up** – Start with a small LR and increase it at the beginning of training.

### Batch Size

Larger batches make better use of GPU parallelism but may generalise less well. Typical ranges: 32–512 for vision tasks, smaller for LLMs due to memory constraints.

---

## 5. Common Architectures

| Architecture | Domain | Key Paper / Year |
|-------------|--------|-----------------|
| ResNet | Image classification | He et al., 2015 |
| VGG | Image classification | Simonyan et al., 2014 |
| YOLO | Object detection | Redmon et al., 2015 |
| UNet | Image segmentation | Ronneberger et al., 2015 |
| Transformer | NLP / general | Vaswani et al., 2017 |
| BERT | NLP understanding | Devlin et al., 2018 |
| GPT-style | Text generation | Radford et al., 2018+ |
| ViT | Vision transformer | Dosovitskiy et al., 2020 |

---

## 6. Hands-On Exercise: Training a Simple Classifier on GPU

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# Model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
).to(device)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
```

Run the script and observe:

1. GPU utilisation via `watch -n 1 nvidia-smi`
2. Loss decreasing across epochs

---

## Key Takeaways

- Neural networks learn by iteratively adjusting weights to minimise a loss function.
- The training loop (forward → loss → backward → optimise) is universal across frameworks.
- Moving tensors and models to the GPU (`.to(device)`) is all you need to leverage NVIDIA acceleration in PyTorch.

---

## Additional Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book (Goodfellow et al.) – free online](https://www.deeplearningbook.org/)
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

---

**Previous:** [Session 1 – Introduction to NVIDIA AI & GPU Computing](session_01_intro_to_nvidia_ai.md)  
**Next:** [Session 3 – CUDA Programming Basics](session_03_cuda_programming_basics.md)
