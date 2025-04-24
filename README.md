# Knowledge Distillation on CIFAR-10

This is an implementation of the paper "Distilling the Knowledge in a Neural Network" by Hinton et al(https://arxiv.org/abs/1503.02531).. The implementation performs knowledge distillation on the CIFAR-10 dataset using PyTorch.

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Implementation Details

The implementation consists of:
- A teacher network (larger model with ResNet-like architecture)
- A student network (smaller model)
- Knowledge distillation training process
- Baseline training for comparison

## How to Run

1. First, train the teacher network:
```bash
python main.py --train-teacher --epochs 200
```

2. Then, train the student network with knowledge distillation:
```bash
python main.py --epochs 200 --temperature 4.0 --alpha 0.1
```

This will:
1. Load the pre-trained teacher model
2. Train the student model with knowledge distillation
3. Train a baseline student model without knowledge distillation for comparison

## Arguments

- `--batch-size`: Batch size for training (default: 128)
- `--epochs`: Number of epochs to train (default: 200)
- `--num-workers`: Number of workers for data loading (default: 2)
- `--temperature`: Temperature parameter for knowledge distillation (default: 4.0)
- `--alpha`: Weight for soft targets in knowledge distillation loss (default: 0.1)
- `--train-teacher`: Flag to train the teacher network

## Implementation Notes

1. The teacher network is a larger model with ResNet-like architecture
2. The student network is a smaller CNN with fewer parameters
3. Knowledge distillation uses both hard targets (ground truth) and soft targets (teacher's softened outputs)
4. The distillation loss is a weighted combination of:
   - Cross-entropy with hard targets
   - KL divergence with soft targets
5. Temperature parameter controls how much to soften the teacher's output distribution
6. Alpha parameter controls the weight between hard and soft targets in the loss function

## Results

Training Parameters:
- Temperature: 4.0
- Alpha: 0.1
- Epochs: 200
- Batch Size: 128

Final Test Accuracies:
- Teacher Network: 0.9313
- Student Network (with KD): 0.9022
- Student Network (Baseline): 0.8997

Improvement with KD: 0.25%
