# Knowledge Distillation on CIFAR-10
This repository contains an implementation of **Knowledge Distillation** as introduced in the paper [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) by Hinton et al. The models are trained and evaluated on the CIFAR-10 dataset using PyTorch.
---
##  Requirements
Install all required packages using:
```bash
pip install -r requirements.txt
```
---
##  Overview
This project includes:
- **Teacher Network**: A larger ResNet-like model.
- **Student Network**: A smaller, lightweight CNN.
- **Knowledge Distillation Training**: Student learns from both true labels and softened teacher outputs.
- **Baseline Training**: For comparison, the student is also trained without knowledge distillation.
---
##  How to Run
### 1. Train the Teacher Network
```bash
python main.py --train-teacher --epochs 200
```
### 2. Train the Student Network with Knowledge Distillation
```bash
python main.py --epochs 200 --temperature 4.0 --alpha 0.1
```
During this process:
- The pre-trained teacher model is loaded automatically.
- The student model is trained using knowledge distillation.
- Additionally, a baseline student model is trained without knowledge distillation for comparison.
---
##  How to Use the Model
After training, you can load and evaluate a saved model like this:
```python
import torch
from models import TeacherNet, StudentNet
from utils import get_cifar10_loaders

# Load the trained models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load teacher model
teacher = TeacherNet().to(device)
teacher.load_state_dict(torch.load('best_teacher.pth'))
teacher.eval()

# Load student model (with knowledge distillation)
student_kd = StudentNet().to(device)
student_kd.load_state_dict(torch.load('best_student_kd.pth'))
student_kd.eval()

# Load baseline student model
student_base = StudentNet().to(device)
student_base.load_state_dict(torch.load('best_student_base.pth'))
student_base.eval()

# Get data loader
_, testloader = get_cifar10_loaders(batch_size=128)

# Example inference
def predict(model, image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted

# Example batch inference
def evaluate_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Get accuracy on test set
teacher_acc = evaluate_model(teacher, testloader)
student_kd_acc = evaluate_model(student_kd, testloader)
student_base_acc = evaluate_model(student_base, testloader)

print(f"Teacher Accuracy: {teacher_acc:.2f}%")
print(f"Student (KD) Accuracy: {student_kd_acc:.2f}%")
print(f"Student (Base) Accuracy: {student_base_acc:.2f}%")
```
---
##  Training Arguments
| Argument | Description |
|:---|:---|
| `--batch-size` | Batch size for training |
| `--epochs` | Number of training epochs |
| `--num-workers` | Number of worker threads for data loading |
| `--temperature` | Softening parameter for teacher outputs |
| `--alpha` | Weight between hard and soft targets in the loss |
| `--train-teacher` | Flag to train the teacher network |
---
##  Implementation Details
- **Teacher Model**: A deep, ResNet-inspired architecture.
- **Student Model**: A compact CNN with significantly fewer parameters.
- **Loss Function**:
  - Cross-entropy with ground truth labels (hard targets).
  - KL divergence with softened teacher outputs (soft targets).
  - Final loss is a weighted sum controlled by `alpha`.
- **Temperature**: Adjusts the "softness" of the teacher’s output probability distribution.
- **Training Workflow**:
  - Train teacher model.
  - Use trained teacher to guide student model training.
  - Train a baseline student without distillation for comparison.
---
##  Results
**Training Setup**:
- Temperature (`T`): 4.0
- Alpha (`α`): 0.1
- Epochs: 200
- Batch Size: 128

**Final Test Accuracies**:
| Model | Accuracy |
|:---|:---|
| Teacher Network | 93.13% |
| Student Network (with KD) | 90.22% |
| Student Network (Baseline) | 89.97% |

**Knowledge Distillation Improvement**: **+0.25%** over baseline.
