import torch
import torch.nn as nn
from models import TeacherNet, StudentNet
from train import train_teacher, train_student_kd, evaluate
from utils import get_cifar10_loaders
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def plot_training_curves(teacher_history, student_kd_history, student_base_history, save_path='results'):
    """Plot training curves for all models"""
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy curves
    plt.subplot(2, 1, 1)
    plt.plot(teacher_history['test_acc'], label='Teacher', color='blue')
    plt.plot(student_kd_history['test_acc'], label='Student (KD)', color='green')
    plt.plot(student_base_history['test_acc'], label='Student (Base)', color='red')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss curves
    plt.subplot(2, 1, 2)
    plt.plot(teacher_history['train_loss'], label='Teacher', color='blue')
    plt.plot(student_kd_history['train_loss'], label='Student (KD)', color='green')
    plt.plot(student_base_history['train_loss'], label='Student (Base)', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'))
    plt.close()

def print_results(teacher_acc, student_kd_acc, student_base_acc, args):
    """Print detailed results of the training"""
    print("\n" + "="*50)
    print("KNOWLEDGE DISTILLATION RESULTS")
    print("="*50)
    print(f"Training Parameters:")
    print(f"- Temperature: {args.temperature}")
    print(f"- Alpha: {args.alpha}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch Size: {args.batch_size}")
    print("\nFinal Test Accuracies:")
    print(f"Teacher Network: {teacher_acc:.4f}")
    print(f"Student Network (with KD): {student_kd_acc:.4f}")
    print(f"Student Network (baseline): {student_base_acc:.4f}")
    print(f"\nImprovement with KD: {(student_kd_acc - student_base_acc)*100:.2f}%")
    print("="*50)

def main(args):
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get data loaders
    trainloader, testloader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Initialize history dictionaries
    teacher_history = {'test_acc': [], 'train_loss': []}
    student_kd_history = {'test_acc': [], 'train_loss': []}
    student_base_history = {'test_acc': [], 'train_loss': []}

    # Initialize teacher model
    teacher = TeacherNet().to(device)
    best_teacher_acc = 0.0

    # Train or load teacher model
    if args.train_teacher:
        print("Training teacher network...")
        best_teacher_acc = train_teacher(
            teacher,
            trainloader,
            testloader,
            device,
            epochs=args.epochs,
            history=teacher_history
        )
        print(f'Best teacher accuracy: {best_teacher_acc:.4f}')
    else:
        try:
            teacher.load_state_dict(torch.load('best_teacher.pth'))
            print("Loaded pre-trained teacher model")
            # Evaluate the loaded teacher model
            best_teacher_acc = evaluate(teacher, testloader, device)
            print(f'Loaded teacher accuracy: {best_teacher_acc:.4f}')
        except (FileNotFoundError, RuntimeError) as e:
            print("No pre-trained teacher model found or model architecture mismatch.")
            print("Please train the teacher model first with --train-teacher flag.")
            return

    print("Training student network with knowledge distillation...")
    student = StudentNet().to(device)
    best_student_acc = train_student_kd(
        student,
        teacher,
        trainloader,
        testloader,
        device,
        temperature=args.temperature,
        alpha=args.alpha,
        epochs=args.epochs,
        history=student_kd_history
    )
    print(f'Best student accuracy: {best_student_acc:.4f}')

    # Train student network without knowledge distillation for comparison
    print("Training student network without knowledge distillation...")
    student_baseline = StudentNet().to(device)
    best_baseline_acc = train_teacher(
        student_baseline,
        trainloader,
        testloader,
        device,
        epochs=args.epochs,
        history=student_base_history
    )
    print(f'Best baseline student accuracy: {best_baseline_acc:.4f}')

    # Plot training curves
    plot_training_curves(teacher_history, student_kd_history, student_base_history, results_dir)
    
    # Print detailed results
    print_results(best_teacher_acc, best_student_acc, best_baseline_acc, args)
    
    # Save results to file
    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        f.write(f"Teacher Network Accuracy: {best_teacher_acc:.4f}\n")
        f.write(f"Student Network (with KD) Accuracy: {best_student_acc:.4f}\n")
        f.write(f"Student Network (baseline) Accuracy: {best_baseline_acc:.4f}\n")
        f.write(f"Improvement with KD: {(best_student_acc - best_baseline_acc)*100:.2f}%\n")
        f.write(f"\nTraining Parameters:\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Distillation')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='number of workers for data loading (default: 2)')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='temperature for knowledge distillation (default: 4.0)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='weight for soft targets (default: 0.1)')
    parser.add_argument('--train-teacher', action='store_true',
                        help='train teacher network')
    
    args = parser.parse_args()
    main(args) 