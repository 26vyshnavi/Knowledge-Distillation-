import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import accuracy, AverageMeter

def train_teacher(model, trainloader, testloader, device, epochs=200, history=None):
    """Train the teacher network"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()

        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            acc = accuracy(outputs, targets)
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(acc, inputs.size(0))
            
            pbar.set_postfix({'loss': f'{train_loss.avg:.4f}', 
                            'acc': f'{train_acc.avg:.4f}'})

        # Evaluate on test set
        test_acc = evaluate(model, testloader, device)
        print(f'Epoch {epoch+1}: Test Acc: {test_acc:.4f}')
        
        # Update history if provided
        if history is not None:
            history['test_acc'].append(test_acc)
            history['train_loss'].append(train_loss.avg)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_teacher.pth')
            
        scheduler.step()
    
    return best_acc

def train_student_kd(student, teacher, trainloader, testloader, device, 
                    temperature=4.0, alpha=0.1, epochs=200, history=None):
    """Train the student network using knowledge distillation"""
    teacher.eval()  # Teacher network should be in eval mode
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        student.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()

        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get soft targets from teacher
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
                soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
            
            # Student forward pass
            student_outputs = student(inputs)
            student_soft = F.softmax(student_outputs / temperature, dim=1)
            
            # Calculate losses
            # Hard targets loss
            hard_loss = criterion_ce(student_outputs, targets)
            
            # Soft targets loss (KL divergence)
            # Note: KL divergence expects log probabilities for the first argument
            # and probabilities for the second argument
            soft_loss = F.kl_div(
                F.log_softmax(student_outputs / temperature, dim=1),
                soft_targets,
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # Combined loss
            loss = (1 - alpha) * hard_loss + alpha * soft_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(student_outputs, targets)
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(acc, inputs.size(0))
            
            pbar.set_postfix({'loss': f'{train_loss.avg:.4f}', 
                            'acc': f'{train_acc.avg:.4f}'})

        # Evaluate on test set
        test_acc = evaluate(student, testloader, device)
        print(f'Epoch {epoch+1}: Test Acc: {test_acc:.4f}')
        
        # Update history if provided
        if history is not None:
            history['test_acc'].append(test_acc)
            history['train_loss'].append(train_loss.avg)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), 'best_student_kd.pth')
            
        scheduler.step()
    
    return best_acc

def evaluate(model, testloader, device):
    """Evaluate the model on the test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return correct / total 
