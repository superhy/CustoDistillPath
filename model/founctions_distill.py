import torch

def train_distillation_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        student_outputs, teacher_outputs = model(None, features)
        loss = criterion(student_outputs, teacher_outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)