import torch.nn as nn

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_outputs, teacher_outputs, labels):
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(student_outputs, labels)
        
        # Knowledge distillation loss
        soft_student = (student_outputs / self.temperature).softmax(dim=1)
        soft_teacher = (teacher_outputs / self.temperature).softmax(dim=1)
        kd_loss = self.kl_loss(soft_student.log(), soft_teacher) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
        return total_loss