# scripts/distillation.py
import torch
import torch.nn.functional as F


def similarity_preserving_loss(teacher_activations, student_activations):
    def compute_similarity_matrix(activations):
        norm_activations = F.normalize(activations, p=2, dim=1)
        similarity_matrix = torch.mm(norm_activations, norm_activations.t())
        return similarity_matrix

    teacher_similarity = compute_similarity_matrix(teacher_activations)
    student_similarity = compute_similarity_matrix(student_activations)
    loss = F.mse_loss(student_similarity, teacher_similarity)
    return loss


def generate_rna_family_loss(output, target):
    return F.cross_entropy(output, target)


def train_with_distillation(teacher_model, student_model, data_loader, criterion, optimizer, device):
    teacher_model.train()
    student_model.train()

    for data in data_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass through teacher model
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        # Forward pass through student model
        student_outputs = student_model(inputs)

        # Compute losses
        distillation_loss = similarity_preserving_loss(teacher_outputs, student_outputs)
        generation_loss = generate_rna_family_loss(student_outputs, labels)
        loss = generation_loss + distillation_loss

        loss.backward()
        optimizer.step()

    return student_model
