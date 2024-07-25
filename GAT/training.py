import torch
from torch.optim import Adam
from embedding_module import embedding_module, mask_token_id
import numpy as np


def train_model(model, data_loader, epochs=300, lr=0.0003, weight_decay=0.0003):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            x, edge_index, labels = data.x, data.edge_index, data.y

            # Mask input
            x_masked, mask = embedding_module.mask_input(x, mask_token_id)

            # Forward pass
            out = model(x_masked, edge_index)

            # Compute loss only on masked positions
            loss = criterion(out[mask], x[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}')


def evaluate_model(model, data_loader):
    model.eval()
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for data in data_loader:
            x, edge_index, labels = data.x, data.edge_index, data.y

            # Mask input
            x_masked, mask = embedding_module.mask_input(x, mask_token_id)

            out = model(x_masked, edge_index)
            predictions = out.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)
    return total_correct / total_examples


def fine_tune_model(model, train_loader, val_loader, epochs=50, lr=0.0003, weight_decay=0.0003):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            x, edge_index, labels = data.x, data.edge_index, data.y

            # Mask input
            x_masked, mask = embedding_module.mask_input(x, mask_token_id)

            out = model(x_masked, edge_index)
            loss = criterion(out[mask], x[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate_model(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}, Validation Accuracy: {val_acc}')
