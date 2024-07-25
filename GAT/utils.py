import matplotlib.pyplot as plt

def visualize_predictions(model, data_loader):
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            x, edge_index, labels = data.x, data.edge_index, data.y
            out = model(x, edge_index)
            predictions = out.argmax(dim=1)
            plt.figure(figsize=(10, 5))
            plt.plot(predictions.cpu().numpy(), label='Predictions')
            plt.plot(labels.cpu().numpy(), label='Ground Truth')
            plt.legend()
            plt.show()
