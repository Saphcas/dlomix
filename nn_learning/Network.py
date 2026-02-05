import torch
import torch.nn as nn

print(torch.cuda.is_available())
print(torch.version.cuda)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        # predict mean, variance, and missingness
        self.mean = nn.Linear(512, 1)
        self.var = nn.Linear(512, 1)
        self.missingness = nn.Linear(512, 1)

    def forward(self, x):
        x = self.flatten(x)
        embeddings = self.linear_relu_stack(x)  # one output

        mean = self.mean(embeddings)
        var = self.var(embeddings)
        missingness = self.missingness(embeddings)

        return mean, var, missingness
    

    # mock training loop
if __name__ == "__main__":
    # fake input
    import torch

    data = torch.randn(64, 1, 28, 28)  # batch of 64 images
    fake_labels = torch.randn(64, 3)  # batch of 64 labels (mean, var, missingness)

    model = NeuralNetwork()
    mean, var, missingness = model(data)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # calculate loss

    criterion = nn.MSELoss()
    loss_mean = criterion(mean, fake_labels[:, 0:1])
    loss_var = criterion(var, fake_labels[:, 1:2])
    loss_missingness = criterion(missingness, fake_labels[:, 2:3])

    total_loss = loss_mean + loss_var + loss_missingness

    total_loss.backward()
    optimizer.step()