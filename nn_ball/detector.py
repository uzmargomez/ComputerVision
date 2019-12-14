import torch
import torch.nn as nn
import torch.nn.functional as F

# Main Neural Network object
# This trained network works very well with circles
# which radius is larger that 0.1 image size
# smaller that that it seems to not work so well
# Play with this architecture to see if you can
# find better predictions

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 48 * 48, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


    def predict(self, x):
        y_hat = self.forward(x)

        y_hat[:, 0] = torch.sigmoid(y_hat[:, 0])

        return y_hat.detach().numpy()


    def predict_with_conv(self, x):
        c1 = self.pool(F.relu(self.conv1(x)))
        c2 = self.pool(F.relu(self.conv2(c1)))
        x = self.flatten(c2)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x[:, 0] = torch.sigmoid(x[:, 0])

        return x.detach().numpy(), c1.detach().numpy(), c2.detach().numpy()
