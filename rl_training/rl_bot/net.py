import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import platform



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Conv2d(7, 16, kernel_size=7, stride=1, padding=3)
        self.layer2 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3)
        self.layer3 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3)
        self.layer4 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = F.sigmoid(self.layer1(x))
        out = F.sigmoid(self.layer2(out))
        out = F.sigmoid(self.layer3(out))
        out = F.sigmoid(self.layer4(out))
        return out

    def my_train(self, inputs, labels, epochs=10):

        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(self.parameters())



        for epoch in range(epochs):  # loop over the dataset multiple times
            if epoch % 1 == 0:
                print("Epoch:", epoch, "out of:", epochs)
            running_loss = 0.0


            # get the inputs


            # wrap them in Variable
            ins, outs = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self(ins)
            loss = criterion(outputs, outs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            print("loss: {} running loss: {}".format(loss.data[0], running_loss))
