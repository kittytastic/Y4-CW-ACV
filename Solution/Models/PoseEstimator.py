import numpy as np
import torch
from .keypointrcnn import KeyPointRCNN

class ClassifyPose(torch.nn.Module):
    def __init__(self, keypointrcnn:KeyPointRCNN, learning_rate = 1e-4):
        super().__init__()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.keypointrcnn = keypointrcnn

        self.linear1 = torch.nn.Linear(17*4, 200)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 200)
        self.linear3 = torch.nn.Linear(200, 200)
        self.linear4 = torch.nn.Linear(200, 5)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.softmax(x)
        return x
    
    def calc_loss(self, approx, truth):
        return (approx - truth).pow(2).mean()

    def backpropagate(self, loss):
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
    
    def pre_proccess(self, frames):
        frame_results = self.keypointrcnn.process_frames(frames)

        keypoint_tensor = torch.zeros([len(frame_results), 17*4], dtype=torch.float32)
        for i in range(len(frame_results)):
            assert(len(frame_results[i])>0)
            frame_person = frame_results[i][0]
            print("Found a person")



def TrainModel(model:ClassifyPose, total_epoch, train_iter, device):

    
    epoch_loss = []
    

    for epoch in range(total_epoch):
        
        iter_loss = np.zeros(0)
        loss_item = None

        for i in range(10):
            # Get Data
            x,t = next(train_iter)
            x,t = x.to(device), t.to(device)

            # Step Model
            loss = model.forwardStep(x)
            model.backpropagate(loss)
            
            # Collect Stats
            loss_item = loss.detach().item()

            iter_loss = np.append(iter_loss, loss_item)

        
        
        epoch_loss.append(iter_loss.mean())

        # Print Status
        epoch_iter.set_description("Current Loss %.5f    Epoch" % loss_item)

    return (epoch_loss, t_mmd, t_recon)