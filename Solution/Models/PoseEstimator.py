import numpy as np
import torch
from .keypointrcnn import KeyPointRCNN
import wandb

class ClassifyPose(torch.nn.Module):
    def __init__(self, learning_rate = 1e-4):
        super().__init__()
        self.keypointrcnn = None 

        self.linear1 = torch.nn.Linear(17*4, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 200)
        self.linear3 = torch.nn.Linear(200, 200)
        self.linear4 = torch.nn.Linear(200, 5)
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()


        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        return x
    
    def calc_loss(self, outputs, labels):
        return self.cross_ent_loss(outputs, labels) 

    def backpropagate(self, loss):
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
    
    def pre_proccess(self, frames):
        frame_results = self.keypointrcnn.process_frames(frames)

        found_people = [i for i in range(len(frame_results)) if len(frame_results[i]['labels'])>0]
        keypoint_tensor = torch.zeros([len(found_people), 17*4], dtype=torch.float32)

        for idx in found_people:
            print("Found a person")
            key_points, key_points_score = frame_results[idx]['keypoints'], frame_results[idx]['keypoints_score']
            flattened_tensor = torch.cat((torch.tensor(key_points.flatten()), torch.tensor(key_points_score.flatten())))
            keypoint_tensor[idx] = flattened_tensor


def check_accuracy(device, model, test_loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for data  in test_loader:
            key_points = data["keypoints"]
            scores = data["scores"]
            labels = data["class"]
            scores = scores.unsqueeze(-1)
            data_full = torch.cat((key_points, scores), -1)
            data_full = data_full.flatten(-2, -1)

            data_full, labels = data_full.to(device), labels.to(device)
            
            scores = model.forward(data_full)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 

def TrainModel(model:ClassifyPose, total_epoch, train_iter, device, test_iter):
    epoch_loss = []
    

    for epoch in range(total_epoch):
        data_iter = iter(train_iter)
        
        iter_loss = np.zeros(0)
        loss_item = None
        model.train()
        for data in data_iter:
            # Get Data
            
            key_points = data["keypoints"]
            scores = data["scores"]
            labels = data["class"]

            scores = scores.unsqueeze(-1)
            data_full = torch.cat((key_points, scores), -1)
            data_full = data_full.flatten(-2, -1)

            
            
            data_full, labels = data_full.to(device), labels.to(device)

            # Step Model
            outputs = model.forward(data_full)
            loss = model.calc_loss(outputs, labels)
            model.backpropagate(loss)
            
            # Collect Stats
            loss_item = loss.detach().item()

            iter_loss = np.append(iter_loss, loss_item)

        
        
        epoch_loss.append(iter_loss.mean())

        # Print Status
        wandb.log({"loss":epoch_loss[-1]})
        print("Current Loss %.5f    Epoch" % loss_item)
        check_accuracy(device, model, test_iter)

    return epoch_loss