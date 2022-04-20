from typing import Any, Dict
import numpy as np
import torch
import wandb
import os
from tqdm import trange

class ClassifyPose(torch.nn.Module):
    def __init__(self, learning_rate = 1e-4):
        super().__init__()
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
    

    def checkpoint(self, path, name, verbose:bool=False):
        checkpoint_path = os.path.join(path, f'{name}.chkpt')
        torch.save({'model':self.state_dict()}, checkpoint_path)
        if verbose: print(f"Saved model to: {checkpoint_path}")

    def restore(self, path, name):
        params = torch.load(os.path.join(path, f'{name}.chkpt'))
        self.load_state_dict(params['model'])

    def pack_keypoint_tensor(self, data:Dict[str, Any])->Any:
        key_points = data["nomalised_keypoints"]
        scores = data["scores"]
        
        scores = scores.unsqueeze(-1)
        data_full = torch.cat((key_points, scores), -1)
        data_full = data_full.flatten(-2, -1)

        return data_full

    def check_accuracy(self, device, test_loader, verbose=False):
        model = self
        num_correct = 0
        num_samples = 0
        model.eval()
        
        with torch.no_grad():
            for data  in test_loader:
                labels = data["class"]
                data_full = self.pack_keypoint_tensor(data) 

                data_full, labels = data_full.to(device), labels.to(device)
                
                scores = model.forward(data_full)
                _, predictions = scores.max(1)
                correct = predictions == labels
                num_correct += (predictions == labels).sum()
                num_samples += predictions.size(0)
                if verbose:
                    for idx, c in enumerate(correct):
                        if not c:
                            print(f"{data['file_name'][idx]}\t\tguessed: {predictions[idx]}   was:{labels[idx]}")



        if verbose:
            print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
        return float(num_correct)/float(num_samples)*100

    def do_training(self, device, total_iters, train_iter, test_iter):
        model = self
        epoch_loss = []
        epoch_iter = trange(total_iters)

        for _ in epoch_iter:
            iter_loss = np.zeros(0)
            loss_item = None
            model.train()
            for data in train_iter:
                # Get Data
                labels = data["class"]
                data_full = self.pack_keypoint_tensor(data)
                
                data_full, labels = data_full.to(device), labels.to(device)

                # Step Model
                outputs = model.forward(data_full)
                loss = model.calc_loss(outputs, labels)
                model.backpropagate(loss)
                
                # Collect Stats
                loss_item = loss.detach().item()

                iter_loss = np.append(iter_loss, loss_item)

        
            epoch_loss.append(iter_loss.mean())
            acc = self.check_accuracy(device, test_iter)
            
            wandb.log({"loss":epoch_loss[-1], "acc":acc})
            epoch_iter.set_description(f"Current Loss {epoch_loss[-1]:.5f}    Epoch")

        return epoch_loss