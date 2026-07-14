import torch
from modules.calculate_metrics import calculate_metrics
from modules.evaluator import evaluator
from pathlib import Path
import pandas as pd
import os

class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        device="cuda",
        criterion=None
        ):
        
        self.model = model
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        self.model.to(device)
    
    # private method to single run of model
    def __train_one_epoch(self, data):
        # enable training mode
        self.model.train()
        # total loss
        total_loss = 0
        # preds and targets
        all_preds = []
        all_targets = []
        # loop on each batch
        for x,y in data:
            # convert to the device type
            x = x.to(self.device)
            y = y.to(self.device)
            # use optimizer
            self.optimizer.zero_grad()
            # pred
            output = self.model(x)            
            #calculate loss
            loss = self.criterion(output, y)
            # backward propagation
            loss.backward()
            self.optimizer.step()
            # add to total loss
            total_loss += loss.item()
            # preds calculated
            preds = torch.argmax(output, dim=1)
            # append to all_preds and all_targets
            all_preds.append(preds)
            all_targets.append(y)
        
        # training loss
        training_loss = total_loss/len(data)
        return training_loss, torch.cat(all_preds), torch.cat(all_targets)
    
    # public method to train complete model
    def fit(self, epochs, save_path, checkpoint):
        # metrics epoch
        metrics = []
        
        # make a save path
        os.makedirs(f"{save_path}/models", exist_ok=True)
        
        # loop each epoch to train
        for epoch in range(1, epochs+1):
            # training 
            training_loss, training_preds, training_targets = self.__train_one_epoch(self.train_data)
            training_metrics = calculate_metrics(training_targets, training_preds, "train_")
            # save
            if epoch%checkpoint == 0:
                torch.save(self.model.state_dict(), os.path.join(save_path, f"/models/epoch_{epoch}.pt"))      
            # validation
            validation_loss, validation_preds, validation_targets = evaluator(
                self.model, 
                self.val_data, 
                self.criterion, 
                self.device
                )
            validation_metrics = calculate_metrics(validation_targets, validation_preds, "val_")
            # complete metrics
            metrics.append({
                "epoch":epoch,
                "train_loss": training_loss,
                "val_loss": validation_loss,
                **training_metrics,
                **validation_metrics
            })
            if epoch==0:
                print(f"Trainer working properly!")
            # print to show a little status
            if (epoch+1)%5==0:
                print(f"EPOCH {epoch} completed | Training Loss = {training_loss} | Validation Loss = {validation_loss}")
        
        # save metrics to directory
        metrics = pd.DataFrame(metrics)
        metrics.to_csv(os.path.join(save_path, f"train_val_metrics.csv"), index=Flase)