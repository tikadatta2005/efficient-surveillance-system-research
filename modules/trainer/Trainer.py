import torch

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
        
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
    
    # single run of model
    def train_one_epoch(self, data):
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
            loss = self.criterion(outputs, y)
            # backward propagation
            loss.backward()
            self.optimizer.step()
            # add to total loss
            total_loss += loss.item()
            # preds calculated
            preds = torch.argmax(output, dim=1)
            # append to all_preds and all_targets
            all_preds.extend(preds)
            all_targets.extend(y)
        
        # training loss
        training_loss = total_loss/len(data)
        return training_loss, all_preds, all_targets
        
    