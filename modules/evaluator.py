import torch

def evaluator(model, data, criterion, device):
    # on evaludation model
    model.eval()
    # total loss
    total_loss = 0
    # preds and targets
    all_preds = []
    all_targets = []
    # loop all batches
    for x,y in data:
        # load to device
        x = x.to(device)
        y = y.to(device)
        # predict
        output = model(x)
        # calculate loss
        loss = criterion(output, y)
        # make prediction
        pred = torch.argmax(output, dim=1)
        # add to total loss
        total_loss += loss.item()
        # append to all preds and targets
        all_preds.extend(pred)
        all_targets.extend(target)
    
    # calculate evalloss
    eval_loss = total_loss/len(data)
    # return the loss, preds and targets
    return eval_loss, all_preds, all_targets
