import torch
import copy
import pandas as pd
from modules.calculate_metrics import calculate_metrics
from modules.evaluator import evaluator

def test(model, save_path, data, criterion, device):
    # set model to device
    model.to(device)
    # load checkpoint 
    checkpoint = torch.load(save_path, map_location=device)
    # make a copy of the model
    test_model = copy.deepcopy(model)
    # load state dict
    test_model.load_state_dict(checkpoint)
    # evaluate model  
    test_loss, test_preds, test_targets = evaluator(model, data, criterion, device)
    # get metric 
    metric = calculate_metrics(test_targets, test_preds, prefix="test_")
    # show loss
    metric["test_loss"] = test_loss
    # return metric
    return metric