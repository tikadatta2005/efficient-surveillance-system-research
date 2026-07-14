import torch
import copy
import pandas as pd
from modules.calculate_metrics import calculate_metrics
from modules.evaluator import evaluator

def test(model, save_path, data, criterion, device):
    # set model to device
    model.to(device)
    # check points file
    model_files = [
            f for f in os.listdir(f"{save_path}/models")
            if f.endswith(".pt") and "epoch_" in f
        ]
    
    # metrics
    metrics = []
    
    # loop through each file
    for epoch, file in enumerate(model_files):
        # deep copy the model
        test_model = copy.deepcopy(model)
        # load checkpoint
        checkpoint = torch.load(
            os.path.join(save_path, file),
            map_location=device
            )
        # load state dict to model
        test_model.load_state_dict(checkpoint)
        # evaluate model
        test_loss, test_preds, test_targets = evaluator(model, data, criterion, device)
        # get metric 
        metric = calculate_metrics(test_targets, test_preds, prefix="test_")
        metric["test_loss"] = test_loss
        metric["epoch"] = epoch+1
        # append to the metrics
        metrics.append(metric)
    
    # convert to pandas
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(os.path.join(f"{save_path}/test_metrics.csv"), index=False)
    return metrics