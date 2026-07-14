from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(actual, predicted, prefix=""):
    actual = actual.detach().cpu().numpy()
    predicted = predicted.detach().cpu().numpy()

    return {
        f"{prefix}accuracy": accuracy_score(actual, predicted),
        f"{prefix}precision": precision_score(
            actual, predicted, average="macro", zero_division=0
        ),
        f"{prefix}recall": recall_score(
            actual, predicted, average="macro", zero_division=0
        ),
        f"{prefix}f1_score": f1_score(
            actual, predicted, average="macro", zero_division=0
        )
    }