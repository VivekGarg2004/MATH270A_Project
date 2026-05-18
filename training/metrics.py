from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred):

    metrics = {}

    metrics["macro_f1"] = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    metrics["worker_f1"] = f1_score(
        y_true,
        y_pred,
        labels=[0],
        average="macro"
    )

    metrics["manager_f1"] = f1_score(
        y_true,
        y_pred,
        labels=[1],
        average="macro"
    )

    metrics["ceo_f1"] = f1_score(
        y_true,
        y_pred,
        labels=[2],
        average="macro"
    )

    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred
    )

    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        digits=4
    )

    return metrics