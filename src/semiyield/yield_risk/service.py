"""Manufacturing model interface shared by SECOM and synthetic demonstrations."""

from semiyield.common.modeling import train_model


def train_manufacturing(features, target, *, model="logistic", seed=42):
    return train_model(features, target, model_name=model, random_state=seed, calibrate=False)
