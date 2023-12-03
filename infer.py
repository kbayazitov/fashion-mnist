import os

import hydra
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from torchvision import datasets, transforms

from config import Params
from src.fashionmnist.model import CNN
from src.fashionmnist.train_utils import val_model

device = "cuda" if torch.cuda.is_available() else "cpu"
cs = ConfigStore.instance()
cs.store(name="params", node=Params)
OUTPUT_FILE = "predictions.csv"


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    FMNIST_test = datasets.FashionMNIST(
        root="fmnist", train=False, download=True, transform=transforms.ToTensor()
    )
    model_path = os.path.join(cfg["training"]["export_dir"], "model.pth")
    # model_path = f"{cfg['training']['export_dir']}\model.pth"
    # model_path = os.path.join("export_dir", "model.pth")
    Model = CNN()
    Model.load_state_dict(torch.load(model_path))
    Model.to(device)
    Model.eval()

    val_targets, val_preds, test_acc, test_loss = val_model(Model, FMNIST_test)

    print(f"test accuracy={round(test_acc, 2)}")
    print(f"test loss={round(test_loss, 2)}")

    predictions = pd.DataFrame(
        {"True labels": val_targets, "Predicted labels": val_preds}
    )
    predictions.to_csv(
        os.path.join(cfg["training"]["export_dir"], OUTPUT_FILE), index=False
    )


if __name__ == "__main__":
    main()
