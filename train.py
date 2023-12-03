import hydra
import torch
from hydra.core.config_store import ConfigStore
from torchvision import datasets, transforms

from config import Params
from src.fashionmnist.model import CNN
from src.fashionmnist.train_utils import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"

cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    FMNIST_train = datasets.FashionMNIST(
        root="fmnist",
        train="True",
        download="True",
        transform=transforms.ToTensor(),
    )

    Model = CNN()
    Model.to(device)
    train_model(Model, FMNIST_train, cfg)


if __name__ == "__main__":
    main()
