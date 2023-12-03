from dataclasses import dataclass


@dataclass
class Training:
    export_dir: str
    epochs: int
    lr: float


@dataclass
class Params:
    training: Training
