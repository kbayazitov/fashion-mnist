import os

import torch
from sklearn import metrics


def train_model(model, train_data, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    loss_function = torch.nn.CrossEntropyLoss()

    for i in range(cfg["training"]["epochs"]):
        train_generator = torch.utils.data.DataLoader(
            train_data, batch_size=64, shuffle=True
        )
        model.train()
        for x, y in train_generator:
            optimizer.zero_grad()
            x = x.view([-1, 1, 28, 28]).to(model.device)
            y = y.to(model.device)
            predict = model(x)
            loss = loss_function(predict, y)
            loss.backward()
            optimizer.step()

    if not os.path.exists(cfg["training"]["export_dir"]):
        os.makedirs(cfg["training"]["export_dir"])
    torch.save(model.state_dict(), f"{cfg['training']['export_dir']}/model.pth")


def val_model(model, test_data, input_shape=[-1, 1, 28, 28]):
    loss_function = torch.nn.CrossEntropyLoss()

    test_generator = torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=False
    )
    test_true = 0
    test_loss = 0
    val_targets = []
    val_preds = []
    for x, y in test_generator:
        x = x.view([-1, 1, 28, 28]).to(model.device)
        y = y.to(model.device)
        output = model(x)

        loss = loss_function(output, y)

        val_targets += list(y.cpu().detach().numpy())
        val_preds += list(torch.argmax(output, axis=1).cpu().detach().numpy())
        true_label = y.cpu()
        pred_label = torch.argmax(output, axis=1).cpu()
        test_true += metrics.accuracy_score(true_label, pred_label)
        test_loss += loss.cpu().item()

    return (
        val_targets,
        val_preds,
        test_true * 100 / len(test_data),
        test_loss * 100 / len(test_data),
    )
