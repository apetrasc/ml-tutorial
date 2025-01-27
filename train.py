import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from src.datasets.cifar10 import x_train,t_train,x_test,train_dataset,test_dataset
from src.models.cnn import conv_net
from src.utils.utils import fix_seed,gcn,ZCAWhitening,init_weights

#@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")

def train(args: DictConfig):
    fix_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")
        
    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    val_size = 3000
    trainval_data = train_dataset(x_train, t_train)
    test_data = test_dataset(x_test)
    train_data, val_data = torch.utils.data.random_split(trainval_data, [len(trainval_data)-val_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **loader_args)
    test_loader = torch.utils.data.DataLoader(
        test_data, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = conv_net.to(args.device)
    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loss_function = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    # ------------------
    #   Start training
    # ------------------  
    #l1_lambda = 1e-5  
    #l2_lambda = 1e-4 #use when you want to regularlize l1l2norm
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=10, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        conv_net.train()
        n_train = 0
        acc_train = 0
        losses_train = []

        for x, t in tqdm(train_loader, desc="Train"):
            x, t = x.to(args.device), t.to(args.device)
            optimizer.zero_grad()

            y = model(x)
            loss = loss_function(y, t)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(conv_net.parameters(), max_norm=1.0)

            optimizer.step()

            pred = y.argmax(dim=1)
            acc_train += (pred == t).sum().item()
            n_train += t.size(0)
            losses_train.append(loss.item())

        # 検証ループ
        conv_net.eval()
        n_val = 0
        acc_val = 0
        losses_valid = []

        with torch.no_grad():
            for x, t in tqdm(val_loader, desc="Validation"):
                x, t = x.to(args.device), t.to(args.device)

                y = conv_net(x)
                loss = loss_function(y, t)

                pred = y.argmax(dim=1)
                acc_val += (pred == t).sum().item()
                n_val += t.size(0)
                losses_valid.append(loss.item())
        scheduler.step()

        print(f'EPOCH: {epoch}, Train [Loss: {np.mean(losses_train):.3f}, Accuracy: {acc_train/n_train:.3f}], Valid [Loss: {np.mean(losses_valid):.3f}, Accuracy: {acc_val/n_val:.3f}]')
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(acc_val) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(acc_val)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    train()