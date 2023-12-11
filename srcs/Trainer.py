
import torch

## hydra imports
from omegaconf import DictConfig

import srcs.Setting as setting

def train_model():
    pass 

def run(cfg: DictConfig):
    
    ## (1) Setting Dataloader 
    # train_dataloader, val_dataloader = setting.get_dataloader(cfg, cwd='./')
    dataloader = setting.get_dataloader(cfg)
    dataloader.load_data()

    ## (2) Setting Model
    model = setting.get_model(cfg)

    ## (3) Setting Loss Function
    criterion = setting.get_loss_fn(cfg)

    ## (4) Setting Other parameters
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7) #step_size=5, gamma=0.9)
    optimizer, scheduler = None, None 
    epochs = cfg.model.n_epochs

    ## (5) Start Training
    # train_model(cfg, dataloader, model, criterion, optimizer, scheduler, epochs)

    # return best_model, model, train_losses, val_losses

