
''' Learning Process'''
from tqdm import tqdm

import pandas as pd
import numpy as np 
import torch

from omegaconf import DictConfig

import srcs.Setting as setting

def train_model(cfg, dataloader, model, criterion, optimizer, scheduler, epochs):

    ##### Set Variables #####
    ## (1) Dataload & Devices
    dataloader.load_data()
    model, device, multi_gpu = setting.prepare_gpus(cfg, model)

    ## (2) Datatypes 
    data_type_str   = cfg.dataset.data_type
    float_data_type = torch.double
    int_data_type   = torch.long

    ## (3) loss.
    train_losses     = np.zeros(epochs)
    val_losses       = np.zeros(epochs)
    best_val_loss    = np.inf
    best_val_epoch   = 0

    ##### Save the Config file (실험 관리를 위함) #####
    with open('./hydra_config.txt', 'w') as txt:
        txt.writelines(str(cfg))


    for it in tqdm(range(epochs)):
        
        model.train()
        train_loss          = []
        
        for batch_idx, dataset in enumerate(dataloader.train_dataloader):
            
            ### Setting inputs & targets
            inputs, targets = dataset['X'], dataset['y']
            if cfg.dataset.loss_fn in ["MSELoss", "RMSELoss"]:
                inputs, targets = inputs.to(device, dtype=float_data_type), targets.to(device, dtype=float_data_type)
                targets = torch.unsqueeze(targets, 1)

            ### Prediction & Backward  
            optimizer.zero_grad()

            outputs = model(inputs)
            loss    = criterion(outputs, targets)       

            loss.backward()
            optimizer.step()

            ### Save & Print Results ### 
            train_loss.append(loss.item())     
            if batch_idx % 10 == 0:
                print("\nINFO: [train_model] Batch : ", batch_idx, f"\t Mean Train_loss : {np.mean(train_loss):.5f}")
        
        train_loss       = np.mean(train_loss)
        train_losses[it] = train_loss


def run(cfg: DictConfig):
    
    ## (1) Setting Dataloader 
    dataloader = setting.get_dataloader(cfg)

    ## (2) Setting Model
    model = setting.get_model(cfg)

    ## (3) Setting Loss Function
    criterion = setting.get_loss_fn(cfg)

    ## (4) Setting Other parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7) #step_size=5, gamma=0.9)
    epochs    = cfg.model.n_epochs

    ## (5) Start Training
    train_model(cfg, dataloader, model, criterion, optimizer, scheduler, epochs)


