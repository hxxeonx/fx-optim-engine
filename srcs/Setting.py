
''' Environment Settings '''

import torch
import torch.nn as nn

from srcs.utils import * 
from srcs.Dataloader import dataLoader
from transformers.Transformer_optim import Basic_Transformer


def get_dataloader(cfg, cwd = './'):
    print(f'\nINFO: [get_dataloader] Loading Dataloader for dataset: {cfg.dataset.name} for npy: {cfg.dataset.npy}\n')
    return dataLoader(cfg)
    
def get_model(cfg):

    # (2-1) 선택한 모델에 맞게 모델 로드
    model = None
    if cfg.model.name == "Transformer_optimize":
        if cfg.model.model_type == "Basic":
            model = Basic_Transformer(cfg)
    else:
        raise NotImplementedError(f"Invalid model name: {cfg.model.name}")
    
    assert model != None

    # (2-2) 데이터 타입 설정
    if cfg.dataset.data_type == "float64":
        return model.double()
    elif cfg.dataset.data_type == "float32":
        return model.float()
    else:
        print(f"ERROR: [get_model] Given data type is not supported yet: {cfg.dataset.data_type}")
        raise NotImplementedError()

def get_loss_fn(cfg):
    criterion = None
    if cfg.dataset.loss_fn in ['CrossEntropy', 'LogitNormLoss']:
        criterion = nn.CrossEntropyLoss()
    elif cfg.dataset.loss_fn == 'BCEWithLogits':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.dataset.loss_fn == 'BCE':
        criterion = nn.BCELoss()
    elif cfg.dataset.loss_fn == "FocalLoss":
        criterion = FocalLoss(cfg.dataset.focal_loss_gamma)
    elif cfg.dataset.loss_fn == "MSELoss":
        criterion = nn.MSELoss()
    elif cfg.dataset.loss_fn == "RMSELoss":
        criterion = RMSELoss()
    return criterion

def prepare_gpus(cfg, model):
    gpus = cfg.dataset.gpus

    first_gpu=0
    if torch.cuda.is_available() and len(gpus)>0:
        first_gpu = gpus[0]
    
    multi_gpu = False
    avail_devices = torch.cuda.device_count()
    device = torch.device(f"cuda:{first_gpu}" if torch.cuda.is_available() else "cpu")

    if len(gpus) == 1:
        device = torch.device(f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    elif len(gpus) == 0:
        device = torch.device("cpu")
        model = model.to(device)

    elif avail_devices > 1 and len(gpus) > 1:
        print(f"INFO: [prepare_gpus] Preparing Multi-GPU setting with GPUS: {gpus}...")
        model = nn.DataParallel(model, device_ids=gpus)
        model = model.to(device)
        multi_gpu = True

    elif len(gpus) > avail_devices:
        print(f"ERROR: [prepare_gpus] Make sure you have enough GPU's")
        raise NotImplementedError
    
    else:
        print(f"ERROR: [prepare_gpus] Check your gpu lists again!")
        raise NotImplementedError
    
    return model, device, multi_gpu
