
import os
from venv import create
from darts.models import NBEATSModel, TCNModel
import torch
from dilateloss import DILATEloss,shapeDILATEloss, DILATEDIVloss, DILATEMSEloss, WeightedDILATEloss,  DerivativeDILATEloss, WDerivativeDILATEloss, shapeindepDILATEloss
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def build_model(args):
    global criterion
    global min_delta
    if args.criterion == 'dilate': # single output
        criterion = DILATEloss(alpha=args.dalpha, gamma = 0.1, device= 'cuda:{}'.format(args.gpu_index))
        min_delta = 0.01
    elif args.criterion == 'shapedilate': # single output
        criterion = shapeDILATEloss(alpha=args.dalpha, gamma = 0.1, device= 'cuda:{}'.format(args.gpu_index))
        min_delta = 0.01   
    elif args.criterion == 'shapeindepdilate': # single output
        criterion = shapeindepDILATEloss(alpha=args.dalpha, gamma = 0.1, device= 'cuda:{}'.format(args.gpu_index))
        min_delta = 0.01 
    elif args.criterion == "wd": # single output
        criterion = WeightedDILATEloss(alpha=args.dalpha, gamma = 0.1, device= 'cuda:{}'.format(args.gpu_index), g = 0.05)
        min_delta = 0.01 
    # elif args.criterion == 'dvd': # single output
    #     criterion = DerivativeDILATEloss(alpha=args.dalpha, gamma = 0.1, device= 'cuda:{}'.format(args.gpu_index))
    #     min_delta = 0.1     
    elif args.criterion == 'MSE': 
        criterion = torch.nn.MSELoss()
        min_delta = 0.001
        
    global model
    my_stopper = EarlyStopping(monitor="val_loss", patience=5, min_delta=min_delta, mode='min')
    
    def createFolder(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    base_dir = "/home/rj0517/darts/darts_logs"
    createFolder(base_dir + f'/{args.data}/{args.model}')

    if args.model == 'nbeats':
        model = NBEATSModel(input_chunk_length=args.inputlen, output_chunk_length=args.predlen,
        generic_architecture=True,num_stacks=30,num_blocks=1,num_layers=4,layer_widths=256,
        n_epochs = args.epoch,
        loss_fn = criterion,
        nr_epochs_val_period=10, # optimizer_cls = torch.optim.Adam,
        optimizer_kwargs = {'lr': 1e-4}, # lr_scheduler_cls = torch.optim.lr_scheduler.LambdaLR,
        batch_size=1024, 
        pl_trainer_kwargs = {"callbacks": [my_stopper], "accelerator": "gpu", "gpus": [args.gpu_index]},
        force_reset=True,
        save_checkpoints = True,
        log_tensorboard =True,
        # work_dir = "/home/rj0517/darts/darts_logs/{}/{}".format(args.model, args.criterion),
        work_dir = f"/home/rj0517/darts/darts_logs/{args.data}/{args.model}",
        model_name = '{}_{}_{}_{}'.format(args.criterion, args.dalpha, args.inputlen, args.predlen)
        )
    
    elif args.model == 'tcn':
        model = TCNModel(input_chunk_length=args.inputlen, output_chunk_length=args.predlen,
        n_epochs = args.epoch,
        dropout=0.1, dilation_base=2, weight_norm=True, kernel_size=5, num_filters=3, random_state=0,
        loss_fn = criterion,
        nr_epochs_val_period = 10, # optimizer_cls = torch.optim.Adam,
        optimizer_kwargs = {'lr': 1e-4}, # lr_scheduler_cls = torch.optim.lr_scheduler.LambdaLR,
        batch_size=1024, 
        pl_trainer_kwargs = {"callbacks": [my_stopper], "accelerator": "gpu", "gpus": [args.gpu_index]},
        force_reset=True,
        save_checkpoints = True,
        log_tensorboard =True,
        work_dir = f"/home/rj0517/darts/darts_logs/{args.data}/{args.model}",
        model_name = '{}_{}_{}_{}'.format(args.criterion, args.dalpha, args.inputlen, args.predlen)
        )
    
    return model