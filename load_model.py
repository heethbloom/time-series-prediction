

from darts.models import NBEATSModel, TCNModel
import torch
from dilateloss import DILATEloss,  WeightedDILATEloss,  DerivativeDILATEloss, shapeDILATEloss , shapeindepDILATEloss
# DILATEDIVloss, DILATEMSEloss,WDerivativeDILATEloss

def load_model(args):  
    work_dir = f"/home/rj0517/darts/darts_logs/{args.data}/{args.model}/"
    if args.model == 'nbeats':
        model = NBEATSModel.load_from_checkpoint(work_dir + '{}_{}_{}_{}'.format( args.criterion, args.dalpha, args.inputlen, args.predlen ), best=True)
        return model
    
    elif args.model == 'tcn':
        model = TCNModel.load_from_checkpoint(work_dir + '{}_{}_{}_{}'.format( args.criterion, args.dalpha, args.inputlen, args.predlen ), best=True)
        return model
    