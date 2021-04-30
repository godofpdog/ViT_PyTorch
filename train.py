import os 
import json
import torch
import pickle
import argparse
import numpy as np 
from datetime import datetime
from vit_pytorch.modules import ViT, build_head
from vit_pytorch.data import create_loaders
from vit_pytorch.configs import MODEL_CFGS
from vit_pytorch.utils import set_seed, get_num_params, freeze_model, Meter, mkdir, save_model
from vit_pytorch.solver import train_epoch, eval_epoch, get_criterion, get_optimizer, get_scheduler, WarmupScheduler, EarlyStopper


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(args):
    set_seed(args.random_seed)

    # prepare data
    print('Create data loaders.')
    train_loader, valid_loader, num_classes = create_loaders(args)
    print('Number of classes : {}.'.format(num_classes))
    print('Training sample : {}.'.format(len(train_loader.dataset)))
    
    if valid_loader is not None:
        print('Validation sample : {}.'.format(len(valid_loader.dataset)))

    # build model
    print('Build model.')
    is_build_head = False

    if args.model_config in MODEL_CFGS:
        is_build_head = True 
        model_config = MODEL_CFGS[args.model_config]
    else:
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)

    model = ViT(**model_config)

    if args.pretrained_weights is not None:
        model.load_state_dict(torch.load(args.pretrained_weights))
        print('Successfully load pre-trained weights from `{}`'.format(args.pretrained_weights))

    if args.freeze_extractor:
        print('Freeze feature extractor weights.')
        freeze_model(model)

    if model_config['repr_dim'] is not None:
        repr_dim = model_config['repr_dim']
    else:
        repr_dim = model_config['embed_dim']

    if is_build_head:
        model.head = build_head(repr_dim, num_classes)

    model.to(args.device)

    # init meters
    train_meter = Meter()

    if valid_loader is not None:
        valid_meter = Meter()

    # get criterion
    assert num_classes > 1
    loss = 'bce' if num_classes == 2 else 'ce'
    criterion = get_criterion(loss).to(args.device)

    # get optimizer and schedulers
    optimizer = get_optimizer(model, args) 
    warmup_scheduler = WarmupScheduler(optimizer, args.warmup)
    training_scheduler = get_scheduler(optimizer, args)

    if args.patient is not None:
        early_stopper = EarlyStopper(args.monitor, args.patient, args.min_delta)
    else:
        early_stopper = None 

    # output dir
    output_dir = args.output_dir

    if output_dir is None:
        output_dir = os.path.join(
            ROOT_DIR, 'results', 
            datetime.now().strftime('result_%Y-%m-%d-%H-%M')
        )

    mkdir(output_dir)
        
    # training
    best_score = 0
    not_improve_cnt = 0

    print('Start training.')
    for epoch in range(args.max_epoch):
        _meter_t = train_epoch(model, train_loader, criterion, optimizer, Meter(), args.device, epoch + 1)
        train_meter.merge(_meter_t)

        if valid_loader is not None:
            _meter_v = eval_epoch(model, valid_loader, criterion, Meter(), args.device, epoch + 1)
            valid_meter.merge(_meter_v)

        if valid_meter is not None and early_stopper is not None:
            early_stopper.step(np.mean(_meter_v[args.monitor]))

            if early_stopper.is_best and args.save_best:
                weights_path = os.path.join(output_dir, 'improved_ep{}.pt'.format(str(epoch + 1)))
                save_model(model, weights_path)
            else:
                print('No improved count : {}/{}'.format(early_stopper.not_improved_cnt, args.patient))

            if early_stopper.is_early_stop:
                print('Early stop at epoch {}'.format(not_improve_cnt))
                break
                        
    # save results 
    weights_path = os.path.join(output_dir, 'weights.pt')
    save_model(model, weights_path)

    try:
        model_config_path = os.path.join(output_dir, 'model_config.json')

        with open(model_config_path, 'w') as f:
            json.dump(model_config, f)

        print('Successfully save training history to `{}/`'.format(output_dir))

    except Exception as e:
        print(e)

    try:
        train_hist_path = os.path.join(output_dir, 'train_history.csv')
        valid_hist_path = os.path.join(output_dir, 'valid_history.csv')
        
        train_meter.to_dataframe().to_csv(train_hist_path)        

        if valid_meter is not None:
            valid_meter.to_dataframe().to_csv(valid_hist_path)   

        print('Successfully save training history to `{}/*`'.format(output_dir))

    except Exception as e:
        print(e)

    print('Training process done.')
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')

    # paths
    argparser.add_argument('train_dir', type=str, help='Directory of training data.')
    argparser.add_argument('--valid_dir', type=str, help='Directory of validation data.', default=None)
    argparser.add_argument('--valid_rate', type=str, help='Proportion of validation sample splitted from training data.', default=None)  
    argparser.add_argument('--output_dir', type=str, help='Output directory.', default=None)

    # model
    argparser.add_argument('--model_config', type=str, help='Modle arch configuration. (config path or arch name, e.g. "B_16_384")', default='B_16_384')
    argparser.add_argument('--pretrained_weights', type=str, help='Pre-trained weights filename.', default=None)
    argparser.add_argument('--freeze_extractor', type=bool, help='If True, freeze the feature extractor weights.', default=True)
    
    # training
    argparser.add_argument('--batch_size', type=int, help='Batch size.', default=64)
    argparser.add_argument('--init_lr', type=float, help='Initial learning rate.', default=1e-3) 
    argparser.add_argument('--weight_decay', type=float, help='Weight decay (L2 penalty).', default=1e-5)
    argparser.add_argument('--beta1', type=float, help='Adam `betas` param 1.', default=0.9)
    argparser.add_argument('--beta2', type=float, help='Adam `betas` param 2.', default=0.999)   
    argparser.add_argument('--max_epoch', type=int, help='Maximun training epochs.', default=100)
    argparser.add_argument('--patient', type=int, help='Improved patient for early stopping', default=None)
    argparser.add_argument('--monitor', type=str, help='Metric to be monitored', choices=['loss', 'acc'], default='loss')
    argparser.add_argument('--min_delta', type=float, help='Minimum change in the monitored metric to qualify as an improvement', default=0.)
    argparser.add_argument('--save_best', type=bool, help='Whether to save weights from the epoch with the best monitored metric', default=True)
    argparser.add_argument('--warmup', type=int, help='Warmup epochs.', default=0)
    argparser.add_argument('--scheduler', type=str, help='Training scheduler.', choices=['cosine', 'step', 'exp'], default=None)
    argparser.add_argument('--t_max', type=int, help='Maximum number of iterations (cosine).', default=10)
    argparser.add_argument('--eta_min', type=float, help='Minimum learning rate. (cosine)', default=0.)
    argparser.add_argument('--step_size ', type=int, help='Period of learning rate decay. (step)', default=10)
    argparser.add_argument('--gamma', type=float, help='Multiplicative factor of learning rate decay. (step/exp)', default=0.1)

    # augmentation
    argparser.add_argument('--image_size', type=int, help='Input image size.', default=384)
    argparser.add_argument('--crop_margin', type=int, help='Margin for random cropping.', default=32)
    argparser.add_argument('--horizontal_flip', type=float, help='Horizontal flip prob.', default=0.5)
    argparser.add_argument('--rotation', type=float, help='Degree for random rotation.', default=10.)
    argparser.add_argument('--device', type=str, help='Computation device.', default='cuda')
    argparser.add_argument('--random_seed', type=int, help='Random seed in this repo.', default=427)
     
    args = argparser.parse_args()
    main(args)
