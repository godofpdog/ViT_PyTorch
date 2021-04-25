import os 
import torch
import pickle
import argparse
from datetime import datetime
from vit_pytorch.modules import ViT, build_head
from vit_pytorch.data import create_loaders
from vit_pytorch.configs import MODEL_CFGS
from vit_pytorch.utils import set_seed, get_num_params, freeze_model, Meter, mkdir
from vit_pytorch.solver import train_epoch, eval_epoch, get_criterion, get_optimizer, get_scheduler, WarmupScheduler


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
    model_config = MODEL_CFGS[args.model_name]
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

    model.head = build_head(repr_dim, num_classes)
    model.to(args.device)

    # get criterion
    assert num_classes > 1
    loss = 'bce' if num_classes == 2 else 'ce'
    criterion = get_criterion(loss).to(args.device)

    # get optimizer and schedulers
    optimizer = get_optimizer(model, args) 
    warmup_scheduler = WarmupScheduler(optimizer, args.warmup)
    training_scheduler = get_scheduler(optimizer, args)

    # init meters
    train_meter = Meter()
    valid_meter = Meter() if valid_loader is not None else None

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
        train_epoch(model, train_loader, criterion, optimizer, train_meter, args.device, epoch + 1)

        if valid_loader is not None:
            eval_epoch(model, valid_loader, criterion, valid_meter, args.device, epoch + 1)

        if valid_meter is not None:
            if valid_meter['acc'][-1] > best_score:
                print('Validation acc has improved from `%.6f` to `%.6f`' %(valid_meter['acc'][-1], best_score))
                weights_path = os.path.join(output_dir, 'improved_ep{}.pt'.format(str(epoch + 1)))
                torch.save(model.state_dict(), weights_path)
                best_score = valid_meter['acc'][-1]
                not_improve_cnt = 0
            else:
                not_improve_cnt += 1
                print('No improved count : {}/{}'.format(not_improve_cnt, args.patient))
        
        if args.patient is not None:
            if not_improve_cnt >= args.patient:
                print('Early stop at epoch {}'.format(not_improve_cnt))
                break
                
    # save results 
    try:
        weights_path = os.path.join(output_dir, 'weights.pt')
        torch.save(model.state_dict(), weights_path)
        print('Successfully save weights to `{}`'.format(weights_path))
    except Exception as e:
        print(e)

    try:
        train_hist_path = os.path.join(output_dir, 'train_history.pickle')
        valid_hist_path = os.path.join(output_dir, 'valid_history.pickle')

        with open(train_hist_path, 'wb') as f:
            pickle.dump(train_meter, f, protocol=pickle.HIGHEST_PROTOCOL)

        if valid_meter is not None:
            with open(valid_hist_path, 'wb') as f:
                pickle.dump(valid_meter, f, protocol=pickle.HIGHEST_PROTOCOL)

        print('Successfully save training history to `{}/`'.format(output_dir))

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
    argparser.add_argument('--model_name', type=str, help='Modle arch name.', default='B_16_384')
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
    argparser.add_argument('--warmup', type=int, help='Warmup epochs.', default=0)
    argparser.add_argument('--scheduler', type=str, help='Training scheduler.', choices=['cosine, step, exp'], default=None)
    argparser.add_argument('--t_max', type=int, help='Maximum number of iterations (cosine).', default=None)
    argparser.add_argument('--eta_min', type=float, help='Minimum learning rate. (cosine)', default=0)
    argparser.add_argument('--step_size ', type=int, help=' Period of learning rate decay. (step)', default=None)
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
