"""
Convert official jax weights to PyTorch format.

"""

import os 
import torch
import argparse
import numpy as np 
from PIL import Image 
from vit_pytorch.configs import *
from vit_pytorch.modules import ViT
from torchvision import transforms as trns 


def _rename(k):
    k = k.replace('Transformer/encoder_norm', 'transformer.norm')
    k = k.replace('LayerNorm_0', 'norm1')
    k = k.replace('LayerNorm_2', 'norm2')
    k = k.replace('MlpBlock_3/Dense_0', 'mlp.block.0')
    k = k.replace('MlpBlock_3/Dense_1', 'mlp.block.3')
    k = k.replace('encoderblock_', 'blocks.')
    k = k.replace('MultiHeadDotProductAttention_1', 'atten')
    k = k.replace('posembed_input/pos_embedding', 'pos_embedding.embedding')
    k = 'patch_embedding.proj.bias' if k == 'embedding/bias' else k
    k = 'patch_embedding.proj.weight' if k == 'embedding/kernel' else k
    k = 'cls_token' if k == 'cls' else k
    k = k.replace('pre_logits', 'pre_logits.proj.0')
    k = k.replace('kernel', 'weight')
    k = k.replace('scale', 'weight')
    k = k.replace('/', '.')
    k = k.lower()

    return k


def convert(weights_path, state_dict):
    jax_weights = {_rename(k): v for k, v in np.load(weights_path).items()}
    
    pytorch_weights = dict()

    for key, val in state_dict.items():
        
        if 'atten.qkv_proj.weight' in key:
            qkv = []
            
            for s in ('query', 'key', 'value'):
                k = key.replace(
                    'atten.qkv_proj.weight', 'atten.' + s + '.weight')

                w = jax_weights[k]
                w = w.reshape(-1, w.shape[1] * w.shape[2])
                qkv.append(w.transpose(1, 0))

            v = np.vstack(qkv)
            
        elif 'atten.qkv_proj.bias' in key:
            qkv = []

            for s in ('query', 'key', 'value'):
                k = key.replace(
                    'atten.qkv_proj.bias', 'atten.' + s + '.bias')

                w = jax_weights[k].reshape(-1)
                qkv.append(w)

            v = np.vstack(qkv)

        elif 'atten.out_proj.weight' in key:
            k = key.replace('out_proj', 'out')
            v = jax_weights[k]

        elif 'atten.out_proj.bias' in key:
            k = key.replace('out_proj', 'out')
            v = jax_weights[k]
            
        else:
            if key not in jax_weights:
                raise ValueError('Invalid weights name: `{}`.'.format(key))

            v = jax_weights[key]

        # to tensor
        v = torch.from_numpy(v)

        if '.weight' in key:
            if len(val.shape) == 2:
                v = v.transpose(0, 1)
            elif len(val.shape) == 4:
                v = v.permute(3, 2, 0, 1)

        if 'atten.qkv_proj.weight' in key:
            v = v.transpose(0, 1)

        if 'atten.qkv_proj.bias' in key:
            v = v.reshape(-1)

        if 'atten.out_proj.weight' in key:
            v = v.transpose(0, 1)
            v = v.reshape(-1, v.shape[-1]).T

        # assign weights
        pytorch_weights[key] = v
        
    return pytorch_weights


def main(model_name, src_path, dst_path):
    config = MODEL_CFGS.get(model_name)

    if config is None:
        raise ValueError(
            'Invalid model config. ' 
            'Valid options : {}'.format(sorted(MODEL_CFGS.keys()))
        ) 

    model = ViT(**config)
    model.load_state_dict(convert(src_path, model.state_dict()))
    model.eval()

    image_size = int(model_name.split('_')[-1])

    transform = trns.Compose([trns.Resize((image_size, image_size)),
                              trns.ToTensor(),
                              trns.Normalize(0.5, 0.5)])

    img_dir = 'samples/images'
    gts = [285, 254, 388]

    with torch.no_grad():
        paths = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]

        for i, path in enumerate(paths):
            image = transform(Image.open(path)).unsqueeze(0)
            pred = torch.argmax(model(image).squeeze(0), dim=-1)
            
            print('prediction is : {}'.format(pred))

    torch.save(model.state_dict(), dst_path)

    return None 
        

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('src_path', type=str, help='Jax weights path.')  
    argparser.add_argument('dst_path', type=str, help='Converted weights path.')  
    argparser.add_argument('--model_name', type=str, help='Modle arch name.', default='B_16_384')   
    
    args = argparser.parse_args()

    main(args.model_name, args.src_path, args.dst_path)
