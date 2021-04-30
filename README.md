# ViT_PyTorch
This is a simple PyTorch implementation of Vision Transformer (ViT) described in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

![image](https://user-images.githubusercontent.com/39369205/116412646-79987880-a869-11eb-9a43-7bb6b7036015.png)

# Usage
## Download pre-trained weights.
- Download [official jax weights](https://console.cloud.google.com/storage/browser/vit_models;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) and convert to PyTorch format.
```
python convert.py jax/weights/path/ converted/weights/path --model_name B_16_384
```
-  Or use the [converted PyTorch weights](https://drive.google.com/drive/folders/160hak04HM3XjmXo0HRb9YT1V2IUqMa90?usp=sharing) directly.

## Train ViT
You can train ViT on your own dataset by following command : 
```
python train.py /train/data/folder/ --valid_dir /validation/data/folder/ --pretrained_weights /pre-trained/weights/path 
```
Check out the [Colab](https://colab.research.google.com/drive/1s6SMji0U4KzyEdhhoMRaHxxWNaqpsXBZ?usp=sharing) for preparing data, fine-tuning the model, and inference.

### Details about training arguments :
Argument|Description|Type|Default
---|---|---|---
train_dir|Directory of training data.|str|required argument
valid_dir|Directory of validation data.|str|None
valid_rate|Proportion of validation sample split from training data.|float|None
output_dir|Directory of output results where trained weights and training history will be stored.|str|None
model_config|Modle arch configuration. (config path or arch name, e.g. "B_16_384")|str|B_16_384
pretrained_weights|Filename of pre-trained weights. Train from scratch if 'None'.|str|None
freeze_extractor|If True, freeze the feature extractor weights to fine-tune the classification head.|bool|True
batch_size|Batch size.|int|64
init_lr|Initial learning rate.|float|1e-3
weight_decay|Weight decay (L2 penalty).|float|1e-5
beta1|Adam 'betas' param 1.|float|0.9
beta2|Adam 'betas' param 2.|float|0.999
max_epoch|Maximun training epochs.|int|100
patient|Improved patient for early stopping.|int|None
monitor|Metric to be monitored. ('loss' or 'acc')|str|loss
min_delta|Minimum change in the monitored metric to qualify as an improvement.|float|0.0
save_best|Whether to save weights from the epoch with the best monitored metric.|bool|True
warmup|Warmup epochs.|int|0
scheduler|Training scheduler. ('cosine', 'step' or 'exp')|str|None
t_max|Maximum number of iterations. (cosine scheduler)|int|10
eta_min|Minimum learning rate. (cosine scheduler)|float|0.0
step_size|Period of learning rate decay. (step scheduler)|int|10
gamma|Multiplicative factor of learning rate decay. (step/exp scheduler)|float|0.1
image_size|Input image size.|int|384
crop_margin|Margin for random cropping.|int|32
horizontal_flip|Horizontal flip probability.|float|0.5
rotation|Degree for random rotation.|float|10.
device|Computation device. ('cuda' or 'cpu')|str|cuda
random_seed|Random seed in this repo.|int|427
