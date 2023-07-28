# Document for *AdaMPI*
## Environment
```
conda create -n adampi python=3.8

# here we use pytorch 1.11.0 and CUDA 11.3 for an example 

# install pytorch
pip install https://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl

# install torchvision
pip install https://download.pytorch.org/whl/cu113/torchvision-0.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl

# install pytorch3d
conda install https://anaconda.org/pytorch3d/pytorch3d/0.6.2/download/linux-64/pytorch3d-0.6.2-py38_cu113_pyt1100.tar.bz2

# install other libs
pip install \
    numpy==1.19 \
    scikit-image==0.19.1 \
    scipy==1.8.0 \
    pillow==9.0.1 \
    opencv-python==4.4.0.40 \
    tqdm==4.64.0 \
    moviepy==1.0.3 \
    pyyaml \
    matplotlib \
    scikit-learn \
    lpips \
    kornia \
    focal_frequency_loss \
    tensorboard \
    transformers
```

## Inference
### Download Pretrained Model
Download the pretrained model from [here](https://drive.google.com/drive/folders/1NfXUlSTHc390YPkKSeddOghzl0W6q7wn?usp=sharing) and put it to `./adampiweight`.
We release two model, one trained with 32 MPI planes and the other trained with 64 planes.

### 3D Photo Generation
The input to our AdaMPI is a single in-the-wild image with its monocular depth estimation. 
You can use the [DPT](https://github.com/isl-org/DPT) model to obtain the estimated depth map. (If `--disp_path` is not given, the script runs "dpt-hybrid-midas" to estimate the depth map by default. You might see the depth map at `debug/midas_depth.png`)
We provide somne example inputs in `./images`, you can use the image and depth here to test our model. 
Here is an example to run the code: 

```
python gen_3dphoto.py \
    --img_path images/0810.png \
    --disp_path images/depth/0810.png \
    --width 384 \
    --height 256 \
    --save_path 0810.mp4 \
    --ckpt_path adampiweight/adampi_64p.pth
```

Then, you will see the result like that:

<img src="../misc/example_3dphoto.gif">

### Note
* To run our model successfully, make sure both the `--width` and `--height` is a multiple of 128.
* Our model is trained with resolution of 256 x 384; we empirically find our model can also work well at 512 x 768; we have not test the performance of our model under other resolution.
* The code related to MPI rendering is heavily borrowed from [MINE](https://github.com/vincentfung13/MINE).

## Training
### Download and Preprocess the Dataset
Download and extract the COCO dataset:
```
sh download_data.sh  # you can also manually download it from https://github.com/nightrome/cocostuff
```
Run DPT on the dataset to obtain the monocular depth map for each image in COCO:
```
python preprocess_data.py --img_root data/train2017 --save_root data/depth/train2017
python preprocess_data.py --img_root data/val2017 --save_root data/depth/val2017
```

### Training
Before training, change this parameters in `params_coco.yaml` to your data root:
```
data.training_set_path: /root/autodl-tmp/adampi-data/train2017
data.val_set_path: /root/autodl-tmp/adampi-data/val2017
data.training_depth_path: /root/autodl-tmp/adampi-data/depth/train2017
data.val_depth_path: /root/autodl-tmp/adampi-data/depth/val2017

if you directly follow our data processing scripts, data.training_set_path should be set to data/train2017.
```

The following command is tested on a server with `2xRTX3090`.
```
 # for 16 plane training
 # for 32 plane training
```

Note: 
* You can tune the `data.per_gpu_batch_size` and `training.step_iter` to fit your GPU memory. We recommand you to ensure the total batchsize (i.e. `data.per_gpu_batch_size x num_gpus x training.step_iter`) >= 12 to ensure good performance. 
* We support gradient accumulation by the `training.step_iter` parameter. So when you change the `training.step_iter`, you should also change the training iters. 
