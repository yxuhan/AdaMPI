# Document for *AdaMPI*
## Environment
```
conda create -n adampi python=3.8

# here we use pytorch 1.11.0 and CUDA 11.3 for an example 

# install pytorch
pip install https://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl

# install torchvision
pip install https://download.pytorch.org/whl/cu113/torchvision-0.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl

# install other libs
pip install \
    numpy==1.19 \
    scikit-image==0.19.1 \
    scipy==1.8.0 \
    pillow==9.0.1 \
    opencv-python==4.4.0.40 \
    tqdm==4.64.0 \
    moviepy==1.0.3
```

## Code
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
