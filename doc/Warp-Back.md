# Document for *Warp-Back*
## Environment
```
conda create -n warpback python=3.8

# here we use pytorch 1.11.0 and CUDA 11.3 for an example 
# for other version of pytorch or CUDA, you can find the corresponding
# pytorch3d file at https://anaconda.org/pytorch3d/pytorch3d/files

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
    opencv-python==4.4.0.40
```

## Prepare RGBD Dataset
For a single-view dataset like COCO, you should first get the monocular depth estimation for each image using [DPT](https://github.com/isl-org/DPT).

Then, you should organize the RGBD dataset as the following structure:
```
data_root
|- xxx.jpg  // single-view in-the-wild image
|- yyy.png
|- ...
|- dpt_depth  // the output for DPT, i.e. 16-bit disparity map
    |- xxx.png  // the corresponding depth map for xxx.jpg
    |- yyy.png
    |- ...
```
We provide an example toy dataset in `warpback/toydata`, there are ~20 random images from the COCO val set with monocular depth map estimated by DPT. 

## Code
### Stage 1
* We implement the dataloader for *Stage 1: Inpainting Network Training* in `warpback/stage1_dataset.py`.
* You can use it to train your own network specialized for inpainting holes caused by view change.
* `python warpback/stage1_dataset.py` to see a demo.

### Stage 2
* We implement the dataloader for *Stage 2: View Synthesis Training Pair Generation* in `warpback/stage2_dataset.py`.
* We provide our pretrained [EdgeConnect](https://github.com/knazeri/edge-connect) model at [here](https://drive.google.com/drive/folders/1FZZ6laPuqEMSfrGvEWYaDZWEPaHvGm6r?usp=sharing). Download and put it to `warpback/ecweight`, then you can directly use this dataloader to train your own view-synthesis method.
* `python warpback/stage2_dataset.py` to see a demo.

### World Coordinate System
* **X-axis**: point to **right**
* **Y-axis**: point to **down**
* **Z-axis**: point to **front** (a larger z value corresponds to a larger depth value)

### Note
To make the code simple and easy to read, we omit some engineering steps which can make the Warp-Back strategy performs better, and mark them in the code. These steps are:
* Disparity Preprocessing: You can use some kernel to filter the monocular depth estimation to reduce artifacts around depth discontinuities.
* Image Size Preprocessing: In the code we simply resize all the image to a fix size, e.g. 256 x 384; a better way is to crop a random patch with the same aspect ratio as the fix size, and then resize this patch to the fix size.
* Mask Preprocessing: To avoid artifacts on the image-warping mask, you can use some morphological operation to filter it before send it to the inpainting network.
