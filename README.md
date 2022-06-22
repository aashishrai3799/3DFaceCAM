
# Official implementation of Controllable 3D Generative Adversarial Face Model via Disentangling Shape and Appearance

Fariborz Teherkhani, Aashish Rai*, Shaunak Srivastava*, Quankai Gao*, Xuanbai Chen, Farnando de la Torre, Steven Song, Aayush Prakash, Daeil Kim

(* equal contribution)

### More details coming soon!


[[Project Page]](https://aashishrai3799.github.io/3DFaceCAM) [[Video]](#) [[Colab Demo]](#) [[Demo Code]](#) [[Arxiv]](#) 


## Installation 
See [`install.md`](docs/install.md)

## Quick Start 

- Testing
    ```
    python test.py
    ```

## Train your own model

### Preprocess data
    
    python preprocess.py (Coming Soon)
    

### Start training
```
# Shape

cd shape_model/

Train AE
python ae/train.py 

Train GAN
python 3dgan/train.py 

# Texture

cd texture_model/

Train P-GAN
python train.py --init_step 1 --batch_size 128

```




