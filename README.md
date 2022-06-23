
# Official implementation of Controllable 3D Generative Adversarial Face Model via Disentangling Shape and Appearance

Fariborz Teherkhani, Aashish Rai*, Shaunak Srivastava*, Quankai Gao*, Xuanbai Chen, Fernando de la Torre, Steven Song, Aayush Prakash, Daeil Kim

(* equal contribution)

### More details coming soon!


[[Project Page]](https://aashishrai3799.github.io/3DFaceCAM) [[Video]](#) [[Colab Demo]](#) [[Demo Code]](#) [[Arxiv]](#) 


## Installation 
See [`install.md`](docs/install.md)

## Quick Start 

- Testing
    ```
    python generate_faces.py
    ```

## Train your own model

### Preprocess data
    
    python preprocess.py (Coming Soon)
    

### Start training
```
# Shape

Train AE
python train_ae.py 

Train GAN
python train_gan3d.py 

# Texture

Train P-GAN
python train_texture.py --init_step 1 --batch_size 128

```




