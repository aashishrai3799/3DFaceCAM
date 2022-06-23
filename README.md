
# Official implementation of Controllable 3D Generative Adversarial Face Model via Disentangling Shape and Appearance

Fariborz Teherkhani, Aashish Rai*, Shaunak Srivastava*, Quankai Gao*, Xuanbai Chen, Fernando de la Torre, Steven Song, Aayush Prakash, Daeil Kim

(* equal contribution)

### More details coming soon!


[[Project Page]](https://aashishrai3799.github.io/3DFaceCAM) [[Video]](#) [[Colab Demo]](#) [[Demo Code]](#) [[Arxiv]](#) 


## Install packages 
See [`install.md`](docs/install.md)

## Testing

Download pre-trained weights and put the "checkpoints" folder in the main directory. [[Link]](https://drive.google.com/file/d/1hK31wVAoieRiVFydPxnx0MVpx6AnWN1-/view?usp=sharing)

- Generate 3D Faces (mesh and texture)
    ```
    python generate_faces.py
    ```
    
- Generate meshes only
    ```
    python test_gan3d.py
    ```
    
- Generate textures only
    ```
    python test_texture.py
    ```

## Train your own model

### Preprocess data

    python preprocess.py (Coming Soon)
    

### Start training

- Shape
    ```
    Train AE
    python train_ae.py 
    ```
    ```
    Train GAN
    python train_gan3d.py 
    ```

- Texture
    ```
    Train P-GAN
    python train_texture.py --init_step 1 --batch_size 128
    ```




