import  os
import sys
import yaml
sys.path.append(".")

import torch
import torch.nn as nn

import  numpy as np

from mesh_obj import mesh_obj
from architectures import Decoder, Generator
from utils import vertex_to_mesh, get_template_verts, load_label
from train import shape_GAN

class Test:
    def __init__(self,gan_model):
        self.gan = gan_model
        self.result_path = self.gan.cfg["result_path"]
        self.decoder = None
        self.label_dict = load_label()
        self.device = self.gan.device
        self.template_verts = torch.from_numpy(get_template_verts()).to(self.device)
        self.face_v = torch.from_numpy(np.load('./data/face_v.npy')).to(dtype=torch.float32).to(self.device).unsqueeze(0) - 1

        self.z_noise = torch.randn(1,5).to(self.device)
        self.z_id = torch.randn(1,20).to(self.device)
        self.z_id_style = torch.randn(1,20).to(self.device)
        self.z_exp = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=20).to(self.device)


    def set_z_noise(self,z_noise):
        self.z_noise = z_noise


    def set_z_id(self,z_id):
        self.z_id = z_id


    def set_z_id_style(self,z_id_style):
        self.z_id_style = z_id_style


    def set_z_exp(self,z_exp):
        self.z_exp = z_exp


    def load_models(self, path_d, path_gid, path_gexp):
        self.decoder = Decoder(num_features=130, output_size=78951)
        self.decoder.load_state_dict(torch.load(path_d, map_location=self.device))
        self.decoder = self.decoder.to(self.device)

        self.gan.generator.generator_id.load_state_dict(torch.load(path_gid, map_location=self.device))
        self.gan.generator.generator_exp.load_state_dict(torch.load(path_gexp, map_location=self.device))


    def generate_emb(self, z_noise, z_id, z_id_style, z_exp):
        test_noise_id = torch.cat((z_noise, z_id), 1)
        test_noise_ex = torch.cat((z_noise, z_id_style, z_exp), 1)
        emb_id, emb_ex = self.gan.generator(test_noise_id, test_noise_ex)

        return emb_id, emb_ex


    def decode_mesh(self, emb_id, emb_ex):
        z = torch.cat((emb_id,emb_ex), 1)
        return (self.decoder(z)+ self.template_verts).reshape(26317,3)


    def generate_intensity(self, filename, low=0, high=1.5, num_steps=15, exp_list=range(0,20), save_obj=True, render=False):
        for i in exp_list:
            subfolder = os.path.join(self.result_path, filename, str(i)+'_'+self.label_dict[i])
            z_exp = torch.nn.functional.one_hot(torch.tensor([i]), num_classes=20).to(self.device)

            meshes = []
            for count,level in enumerate(np.linspace(low,high,15)):
                emb_id, emb_ex = self.generate_emb(self.z_noise,self.z_id,self.z_id_style, (level*z_exp))
                mesh_verts = self.decode_mesh(emb_id, emb_ex)
                meshes.append(mesh_verts)

                if save_obj:
                    vertex_to_mesh(mesh_verts, count, subfolder)
            if render:
                from renderer_pt3d import renderfaces
                renderfaces(torch.stack(meshes).float(), self.face_v, subfolder+'.png',self.device)


    def generate(self, filename, intensities=False, low=0, high=1.5, num_steps=15, exp_list=range(0,20), save_obj=True, render=False):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        if not intensities:
            meshes = []
            for i in exp_list:
                subfolder = os.path.join(self.result_path, filename)
                z_exp = torch.nn.functional.one_hot(torch.tensor([i]), num_classes=20).to(self.device)

                emb_id, emb_ex = self.generate_emb(self.z_noise, self.z_id, self.z_id_style,z_exp)
                mesh_verts = self.decode_mesh(emb_id, emb_ex)
                meshes.append(mesh_verts)

                if save_obj:
                    vertex_to_mesh(mesh_verts, str(i)+'_'+self.label_dict[i], subfolder)
            if render:
                from renderer_pt3d import renderfaces
                renderfaces(torch.stack(meshes).float(), self.face_v, subfolder+'image.png',self.device)
        else:
            self.generate_intensity(filename=filename,low=low,high=high,num_steps=num_steps,exp_list=exp_list,save_obj=save_obj)


if __name__ == '__main__':
    device = 'cuda:0'
    torch.device(device)

    with open("gan3d/config.yml","r") as cfgfile:
        cfg = yaml.safe_load(cfgfile)

    gan = shape_GAN(cfg, device)

    path_d = './ae/checkpoints/Dec/2000'
    path_gid = os.path.join(gan.folder, 'Generator_Checkpoint_id/8.0')
    path_gexp = os.path.join(gan.folder, 'Generator_Checkpoint_exp/8.0')

    with torch.no_grad():
        test = Test(gan)
        test.load_models(path_d=path_d, path_gid=path_gid, path_gexp=path_gexp)

        for i in range(10):
            test.generate(str(i), save_obj=True, render=False)
            
            
            
            
