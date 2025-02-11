from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from lightning import LightningModule
from psbody.mesh import Mesh

from across.Mesh_Reconstruction.src.model.operations.mesh_operations import generate_transform_matrices, scipy_to_torch_sparse
from across.Mesh_Reconstruction.src.model.operations.pooling_layer import Pool, ChebConv_Coma
import os
import h5py as h

from across.Mesh_Reconstruction.src.utils.convert_tet_to_msh import conv_tet_to_vertices_and_faces



class fem_VAE(LightningModule):
    # This model is the same as Berkeley_Autoencoder but with two dense layers in the encoder and decoder
    def __init__(self, hparams, mean, std):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.mean = mean
        self.std = std
        

        vertices, faces = conv_tet_to_vertices_and_faces(self.hparams.mesh_path)
        template_mesh = Mesh(v=vertices, f=faces)
        self.template_mesh_vertices = template_mesh.v
        self.template_mesh_faces = template_mesh.f
        meshes, adjacency_matrices, self.downsample_matrices, self.upsample_matrices = generate_transform_matrices(
            template_mesh, [2, 2, 2, 2]) 


        self.downsample_matrices = [
            scipy_to_torch_sparse(m) for m in self.downsample_matrices
        ]
        self.downsample_matrices = [mat for mat in self.downsample_matrices]
        self.upsample_matrices = [
            scipy_to_torch_sparse(m) for m in self.upsample_matrices
        ]
        self.upsample_matrices = [mat for mat in self.upsample_matrices]
        adjacency_matrices = [
            scipy_to_torch_sparse(m) for m in adjacency_matrices
        ]
        adjacency_matrices = [mat for mat in adjacency_matrices]

        num_nodes = [len(meshes[i].v) for i in range(len(meshes))]

        print("Generated transformation matrices")
        self.n_layers = 4
        self.L= self.hparams.get("L")
        self.filters =[3, 16, 16, 16, 32, 32]
        self.z = self.hparams.get("z")
        self.K = self.hparams.get("cheby_order")
        self.adjacency_matrices = adjacency_matrices

        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])

        self.cheb = torch.nn.ModuleList([ChebConv_Coma(self.filters[i], self.filters[i+1], self.K)
                                         for i in range(len(self.filters)-2)])
        self.cheb_dec = torch.nn.ModuleList([ChebConv_Coma(self.filters[-i-1], self.filters[-i-2], self.K)
                                             for i in range(len(self.filters)-1)])
        self.cheb_dec[-1].bias = None  # No bias for last convolution layer
        self.pool = Pool()
        self.relu = torch.nn.ReLU()
        self.leakyrelu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters[-1], self.z*4)
        self.enc_mu_lin = torch.nn.Linear(self.z*4, self.z)
        self.enc_logvar_lin = torch.nn.Linear(self.z*4, self.z)
        self.dec_lin_first = torch.nn.Linear(self.z, self.z*4)
        self.dec_lin = torch.nn.Linear(self.z*4, self.filters[-1]*self.upsample_matrices[-1].shape[1])
        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.enc_logvar_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.enc_mu_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin_first.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)

    def encode(self, x):
        for i in range(self.n_layers):
            x = self.cheb[i](x, self.A_edge_index[i].to(x.device), self.A_norm[i].to(x.device))
            x = self.relu(x)
            x = self.pool(x, self.downsample_matrices[i].to(x.device))
        x = x.reshape(x.shape[0], self.enc_lin.in_features)
        x = self.relu(self.enc_lin(x))
        mu = self.enc_mu_lin(x)  
        logvar = self.enc_logvar_lin(x)
        return mu, logvar

    def decode(self, x):
        x = self.relu(self.dec_lin_first(x))
        x = self.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(self.n_layers):
            x = self.pool(x, self.upsample_matrices[-i-1].to(x.device))
            x = self.relu(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1].to(x.device), self.A_norm[self.n_layers-i-1].to(x.device)))
        x = self.cheb_dec[-1](x, self.A_edge_index[0].to(x.device), self.A_norm[0].to(x.device))
        return x

    def reparameterize(self, mu, logvar):
        #from pytorch examples: https://github.com/pytorch/examples/blob/main/vae/main.py#L47
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.filters[0])
        mu,logvar = self.encode(x)
        x_list=[]
        z_list=[]

        for i in range(self.L):
            z = self.reparameterize(mu, logvar)
            x = self.decode(z)
            x = x.reshape(-1, self.filters[0])
            x_list.append(x)
            z_list.append(z)
        return x_list, z_list, mu, logvar
    
    
    def configure_optimizers(self):
        adam_optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr"), weight_decay=self.hparams.get("weight_decay"),)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer, step_size=1, gamma=self.hparams.get("lr_decay"))
        return ({"optimizer": adam_optimizer, "lr_scheduler": {"scheduler": lr_scheduler ,"interval": "epoch", "frequency": 1, "reduce_on_plateau": False, "monitor": "val_loss"}})


    def training_step(self, batch, batch_idx):
        #batch.x=batch.x.double()
        #batch.y=batch.y.double()
        out_list, z_list, mu, logvar = self.forward(batch)
        # wandb log mu values
        self.logger.experiment.log({"train_mu_hist": mu, "train_logvar_hist": logvar, "train_std_hist": torch.exp(0.5*logvar), "train_z": z_list[0]})

        #print("mu: min", mu.min().item(), "max", mu.max().item(), "mean", mu.mean().item(), "std", mu.std().item())
        #print("logvar: min:", logvar.min().item(), "max",logvar.max().item(), "mean", logvar.mean().item(), "std", logvar.std().item())
        #print("z: min:", z.min().item(), "max",z.max().item(), "mean", z.mean().item(), "std", z.std(dim=1).mean().item() )
        #print("out: min", out.min().item(), "max", out.max().item(), "mean", out.mean().item(), "std", out.std().item())
        mse_loss = torch.zeros(1, device=out_list[0].device)
        for l in range(self.L):
            mse_loss += torch.nn.functional.mse_loss(out_list[l], batch.y)
        mse_loss = mse_loss/self.L 
        #print("mse_loss: ", mse_loss.item())
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        #print("kl_loss: ", kl_loss.item())
        loss = mse_loss + kl_loss * self.hparams.get("kl_weight")

        self.log_dict({"train_loss": loss}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log_dict({"train_mse_loss": mse_loss,}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log_dict({"train_kl_loss": kl_loss}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log_dict({"train_mu": z_list[0].mean(dim=1).mean()}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log_dict({"train_std": z_list[0].std(dim=1).mean()}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)

        with torch.no_grad():
            out_rescaled = out_list[0].reshape(batch.num_graphs, -1, 3) * self.std.to(out_list[0].device)
            batch_y_rescaled = batch.y.reshape(batch.num_graphs, -1, 3) * self.std.to(out_list[0].device)
            rmse = torch.sqrt(torch.nn.functional.mse_loss(out_rescaled, batch_y_rescaled))
            l1_loss = torch.nn.functional.l1_loss(out_rescaled, batch_y_rescaled)
            euclid_dist =  (out_rescaled - batch_y_rescaled).pow(2).sum(-1).sqrt().max(1).values.mean()
        self.log_dict({"train_rmse": rmse*1000,}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log_dict({"train_l1_loss": l1_loss*1000,}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log_dict({"train_euclid": euclid_dist*1000,}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        
        return loss
        # only on first gpu
        if self.global_rank == 0 and batch_idx % 10 == 0:
            save_out = out.detach().cpu().numpy()
            save_out = save_out
            expected_out = (batch.y.detach().cpu().numpy())
            vertices_count = save_out.shape[0]//batch.num_graphs
            save_out = save_out[:vertices_count,:]
            expected_out = expected_out[:vertices_count,:]
            save_out = save_out*self.std.numpy()+self.mean.numpy()
            expected_out = (expected_out)*self.std.numpy()+self.mean.numpy()
            
            self.save_obj_file(vertices=save_out, faces=self.template_mesh_faces, filename=os.path.join("visual_output", f'{self.hparams.get("sensor_name")}{batch_idx}_{self.hparams.get("note")}_predicted_{self.hparams.get("seed")}.obj'))
            self.save_obj_file(vertices=expected_out, faces=self.template_mesh_faces, filename=os.path.join("visual_output", f'{self.hparams.get("sensor_name")}{batch_idx}_{self.hparams.get("note")}_target_{self.hparams.get("seed")}.obj'))
        
        return loss

    def validation_step(self, batch, batch_idx):
        #batch.x=batch.x.double()
        #batch.y=batch.y.double()
        out_list, z_list, mu, logvar = self.forward(batch)
        self.logger.experiment.log({"val_mu_hist": mu, "val_logvar_hist": logvar, "val_std_hist": torch.exp(0.5*logvar), "val_z": z_list[0]})

        mse_loss = torch.zeros(1, device=out_list[0].device)
        for l in range(self.L):
            mse_loss += torch.nn.functional.mse_loss(out_list[l], batch.y)
        mse_loss = mse_loss/self.L 
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = mse_loss + kl_loss * self.hparams.get("kl_weight")
        self.log_dict({"val_loss": loss}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log_dict({"val_mse_loss": mse_loss,}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log_dict({"val_kl_loss": kl_loss}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log_dict({"val_mu": z_list[0].mean(dim=1).mean()}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log_dict({"val_std":  z_list[0].std(dim=1).mean()}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)

        with torch.no_grad():
            out_rescaled= out_list[0].reshape(batch.num_graphs, -1, 3) * self.std.to(out_list[0].device)
            batch_y_rescaled = batch.y.reshape(batch.num_graphs, -1, 3) * self.std.to(out_list[0].device)
            rmse = torch.sqrt(torch.nn.functional.mse_loss(out_rescaled, batch_y_rescaled))
            l1_loss = torch.nn.functional.l1_loss(out_rescaled, batch_y_rescaled)
            euclid_dist =  (out_rescaled - batch_y_rescaled).pow(2).sum(-1).sqrt().max(1).values.mean()
        self.log_dict({"val_rmse": rmse*1000,}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)   # FIXME: ADJUST
        self.log_dict({"val_l1_loss": l1_loss*1000,}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log_dict({"val_euclid": euclid_dist*1000,}, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)

        # only on first gpu
        if self.global_rank == 0 and batch_idx % 10 == 0:
            save_out = out_list[0].detach().cpu().numpy()
            save_out = save_out
            expected_out = (batch.y.detach().cpu().numpy())
            vertices_count = save_out.shape[0]//batch.num_graphs
            save_out = save_out[:vertices_count,:]
            expected_out = expected_out[:vertices_count,:]
            save_out = save_out*self.std.numpy()+self.mean.numpy()
            expected_out = (expected_out)*self.std.numpy()+self.mean.numpy()

            self.save_obj_file(vertices=save_out, faces=self.template_mesh_faces, filename=os.path.join("visual_output", f'{self.hparams.get("sensor_name")}{batch_idx}_{self.hparams.get("note")}_predicted_{self.hparams.get("seed")}.obj'))
            self.save_obj_file(vertices=expected_out, faces=self.template_mesh_faces, filename=os.path.join("visual_output", f'{self.hparams.get("sensor_name")}{batch_idx}_{self.hparams.get("note")}_target_{self.hparams.get("seed")}.obj'))
        return loss


    def update_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

    def predict_step(self, batch, batch_idx):
        out, z, mu, logvar = self.forward(batch)
        return out, z, mu, logvar

        # save the output
        if self.global_rank == 0 and batch_idx < 50:
            save_out = out.detach().cpu().numpy()
            save_out = save_out
            expected_out = (batch.y.detach().cpu().numpy())
            vertices_count = save_out.shape[0]//batch.num_graphs
            save_out = save_out[:vertices_count,:]
            expected_out = expected_out[:vertices_count,:]
            save_out = save_out*self.std.numpy()+self.mean.numpy()
            expected_out = (expected_out)*self.std.numpy()+self.mean.numpy()
            split="train"
            self.save_obj_file(vertices=save_out, faces=self.template_mesh_faces, filename=os.path.join("visual_output", f'{split}_{self.hparams.get("sensor_name")}{batch_idx}_{self.hparams.get("note")}_predicted_{self.hparams.get("seed")}.obj'))
            self.save_obj_file(vertices=expected_out, faces=self.template_mesh_faces, filename=os.path.join("visual_output", f'{split}_{self.hparams.get("sensor_name")}{batch_idx}_{self.hparams.get("note")}_target_{self.hparams.get("seed")}.obj'))
        if batch_idx > 50 :
            exit()
        return out, z


    def save_obj_file(self, vertices, faces, filename):
        with open(filename, 'w') as f:
            for v in vertices:
                f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for p in faces:
                f.write('f %d %d %d\n' % (p[0]+1, p[1]+1, p[2]+1))




if __name__ == '__main__':
    with h.File(Path("../../data/biotac_isaacgym_dataset/results.hdf5").resolve(), "r") as f:
        tmp = f["step_0"]["nodal_coords"]["env_0"][:]

    print(tmp.shape)
    tmp = torch.tensor(tmp, dtype=torch.float)
    tmp = torch.unsqueeze(tmp, 0)

    conf = OmegaConf.create({"kl_weight": 1e-9,
                             "L": 1,
                             "val_size": 1,
                             "train_size": 1,
                             "lr": 1e-3,
                             "batch_size": 64,
                             "template_mesh_path": str(Path("../../data/template_mesh.tet").resolve()),
                             "device": "cpu"})

    model = fem_VAE(conf)
    print("Do forward")
    hidden_state = model.forward(tmp)
    print(hidden_state)
    reconstruction = model.decode(hidden_state)
    print(reconstruction)
    print("Finished")
