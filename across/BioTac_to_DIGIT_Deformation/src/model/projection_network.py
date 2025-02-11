import torch.nn
from lightning import LightningModule
from torch.optim import Adam
from omegaconf import DictConfig
from across.BioTac_Signal_to_Deformation.src.model.projection_network import NetworkWrapper

class ProjectionNetworkDeformation(LightningModule):
    def __init__(self, hparams, reconstruction_model = None,):
        super().__init__()

        num_layers = hparams.get("num_layers")
        z_input = hparams.get("z_input")
        z_output = hparams.get("z_output")
        num_neurons = [hparams.get(f"num_neurons_layer_{i}") for i in range(num_layers)]
        activation_functions = [hparams.get(f"activation_layer_{i}") for i in range(num_layers)]
        dropouts = [hparams.get(f"dropout_layer_{i}") for i in range(num_layers)]
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(torch.nn.Linear(z_input, num_neurons[i]))
            else:
                layers.append(torch.nn.Linear(num_neurons[i - 1], num_neurons[i]))
            if activation_functions[i] == "ELU":
                layers.append(torch.nn.ELU())
            elif activation_functions[i] == "ReLU":
                layers.append(torch.nn.ReLU())
            elif activation_functions[i] == "LeakyReLU":
                layers.append(torch.nn.LeakyReLU())
            elif activation_functions[i] == "Sigmoid":
                layers.append(torch.nn.Sigmoid())
            elif activation_functions[i] == "Tanh":
                layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Dropout(dropouts[i]))
        layers.append(torch.nn.Linear(num_neurons[-1], z_output))
        self.network = torch.nn.Sequential(*layers)

        self.fem_vae_model = NetworkWrapper(reconstruction_model)


        # hparams omegaconf to dict 
        try: # in the case it is OmegaConfig
            hparams = DictConfig(hparams) # to avoid any issues with the hparams save
        except: # in the case it is ConfigSpace.configuration_space.Configuration
            hparams = hparams.get_dictionary()

        self.save_hyperparameters(hparams)


    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        adam_optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr"),
                                          weight_decay=self.hparams.get("weight_decay"), )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer, step_size=1, gamma=self.hparams.get("lr_decay"))
        return ({"optimizer": adam_optimizer,
                 "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1,
                                  "reduce_on_plateau": False, "monitor": "val_loss"}})

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        #self.logger.experiment.log({"train_target": y,"train_out": y_hat,})
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)

        # decode the target mesh
        #with torch.no_grad():
        #    target_mesh = self.fem_vae_model.decode(y)
        #    target_mesh_hat = self.fem_vae_model.decode(y_hat)
        #    loss_mesh = torch.nn.functional.mse_loss(target_mesh_hat, target_mesh)
        #    self.log("train_loss_mesh", loss_mesh, sync_dist=True)

        return loss

    def update_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

    def validation_step(self, batch, batch_idx):
        x, y, = batch
        y_hat = self.forward(x)
        #self.logger.experiment.log({"val_target": y,"val_out": y_hat,})
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        # decode the target mesh
        with torch.no_grad():
            if self.fem_vae_model.model is not None:
                target_mesh = self.fem_vae_model.model.decode(y)
                target_mesh_hat = self.fem_vae_model.model.decode(y_hat)
                loss_mesh = torch.nn.functional.mse_loss(target_mesh_hat, target_mesh)
                self.log("val_loss_mesh", loss_mesh, sync_dist=True)

                out_rescaled = target_mesh_hat.reshape(target_mesh_hat.shape[0], -1, 3) * self.std.to(
                    target_mesh_hat.device) + self.mean.to(target_mesh_hat.device)
                target_rescaled = target_mesh.reshape(target_mesh.shape[0], -1, 3) * self.std.to(
                    target_mesh.device) + self.mean.to(target_mesh.device)
                rmse = torch.sqrt(torch.nn.functional.mse_loss(out_rescaled, target_rescaled))
                self.log("val_rmse", rmse * 1000, sync_dist=True)

        return loss
