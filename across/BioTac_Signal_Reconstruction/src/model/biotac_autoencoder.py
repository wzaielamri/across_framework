import torch
from omegaconf import OmegaConf
from lightning import LightningModule
from torch.optim import Adam


class BioTacAutoencoder(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        encoding_layers = [torch.nn.Linear(19, 256),
                           torch.nn.ReLU(),
                           torch.nn.Linear(256, 128),
                           torch.nn.ReLU(),
                           torch.nn.Linear(128, 64),
                           torch.nn.ReLU()]

        decoding_layers = [torch.nn.Linear(8, 64),
                           torch.nn.ReLU(),
                           torch.nn.Linear(64, 128),
                           torch.nn.ReLU(),
                           torch.nn.Linear(128, 256),
                           torch.nn.ReLU(),
                           torch.nn.Linear(256, 19, bias=False)]

        self.encoder = torch.nn.Sequential(*encoding_layers)

        self.decoder = torch.nn.Sequential(*decoding_layers)

        self.mu_layer = torch.nn.Linear(64, 8, bias=False)
        self.log_var_layer = torch.nn.Linear(64, 8, bias=False)

        self.current_kl_weight = 0.0
        if self.hparams.kl_annealing_goal_epoch < 1:
            self.current_kl_weight = self.hparams.kl_weight

    def encode(self, x):
        hidden_state = self.encoder(x)
        mu = self.mu_layer(hidden_state)
        log_var = self.log_var_layer(hidden_state)
        return mu, log_var

    def decode(self, latent_space):
        return self.decoder(latent_space)

    def reparameterize(self, mu, log_var):
        # std = torch.exp(sigma)
        # eps = torch.normal(torch.zeros(mu.shape, dtype=torch.float, device=self.device),
        #                   torch.ones(mu.shape, dtype=torch.float, device=self.device))
        # return eps * sigma + mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)

        latent_space = self.reparameterize(mu, log_var)
        return latent_space, mu, log_var

    def configure_optimizers(self):
        adam_optimizer = Adam(self.parameters(), lr=self.hparams.lr)  # TODO weight decay or lr decay?
        return adam_optimizer

    def training_step(self, batch, batch_idx):
        predict, ground_truth = batch

        mu, log_var = self.encode(predict)

        loss, kl_loss, mse_loss, _, latent_space = self.calculate_loss(mu,
                                                                       log_var,
                                                                       ground_truth)
        self.logger.experiment.log({"train_z": latent_space})
        self.logger.experiment.log({"train_mu_hist": mu})
        self.logger.experiment.log({"train_log_var_hist": log_var})
        self.log_dict({"train_loss": loss,
                       "train_kl_loss": kl_loss.detach(),
                       "train_mse_loss": mse_loss.detach(),
                       "train_log_var": torch.mean(log_var),
                       "train_mu": torch.mean(mu),
                       "train_kl_weight": self.current_kl_weight}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        predict, ground_truth = batch

        mu, log_var = self.encode(predict)

        loss, kl_loss, mse_loss, decoded_latent_space, latent_space = self.calculate_loss(mu,
                                                                                          log_var,
                                                                                          ground_truth)
        self.logger.experiment.log({"val_z": latent_space})
        self.logger.experiment.log({"val_mu_hist": mu})
        self.logger.experiment.log({"val_log_var_hist": log_var})
        self.log_dict({"val_loss": loss,
                       "val_ground_truth_elec_0": ground_truth[0][0],
                       "val_predicted_elec_0": decoded_latent_space[0][0],
                       "val_kl_loss": kl_loss.detach(),
                       "val_mse_loss": mse_loss.detach(),
                       "val_log_var": torch.mean(log_var),
                       "val_mu": torch.mean(mu)})

    def test_step(self, batch, batch_idx):
        predict, ground_truth = batch

        mu, log_var = self.encode(predict)

        loss, kl_loss, mse_loss, decoded_latent_space, latent_space = self.calculate_loss(mu,
                                                                                          log_var,
                                                                                          ground_truth)
        self.logger.experiment.log({"test_z": latent_space})
        self.logger.experiment.log({"test_mu_hist": mu})
        self.logger.experiment.log({"test_log_var_hist": log_var})
        self.log_dict({"test_loss": loss,
                       "test_ground_truth_elec_0": ground_truth[0][0],
                       "test_predicted_elec_0": decoded_latent_space[0][0],
                       "test_kl_loss": kl_loss.detach(),
                       "test_mse_loss": mse_loss.detach(),
                       "test_log_var": torch.mean(log_var),
                       "test_mu": torch.mean(mu)})

    def calculate_loss(self, mu, log_var, ground_truth):
        complete_mse_loss = torch.zeros(ground_truth.shape[0], requires_grad=True, dtype=torch.float,
                                        device=self.device)

        decoded_latent_space = None
        for i in range(self.hparams.L):
            latent_space = self.reparameterize(mu, log_var)

            decoded_latent_space = self.decode(latent_space)

            mse_loss = torch.mean(torch.nn.functional.mse_loss(decoded_latent_space, ground_truth, reduction="none"),
                                  dim=1)
            complete_mse_loss = complete_mse_loss + mse_loss

        complete_mse_loss = complete_mse_loss / self.hparams.L
        # kl_loss = -0.5 * torch.sum(1 + torch.log(log_var.exp()) - mu.exp() - log_var.exp())

        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # kl_norm = (latent_space.shape[1] / ground_truth.shape[1])
        # kl_loss = kl_loss  # * kl_norm

        if self.hparams.kl_weight == 0:
            loss = torch.mean(complete_mse_loss)
        else:
            loss = torch.mean(complete_mse_loss) + self.current_kl_weight * kl_loss # * kl_norm

        return (loss,
                kl_loss,
                torch.mean(complete_mse_loss),
                decoded_latent_space,
                latent_space)

    def on_train_epoch_end(self):
        if self.hparams.kl_annealing_goal_epoch < 1:
            return
        self.current_kl_weight += self.hparams.kl_weight / self.hparams.kl_annealing_goal_epoch
        if self.current_kl_weight > self.hparams.kl_weight:
            self.current_kl_weight = self.hparams.kl_weight


if __name__ == '__main__':
    hparams = OmegaConf.create({"kl_weight": 1e-5,
                                "lr": 1e-4,
                                "batch_size": 128,
                                "L": 1,
                                "kl_annealing_goal_epoch": 0})
    model = BioTacAutoencoder(hparams)
    test_batch = torch.ones((128, 19))
    test_loss = model.training_step((test_batch, test_batch.clone()), 1)
    val_loss = model.validation_step((test_batch, test_batch.clone()), 1)
