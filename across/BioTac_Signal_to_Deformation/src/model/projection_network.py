import torch.nn
import lightning
from torch.optim import Adam


class NetworkWrapper:
    def __init__(self, model):
        self.model = model


class ProjectionNetworkBiotac(lightning.LightningModule):
    def __init__(self, hparams, reconstruction_model=None, norm=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.inverse = bool(self.hparams.inverse)
        self.reconstruction_model = NetworkWrapper(reconstruction_model)
        self.norm = norm

        if not self.inverse:
            num_layers = hparams.get("num_layers")
            num_neurons = [hparams.get(f"num_neurons_layer_{i}") for i in range(num_layers)]
            activation_functions = [hparams.get(f"activation_layer_{i}") for i in range(num_layers)]
            dropouts = [hparams.get(f"dropout_layer_{i}") for i in range(num_layers)]
            layers = []
            for i in range(num_layers):
                if i == 0:
                    layers.append(torch.nn.Linear(128, num_neurons[i]))
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
            layers.append(torch.nn.Linear(num_neurons[-1], 8))
            self.network = torch.nn.Sequential(*layers)
        else:
            num_layers = hparams.get("num_layers")
            num_neurons = [hparams.get(f"num_neurons_layer_{i}") for i in range(num_layers)]
            activation_functions = [hparams.get(f"activation_layer_{i}") for i in range(num_layers)]
            dropouts = [hparams.get(f"dropout_layer_{i}") for i in range(num_layers)]
            layers = []
            for i in range(num_layers):
                if i == 0:
                    layers.append(torch.nn.Linear(8, num_neurons[i]))
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
            layers.append(torch.nn.Linear(num_neurons[-1], 128))
            self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        adam_optimizer = Adam(self.parameters(), lr=self.hparams.lr,
                              weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer, step_size=1, gamma=self.hparams.get("lr_decay"))
        return {"optimizer": adam_optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1,
                                 "reduce_on_plateau": False, "monitor": "val_loss"}}

    def training_step(self, batch, batch_idx):
        value, deformation = batch

        if self.inverse:
            res = self.forward(value)
            target = deformation
        else:
            res = self.forward(deformation)
            target = value

        loss = self.calculate_loss(res, target)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        value, deformation = batch

        if self.inverse:
            res = self.forward(value)
            target = deformation
        else:
            res = self.forward(deformation)
            target = value

        loss = self.calculate_loss(res, target)
        self.log("val_loss", loss)

        if self.reconstruction_model.model is not None:
            res = self.reconstruction_model.model.decode(res)
            target = self.reconstruction_model.model.decode(target)

        if self.norm is not None:
            res = res * self.norm["std"].to(self.device) + self.norm["mean"].to(self.device)
            target = target * self.norm["std"].to(self.device) + self.norm["mean"].to(self.device)

        interpretable_loss = torch.sqrt(self.calculate_loss(res, target)) * 1000
        self.log("val_interpretable_loss", interpretable_loss)

        return loss

    def test_step(self, batch, batch_idx):
        value, deformation = batch

        if self.inverse:
            res = self.forward(value)
            target = deformation
        else:
            res = self.forward(deformation)
            target = value

        loss = self.calculate_loss(res, target)
        self.log("test_loss", loss)

        if self.reconstruction_model.model is not None:
            res = self.reconstruction_model.model.decode(res)
            target = self.reconstruction_model.model.decode(target)

        if self.norm is not None:
            res = res * self.norm["std"].to(self.device) + self.norm["mean"].to(self.device)
            target = target * self.norm["std"].to(self.device) + self.norm["mean"].to(self.device)

        interpretable_loss = torch.sqrt(self.calculate_loss(res, target)) * 1000

        self.log("test_interpretable_loss", interpretable_loss)

        return loss

    def calculate_loss(self, res, target):
        loss = torch.nn.functional.mse_loss(res, target)
        # print(loss)
        return loss


if __name__ == '__main__':
    conf = {
        "lr": 1e-3,
        "inverse": True,
        "weight_decay": 0.99
    }

    model = ProjectionNetworkBiotac(conf)
    inp = torch.ones(8)

    print(model.forward(inp))
