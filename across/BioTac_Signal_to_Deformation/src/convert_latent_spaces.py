from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from across.BioTac_Signal_Reconstruction.src.model.biotac_autoencoder import BioTacAutoencoder
from across.BioTac_Signal_to_Deformation.src.dataset_loader.biotac_value_to_def_data_loader import \
    BioTacValueDeformationModule
from across.Mesh_Reconstruction.src.model.fem_autoencoder import fem_VAE


class FEMNetworkInput:
    def __init__(self, data):
        self.x = data
        self.num_graphs = len(self.x)
        self.edge_index = None


def filter_data(data, deformation_network, value_network, def_norm, indenter_filter=None, depth_filter=0.002):
    latent_value_data = []
    latent_deformation_data = []
    real_value_data = []
    real_deformation_data = []
    rejects_contacts = []
    rejects_indenter_types = []
    indenter_types = []
    trajects = []
    active_trajec = []
    contact_points = []
    deformation_mu = []
    deformation_logvar = []
    value_mu = []
    value_logvar = []
    mse_list = []

    prev = None
    prev_i = 0
    index = 0
    norm = def_norm

    # get surface indices
    #surface_indeces = np.load("biotac_vertices_surface_indices.npy")

    # offset = 1.39700000e-2
    depth = []
    for index, i in tqdm.tqdm(enumerate(data), total=len(data)):
        def_res = deformation_network.forward(FEMNetworkInput(i[1].detach().to(deformation_network.device)))
        value_res = value_network.forward(i[0].detach().to(value_network.device))

        value_loss = torch.sqrt(
            torch.nn.functional.mse_loss(value_network.decode(value_res[0]).to("cpu"), i[0].to("cpu")))
        v = torch.sqrt(
            torch.nn.functional.mse_loss(def_res[0][0].to("cpu") * norm["std"], i[1][0] * norm["std"])) * 1000

        t_mesh = torch.tensor(deformation_network.template_mesh_vertices, dtype=torch.float)
        #euc_dist = torch.norm(t_mesh[surface_indeces] - (i[1] * norm["std"] + norm["mean"])[:, surface_indeces], dim=2,p=2)

        error = i[4] > depth_filter

        if v.isnan() or value_loss.isnan() or torch.any(error):
            rejects_contacts.append(i[2])
            rejects_indenter_types.append(i[3])
            continue

        if indenter_filter is not None:
            if i[3] != indenter_filter:
                rejects_contacts.append(i[2])
                rejects_indenter_types.append(i[3])
                continue

        index += 1

        if not active_trajec:
            active_trajec.append(index)
            prev = i[2]
            prev_i = i[3]
        else:
            dist = np.linalg.norm(i[2].detach().numpy() - prev.detach().numpy())
            if (i[3] != prev_i) or (dist > 5e-4):
                trajects.append(torch.tensor(active_trajec))
                active_trajec = [index]
                prev = i[2]
                prev_i = i[3]
            else:
                active_trajec.append(index)
                prev = i[2]
                prev_i = i[3]

        latent_value_data.append(value_res[0].to("cpu"))
        value_mu.append(value_res[1].to("cpu"))
        value_logvar.append(value_res[2].to("cpu"))

        latent_deformation_data.append(def_res[1][0].to("cpu"))
        deformation_mu.append(def_res[2].to("cpu"))
        deformation_logvar.append(def_res[3].to("cpu"))

        real_value_data.append(i[0].to("cpu"))
        real_deformation_data.append(i[1].to("cpu"))

        indenter_types.append(i[3].to("cpu"))
        contact_points.append(i[2].to("cpu"))
        mse_list.append(v.detach().numpy())
        depth.append(i[4])

    #depth = torch.tensor(depth)
    #print(torch.unique(depth, return_counts=True))

    # return None, None

    trajects.append(torch.tensor(active_trajec))
    test_dict = {
        "value": torch.cat(latent_value_data),
        "deformation": torch.cat(latent_deformation_data),
        "real_value": torch.cat(real_value_data),
        "real_deformation": torch.cat(real_deformation_data),
        "trajects": trajects,
        "indenter_types": torch.cat(indenter_types),
        "contact_points": torch.cat(contact_points),
        "value_mu": torch.cat(value_mu),
        "value_logvar": torch.cat(value_logvar),
        "deformation_mu": torch.cat(deformation_mu),
        "deformation_logvar": torch.cat(deformation_logvar),
        "depth": torch.cat(depth),
    }
    if len(rejects_contacts) > 0:
        test_reject_dict = {
            "contacts": torch.cat(rejects_contacts),
            "indenter_type": torch.cat(rejects_indenter_types)
        }
    else:
        test_reject_dict = {}

    mse_list = np.array(mse_list)

    print(f"MSE Min {np.min(mse_list)}")
    print(f"MSE Max {np.max(mse_list)}")
    print(f"MSE Loss: {np.mean(mse_list)}")
    return test_dict, test_reject_dict


@hydra.main(version_base=None, config_path="../configs", config_name="collect_latent_spaces")
def main(cfg):
    print("Start")
    deformation_norm_dict = torch.load(cfg["checkpoints"]["b_deformation"]["norm"])
    deformation_norm_s = np.array([deformation_norm_dict['mean'].numpy(), deformation_norm_dict['std'].numpy()])

    value_norm = torch.load(cfg["checkpoints"]["b_signal"]["norm"])
    value_norm_s = np.array([value_norm['mean'].numpy(), value_norm['std'].numpy()])

    module = BioTacValueDeformationModule(
        data_dirs=cfg["datasets"]["data"]["train"],
        test_data_dir=cfg["datasets"]["data"]["test"],
        value_norm=value_norm_s,
        deformation_norm=deformation_norm_s,
        batch_size=1
    )
    print("Prepare Data")
    module.prepare_data()
    print("Setup")
    module.setup()

    deformation_network = fem_VAE.load_from_checkpoint(
        # "../BioTac_to_DIGIT_Deformation/src/checkpoints/biotac_batch_size_128_kl_weight_0.005_cheb_order_6:z_128_lr_0.001_L_1_seed_76979_fem_VAE_allData/epoch=299-step=288300.ckpt",
        cfg["checkpoints"]["b_deformation"]["model"],
        mean=0, std=1, z=128)

    value_network = BioTacAutoencoder.load_from_checkpoint(
        cfg["checkpoints"]["b_signal"]["model"])

    print("Loaded model")

    train_data = module.train_dataloader()
    val_data = module.val_dataloader()
    test_data = module.test_dataloader()
    device = "cuda:0"

    deformation_network.to(device)
    value_network.to(device)
    deformation_network.eval()
    value_network.eval()

    with torch.no_grad():
        print("Train Data Conversion")
        train_dict, train_reject_dict = filter_data(train_data, deformation_network, value_network,
                                                    def_norm=deformation_norm_dict,
                                                    indenter_filter=cfg["settings"]["id_filter"],
                                                    depth_filter=cfg["settings"]["depth_filter"])

        print("Validation Data Conversion")
        val_dict, val_reject_dict = filter_data(val_data, deformation_network, value_network,
                                                def_norm=deformation_norm_dict,
                                                indenter_filter=cfg["settings"]["id_filter"],
                                                depth_filter=cfg["settings"]["depth_filter"]
                                                )

        print("Test Data Conversion")
        test_dict, test_reject_dict = filter_data(test_data, deformation_network, value_network,
                                                  def_norm=deformation_norm_dict,
                                                  indenter_filter=cfg["settings"]["id_filter"],
                                                  depth_filter=cfg["settings"]["depth_filter"])

        complete_dict = {
            "train": train_dict,
            "validation": val_dict,
            "test": test_dict,
        }

        complete_reject_dict = {
            "train": train_reject_dict,
            "validation": val_reject_dict,
            "test": test_reject_dict,
        }
    res_folder = Path(cfg["settings"]["result_folder"]).resolve()
    torch.save(complete_dict,
               res_folder / f"accepted_dataset_{cfg['checkpoints']['postfix']}_{cfg['settings']['result_postfix']}.pt")
    torch.save(complete_reject_dict,
               res_folder / f"reject_dataset{cfg['checkpoints']['postfix']}_{cfg['settings']['result_postfix']}.pt")


if __name__ == '__main__':
    main()
