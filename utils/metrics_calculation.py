from pathlib import Path

import hydra

import lightning as pl
import torch

from across.BioTac_Signal_Reconstruction.src.datasets.data_loader import BioTacDataPreperation
from across.BioTac_Signal_Reconstruction.src.model.biotac_autoencoder import BioTacAutoencoder
from across.BioTac_Signal_to_Deformation.src.dataset_loader.biotac_latent_space_loader import BioTacLatentSpaceModule
from across.BioTac_Signal_to_Deformation.src.model.projection_network import ProjectionNetworkBiotac
from across.Mesh_Reconstruction.src.dataset_loader.mesh_data_loader import FEMDataset
from torch_geometric.loader import DataLoader
from across.Mesh_Reconstruction.src.dataset_loader.transform import Normalize

from across.BioTac_to_DIGIT_Deformation.src.model.projection_network import ProjectionNetworkDeformation

import random
import numpy as np

import os
from tqdm import tqdm

os.environ["WANDB_INIT_TIMEOUT"] = "300"


def calcualt_metrices(pred, target, template, target_digit=None, threshhold=0.05):
    # target_digit is the target digit to calculate the deformation region for: it is for the mesh projection network
    batch_size = pred.shape[0]
    if target_digit is None:
        deformation_region = torch.norm(template - target, dim=2, p=2) > threshhold
    else:
        deformation_region = torch.norm(template - target_digit, dim=2, p=2) > threshhold

    rmse = torch.sqrt(torch.mean(
        torch.nn.functional.mse_loss(pred.reshape(batch_size, -1), target.reshape(batch_size, -1),
                                     reduction="none"), dim=[1]))

    euclidean_distance = torch.mean(torch.norm(pred - target, dim=2, p=2), dim=1)

    # TODO: works only with batch_size = 1 because of different deformation_region sizes

    rmse_region = torch.sqrt(torch.mean(
        torch.nn.functional.mse_loss(pred[deformation_region].reshape(batch_size, -1),
                                     target[deformation_region].reshape(batch_size, -1), reduction="none"),
        dim=[1]))

    euclidean_distance_region = torch.mean(
        torch.norm(pred[deformation_region] - target[deformation_region], dim=-1, p=2)).unsqueeze(
        0)  # unsqueeze to make it a batch size of 1 for later concatenation
    return rmse * 1000, euclidean_distance * 1000, rmse_region * 1000, euclidean_distance_region * 1000  # in mu m


def save_obj_file(vertices, faces, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for p in faces:
            f.write('f %d %d %d\n' % (p[0] + 1, p[1] + 1, p[2] + 1))

def save_obj(fem, ind, folder_name, file_name, template):
    fem = fem.cpu().numpy()
    save_obj_file(vertices=fem, faces=template,
                    filename=os.path.join(folder_name, f'{file_name}_{ind}.obj'))



@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg):
    ######################## Signal Reconstuction ##################################################
    signal_model = BioTacAutoencoder.load_from_checkpoint(Path(cfg["experiment"]["signal_checkpoint"]).resolve())
    signal_model = signal_model.eval()
    signal_data_module = BioTacDataPreperation(
        cfg["experiment"]["signal_train_data"],
        test_data_path=cfg["experiment"]["signal_test_data"],
        indenter_filter=cfg["experiment"]["indenter_filter"],
        batch_size=1
    )
    signal_data_module.prepare_data()
    signal_data_module.setup("")

    with torch.no_grad():
        signal_all_loss = []
        for batch in tqdm(signal_data_module.test_dataloader()):
            latent, mu, logvar = signal_model.forward(batch[0].to(signal_model.device))
            decoded = signal_model.decode(latent).to("cpu")
            rmse_loss = torch.sqrt(torch.nn.functional.mse_loss(decoded, batch[1].to("cpu")))
            signal_all_loss.append(rmse_loss)

        signal_all_loss = torch.tensor(signal_all_loss)

        print(f"RMSE Signal Reconstuction: {torch.mean(signal_all_loss):.4f}")
        print(f"RMSE STD Signal Reconstuction: {torch.std(signal_all_loss):.4f}")
        print(f"MAX RMSE Signal Reconstuction: {torch.max(signal_all_loss):.4f}")
        print(f"MIN RMSE Signal Reconstuction: {torch.min(signal_all_loss):.4f}")
    ################################################################################################
    checkpoint_path_signal_to_mesh = cfg["experiment"]["signal_to_deformation_checkpoint"]

    checkpoint_path_biotac = cfg["experiment"]["biotac_deformation_checkpoint"]
    checkpoint_path_digit = cfg["experiment"]["digit_deformation_checkpoint"]

    checkpoint_path_projection_fem = cfg["experiment"]["biotact_to_digit_checkpoint"]

    #read the data

    dataset_biotac_signal = cfg["experiment"]["signal_to_deformation_dataset"]

    #read the data
    data_biotac_signal = torch.load(dataset_biotac_signal)
    fem_biotac_target_data = data_biotac_signal["test"]["real_deformation"]

    # extract the seed from the checkpoint path
    seed_biotac = checkpoint_path_biotac.split("seed_")[1].split("_")[0]
    seed_digit = checkpoint_path_digit.split("seed_")[1].split("_")[0]

    batch_size = 1
    num_workers = 2

    digit_mesh_path = cfg["general"]["digit_mesh_path"]
    biotac_mesh_path = cfg["general"]["biotac_mesh_path"]

    dataset_path = cfg["general"]["dataset_path"]  # TODO What exactly lies here?
    dataset_len = -1
    network_name = "fem_VAE"

    device = cfg["general"]["device"]

    normalize_transform = Normalize()

    if cfg["experiment"]["indenter_filter"] is not None:
    #if int(cfg["experiment"]["indenter_filter"]) == 0:
        dataset_test_biotac = FEMDataset(dataset_path, dtype='test', pre_transform=normalize_transform,
                                         mesh_path=biotac_mesh_path,
                                         sensor_names=["biotac"], dataset_len=dataset_len, min_distance=0.1,
                                         max_distance=2, envs_to_use=[cfg["experiment"]["indenter_filter"]])
        dataset_test_digit = FEMDataset(dataset_path, dtype='test', pre_transform=normalize_transform,
                                        mesh_path=digit_mesh_path,
                                        sensor_names=["digit"], dataset_len=dataset_len, min_distance=0.1,
                                        max_distance=2, envs_to_use=[cfg["experiment"]["indenter_filter"]])
    else:
        dataset_test_biotac = FEMDataset(dataset_path, dtype='test', pre_transform=normalize_transform,
                                         mesh_path=biotac_mesh_path,
                                         sensor_names=["biotac"], dataset_len=dataset_len, min_distance=0.1,
                                         max_distance=2)
        dataset_test_digit = FEMDataset(dataset_path, dtype='test', pre_transform=normalize_transform,
                                        mesh_path=digit_mesh_path,
                                        sensor_names=["digit"], dataset_len=dataset_len, min_distance=0.1,
                                        max_distance=2)

    test_loader_biotac = DataLoader(dataset_test_biotac, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, )
    test_loader_digit = DataLoader(dataset_test_digit, batch_size=batch_size, shuffle=False, num_workers=num_workers, )
    #test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, )

    model_obj = getattr(__import__('across.Mesh_Reconstruction.src.model.fem_autoencoder', fromlist=[network_name]), network_name)

    # load the checkpoint
    model_biotac = model_obj.load_from_checkpoint(checkpoint_path_biotac, mean=0,
                                                  std=1, mesh_path=biotac_mesh_path)  # mean and std are not needed for encoding
    print("Loaded Checkpoint from:", checkpoint_path_biotac)
    # encode the data
    model_biotac.eval()
    # update the mean and std of the model for visualization
    model_biotac.update_mean_std(mean=dataset_test_biotac.mean, std=dataset_test_biotac.std)

    # DIGIT fem Network
    # load the checkpoint
    model_digit = model_obj.load_from_checkpoint(checkpoint_path_digit, mean=0, std=1, mesh_path=digit_mesh_path) # mean and std are not needed for encoding 
    print("Loaded Checkpoint from:", checkpoint_path_digit)
    # encode the data
    model_digit.eval()
    # predict the data and save it 
    # update the mean and std of the model for visualization
    model_digit.update_mean_std(mean=dataset_test_digit.mean, std=dataset_test_digit.std)


    model_projection = ProjectionNetworkDeformation.load_from_checkpoint(
        checkpoint_path_projection_fem, fem_vae_model=None)  # mean and std are not needed for encoding
    model_projection.eval()

    model_biotac = model_biotac.to(device)
    model_digit = model_digit.to(device)
    model_projection = model_projection.to(device)

    ########################## Signal To Deformation #####################
    signal_to_deformation_model = ProjectionNetworkBiotac.load_from_checkpoint(checkpoint_path_signal_to_mesh)
    signal_to_deformation_model = signal_to_deformation_model.to(device)
    signal_to_deformation_model = signal_to_deformation_model.eval()
    value_norms = torch.load(cfg["experiment"]["signal_value_norms"])
    real_values_dict = data_biotac_signal
    latent_space_module = BioTacLatentSpaceModule(
        data_file=cfg["experiment"]["signal_to_deformation_dataset"], batch_size=1,
        use_mu=True)
    latent_space_module.prepare_data()
    latent_space_module.setup()

    latent_test = iter(latent_space_module.test_dataloader())

    all_predicted_real = []
    all_predicted_latent = []
    all_decoded_target = []
    with torch.no_grad():
        for batch in tqdm(latent_test):
            inp, target = batch

            predicted_latent = signal_to_deformation_model.forward(inp.to(signal_to_deformation_model.device)).to("cpu")
            all_predicted_latent.append(predicted_latent)

            real_predicted = model_biotac.decode(predicted_latent.to(signal_to_deformation_model.device)).to("cpu")
            all_predicted_real.append(real_predicted)

            target_decoded = model_biotac.decode(target.to(signal_to_deformation_model.device)).to("cpu")
            all_decoded_target.append(target_decoded)

    data_output = {
        "decoded_predicted": torch.cat(all_predicted_real),
        "latent_predicted": torch.cat(all_predicted_latent),
        "decoded_target": torch.cat(all_decoded_target),
        "trajectories": real_values_dict["test"]["trajects"],
        "indenter_types": torch.squeeze(real_values_dict["test"]["indenter_types"], dim=1),
        "real_value_norm": real_values_dict["test"]["real_value"],
        "real_value": real_values_dict["test"]["real_value"]* value_norms['std'] + value_norms['mean']
    }

    torch.save(data_output, cfg["experiment"]["signal_to_deformation_test_set_decoded"])
    ######################################################################

    template_mesh_faces_biotac = dataset_test_biotac.template_mesh_faces
    template_mesh_faces_digit = dataset_test_digit.template_mesh_faces

    template_mesh_vertices_biotac = torch.tensor(dataset_test_biotac.template_mesh_vertices * 1000, dtype=torch.float32,
                                                 device=device)
    template_mesh_vertices_digit = torch.tensor(dataset_test_digit.template_mesh_vertices * 1000, dtype=torch.float32,
                                                device=device)

    latent_predicted_list = data_output['latent_predicted']
    latent_target_list = data_biotac_signal["test"]['deformation']

    latent_predicted_dl = DataLoader(latent_predicted_list, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, )
    latent_target_dl = DataLoader(latent_target_list, batch_size=batch_size, shuffle=False, num_workers=num_workers, )

    biotac_mesh_target_dl = DataLoader(fem_biotac_target_data, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, )


    rmse_signal_projection_list = None
    euclidean_distance_signal_projection_list = None
    rmse_signal_projection_region_list = None
    euclidean_distance_signal_projection_region_list = None

    folder_name=cfg["experiment"]["folder_name"]
    deformation_region_threshhold = cfg["general"]["deformation_region_threshhold"]
    with torch.no_grad():
        for i, (biotac_pred, biotac_target, biotac_mesh_target) in tqdm(
                enumerate(zip(latent_predicted_dl, latent_target_dl, biotac_mesh_target_dl)),
                total=len(latent_predicted_dl)):
            biotac_pred = biotac_pred.to(device)
            biotac_target = biotac_target.to(device)

            biotac_pred = biotac_pred.reshape(batch_size, -1, 128)
            biotac_target = biotac_target.reshape(batch_size, -1, 128)

            out_biotac_predicted = model_biotac.decode(biotac_pred)
            out_biotac_target = model_biotac.decode(biotac_target)

            out_biotac_predicted = out_biotac_predicted.reshape(batch_size, -1, model_biotac.filters[0])
            out_biotac_target = out_biotac_target.reshape(batch_size, -1, model_biotac.filters[0])

            out_biotac_rescaled = (out_biotac_predicted * model_biotac.std.to(device) + model_biotac.mean.to(
                device)) * 1000
            out_biotac_rescaled_target = (out_biotac_target * model_biotac.std.to(device) + model_biotac.mean.to(
                device)) * 1000

            rmse, euclidean_distance, rmse_region, euclidean_distance_region = calcualt_metrices(out_biotac_rescaled,
                                                                                                 out_biotac_rescaled_target,
                                                                                                 template_mesh_vertices_biotac,
                                                                                                 threshhold=deformation_region_threshhold)

            if rmse_signal_projection_list is None:
                rmse_signal_projection_list = rmse
                euclidean_distance_signal_projection_list = euclidean_distance
                rmse_signal_projection_region_list = rmse_region
                euclidean_distance_signal_projection_region_list = euclidean_distance_region
            else:
                rmse_signal_projection_list = torch.cat((rmse_signal_projection_list, rmse), dim=0)
                euclidean_distance_signal_projection_list = torch.cat(
                    (euclidean_distance_signal_projection_list, euclidean_distance), dim=0)
                rmse_signal_projection_region_list = torch.cat((rmse_signal_projection_region_list, rmse_region), dim=0)
                euclidean_distance_signal_projection_region_list = torch.cat(
                    (euclidean_distance_signal_projection_region_list, euclidean_distance_region), dim=0)

            # convert the data
            digit_ls_pred = model_projection(biotac_pred)
            digit_mesh_converted = model_digit.decode(digit_ls_pred)
            digit_mesh_converted = digit_mesh_converted.reshape(batch_size, -1,
                                                                model_digit.filters[0])
            digit_mesh_converted = (digit_mesh_converted * model_digit.std.to(
                device) + model_digit.mean.to(device)) * 100  # 100 for blender (cm)

            biotac_mesh_target = biotac_mesh_target.to(device)
            biotac_mesh_target = biotac_mesh_target.reshape(batch_size, -1, 3)
            biotac_mesh_target = (biotac_mesh_target * model_biotac.std.to(device) + model_biotac.mean.to(
                device)) * 100  # 100 for blender (cm)

            biotac_reconstructed = model_biotac.decode(biotac_pred)
            biotac_reconstructed = biotac_reconstructed.reshape(batch_size, -1, model_biotac.filters[0])
            biotac_reconstructed = (biotac_reconstructed * model_biotac.std.to(device) + model_biotac.mean.to(
                device)) * 100  # 100 for blender (cm)

            #TODO: only first batch element, however batch_size is anyway 1
            if cfg["general"]["save_mesh"]:
                save_obj(biotac_mesh_target[0], i, folder_name, "biotac_mesh_target", template_mesh_faces_biotac)
                save_obj(biotac_reconstructed[0], i, folder_name, "biotac_mesh_predicted", template_mesh_faces_biotac)
                save_obj(digit_mesh_converted[0], i, folder_name, "digit_mesh_predicted", template_mesh_faces_digit)

    print(
        f"RMSE Signal Projection: {rmse_signal_projection_list.mean().item():.2f} ({rmse_signal_projection_list.std().item():.2f})")
    print(
        f"Euclidean Distance Signal Projection: {euclidean_distance_signal_projection_list.mean().item():.2f} ({euclidean_distance_signal_projection_list.std().item():.2f})")
    print(
        f"RMSE Signal Projection Region: {rmse_signal_projection_region_list.mean().item():.2f} ({rmse_signal_projection_region_list.std().item():.2f})")
    print(
        f"Euclidean Distance Signal Projection Region: {euclidean_distance_signal_projection_region_list.mean().item():.2f} ({euclidean_distance_signal_projection_region_list.std().item():.2f})")
    print()

    test_loader_biotac = DataLoader(dataset_test_biotac, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, )
    test_loader_digit = DataLoader(dataset_test_digit, batch_size=batch_size, shuffle=False, num_workers=num_workers, )


    vertices_count_digit = 6103
    vertices_count_biotac = 4246

    loss_rmse_biotac_fem_list = None
    loss_eucl_biotac_fem_list = None
    loss_rmse_biotac_fem_region_list = None
    loss_eucl_biotac_fem_region_list = None

    loss_rmse_digit_fem_list = None
    loss_eucl_digit_fem_list = None
    loss_rmse_digit_fem_region_list = None
    loss_eucl_digit_fem_region_list = None

    loss_rmse_digit_proj_fem_list = None
    loss_eucl_digit_proj_fem_list = None
    loss_rmse_digit_proj_fem_region_list = None
    loss_eucl_digit_proj_fem_region_list = None

    with torch.no_grad():
        for i, (test_biotac, test_digit) in tqdm(enumerate(zip(test_loader_biotac, test_loader_digit)),
                                                 total=len(test_loader_biotac)):
            test_biotac = test_biotac.x
            test_digit = test_digit.x

            test_biotac = test_biotac.to(device)
            test_digit = test_digit.to(device)

            test_biotac = test_biotac.reshape(-1, vertices_count_biotac, model_biotac.filters[0])
            test_digit = test_digit.reshape(-1, vertices_count_digit, model_digit.filters[0])

            mu_biotac_predicted, logvar_biotac_predicted = model_biotac.encode(test_biotac)
            mu_digit_predicted, logvar_digit_predicted = model_digit.encode(test_digit)

            z_biotac_predicted = model_biotac.reparameterize(mu_biotac_predicted, logvar_biotac_predicted)
            z_digit_predicted = model_digit.reparameterize(mu_digit_predicted, logvar_digit_predicted)

            out_biotac_predicted = model_biotac.decode(z_biotac_predicted)
            out_digit_predicted = model_digit.decode(z_digit_predicted)

            # calcualte metrices for mesh projection network
            z_digit_projected = model_projection(z_biotac_predicted)
            out_digit_projected_predicted = model_digit.decode(z_digit_projected)
            out_digit_projected_predicted = out_digit_projected_predicted.reshape(-1, vertices_count_digit,
                                                                                  model_digit.filters[0])
            out_digit_projected_predicted_rescaled = (out_digit_projected_predicted * model_digit.std.to(
                device) + model_digit.mean.to(device)) * 1000
            #####

            out_biotac_predicted = out_biotac_predicted.reshape(-1, vertices_count_biotac, model_biotac.filters[0])
            out_digit_predicted = out_digit_predicted.reshape(-1, vertices_count_digit, model_digit.filters[0])

            out_biotac_rescaled = (out_biotac_predicted * model_biotac.std.to(device) + model_biotac.mean.to(
                device)) * 1000
            target_biotac_rescaled = (test_biotac * model_biotac.std.to(device) + model_biotac.mean.to(device)) * 1000

            out_digit_rescaled = (out_digit_predicted * model_digit.std.to(device) + model_digit.mean.to(device)) * 1000
            target_digit_rescaled = (test_digit * model_digit.std.to(device) + model_digit.mean.to(device)) * 1000

            rmse_biotac, euclidean_distance_biotac, rmse_biotac_region, euclidean_distance_biotac_region = calcualt_metrices(
                out_biotac_rescaled, target_biotac_rescaled, template_mesh_vertices_biotac, threshhold=deformation_region_threshhold)
            rmse_digit, euclidean_distance_digit, rmse_digit_region, euclidean_distance_digit_region = calcualt_metrices(
                out_digit_rescaled, target_digit_rescaled, template_mesh_vertices_digit, threshhold=deformation_region_threshhold)

            rmse_digit_proj, euclidean_distance_digit_proj, rmse_digit_region_proj, euclidean_distance_digit_region_proj = calcualt_metrices(
                out_digit_projected_predicted_rescaled, out_digit_rescaled, template_mesh_vertices_digit,
                target_digit=target_digit_rescaled, threshhold= deformation_region_threshhold)

            if loss_rmse_biotac_fem_list is None:
                loss_rmse_biotac_fem_list = rmse_biotac
                loss_rmse_digit_fem_list = rmse_digit
                loss_rmse_biotac_fem_region_list = rmse_biotac_region
                loss_rmse_digit_fem_region_list = rmse_digit_region
                loss_eucl_biotac_fem_list = euclidean_distance_biotac
                loss_eucl_digit_fem_list = euclidean_distance_digit
                loss_eucl_biotac_fem_region_list = euclidean_distance_biotac_region
                loss_eucl_digit_fem_region_list = euclidean_distance_digit_region

                loss_rmse_digit_proj_fem_list = rmse_digit_proj
                loss_eucl_digit_proj_fem_list = euclidean_distance_digit_proj
                loss_rmse_digit_proj_fem_region_list = rmse_digit_region_proj
                loss_eucl_digit_proj_fem_region_list = euclidean_distance_digit_region_proj


            else:
                loss_rmse_biotac_fem_list = torch.cat((loss_rmse_biotac_fem_list, rmse_biotac), dim=0)
                loss_rmse_digit_fem_list = torch.cat((loss_rmse_digit_fem_list, rmse_digit), dim=0)
                loss_rmse_biotac_fem_region_list = torch.cat((loss_rmse_biotac_fem_region_list, rmse_biotac_region),
                                                             dim=0)
                loss_rmse_digit_fem_region_list = torch.cat((loss_rmse_digit_fem_region_list, rmse_digit_region), dim=0)
                loss_eucl_biotac_fem_list = torch.cat((loss_eucl_biotac_fem_list, euclidean_distance_biotac), dim=0)
                loss_eucl_digit_fem_list = torch.cat((loss_eucl_digit_fem_list, euclidean_distance_digit), dim=0)
                loss_eucl_biotac_fem_region_list = torch.cat(
                    (loss_eucl_biotac_fem_region_list, euclidean_distance_biotac_region), dim=0)
                loss_eucl_digit_fem_region_list = torch.cat(
                    (loss_eucl_digit_fem_region_list, euclidean_distance_digit_region), dim=0)

                loss_rmse_digit_proj_fem_list = torch.cat((loss_rmse_digit_proj_fem_list, rmse_digit_proj), dim=0)
                loss_eucl_digit_proj_fem_list = torch.cat(
                    (loss_eucl_digit_proj_fem_list, euclidean_distance_digit_proj), dim=0)
                loss_rmse_digit_proj_fem_region_list = torch.cat(
                    (loss_rmse_digit_proj_fem_region_list, rmse_digit_region_proj), dim=0)
                loss_eucl_digit_proj_fem_region_list = torch.cat(
                    (loss_eucl_digit_proj_fem_region_list, euclidean_distance_digit_region_proj), dim=0)

    print(
        f"RMSE Biotac FEM: {torch.mean(loss_rmse_biotac_fem_list).item():.2f} ({torch.std(loss_rmse_biotac_fem_list).item():.2f})")
    print(
        f"Euclidean Distance Biotac FEM: {torch.mean(loss_eucl_biotac_fem_list).item():.2f} ({torch.std(loss_eucl_biotac_fem_list).item():.2f})")
    print(
        f"RMSE Biotac FEM Region: {torch.mean(loss_rmse_biotac_fem_region_list).item():.2f} ({torch.std(loss_rmse_biotac_fem_region_list).item():.2f})")
    print(
        f"Euclidean Distance Biotac FEM Region: {torch.mean(loss_eucl_biotac_fem_region_list).item():.2f} ({torch.std(loss_eucl_biotac_fem_region_list).item():.2f})")
    print()

    print(
        f"RMSE Digit FEM: {torch.mean(loss_rmse_digit_fem_list).item():.2f} ({torch.std(loss_rmse_digit_fem_list).item():.2f})")
    print(
        f"Euclidean Distance Digit FEM: {torch.mean(loss_eucl_digit_fem_list).item():.2f} ({torch.std(loss_eucl_digit_fem_list).item():.2f})")
    print(
        f"RMSE Digit FEM Region: {torch.mean(loss_rmse_digit_fem_region_list).item():.2f} ({torch.std(loss_rmse_digit_fem_region_list).item():.2f})")
    print(
        f"Euclidean Distance Digit FEM Region: {torch.mean(loss_eucl_digit_fem_region_list).item():.2f} ({torch.std(loss_eucl_digit_fem_region_list).item():.2f})")
    print()

    print(
        f"RMSE Digit Proj FEM: {torch.mean(loss_rmse_digit_proj_fem_list).item():.2f} ({torch.std(loss_rmse_digit_proj_fem_list).item():.2f})")
    print(
        f"Euclidean Distance Digit Proj FEM: {torch.mean(loss_eucl_digit_proj_fem_list).item():.2f} ({torch.std(loss_eucl_digit_proj_fem_list).item():.2f})")
    print(
        f"RMSE Digit Proj FEM Region: {torch.mean(loss_rmse_digit_proj_fem_region_list).item():.2f} ({torch.std(loss_rmse_digit_proj_fem_region_list).item():.2f})")
    print(
        f"Euclidean Distance Digit Proj FEM Region: {torch.mean(loss_eucl_digit_proj_fem_region_list).item():.2f} ({torch.std(loss_eucl_digit_proj_fem_region_list).item():.2f})")


if __name__ == '__main__':
    main()
