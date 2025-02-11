from pathlib import Path

import h5py
import numpy as np


def split_data(data_dict, trajects, filtered_dict):
    for traject in trajects:
        for index in traject:
            filtered_dict["contact_loc"].append(data_dict["contact_loc"][index])
            filtered_dict["electrode_vals"].append(data_dict["electrode_vals"][index])
            filtered_dict["force_vec"].append(data_dict["force_vec"][index])
            filtered_dict["indenter_type"].append(data_dict["indenter_type"][index])
            filtered_dict["nodal_res"].append(data_dict["nodal_res"][index])
            filtered_dict["force_res"].append(data_dict["force_res"][index])


if __name__ == '__main__':
    data_paths = [
        Path("../across/Data/datasets/BioTac_Pure_Exp_Dataset/shuffled_train_data.hdf5").resolve()

    ]
    simulated_data_paths = [
        Path("../across/Data/datasets/BioTac_Pure_Exp_Dataset/simulated_shuffled_train_data.hdf5").resolve()
    ]

    biotac_data = {
        "contact_loc": [],
        "electrode_vals": [],
        "force_vec": [],
        "indenter_type": [],
        "nodal_res": [],
        "force_res": []
    }

    train_biotac_data = {
        "contact_loc": [],
        "electrode_vals": [],
        "force_vec": [],
        "indenter_type": [],
        "nodal_res": [],
        "force_res": []
    }

    test_biotac_data = {
        "contact_loc": [],
        "electrode_vals": [],
        "force_vec": [],
        "indenter_type": [],
        "nodal_res": [],
        "force_res": []
    }

    test_percentage = 0.15

    for path, sim_path in zip(data_paths, simulated_data_paths):
        trajects = []
        active_trajec = []
        with h5py.File(path, 'r') as f:
            biotac_data["contact_loc"] = f["contact_loc"]["data"][:]
            biotac_data["electrode_vals"] = f["electrode_vals"]["data"][:]
            biotac_data["force_vec"] = f["force_vec"]["data"][:]
            biotac_data["indenter_type"] = f["indenter_type"]["data"][:]

        with h5py.File(sim_path, 'r') as f:
            biotac_data["nodal_res"] = f["nodal_res"][:]
            biotac_data["force_res"] = f["force_res"][:]

        i = biotac_data["indenter_type"]
        c = biotac_data["contact_loc"]

        prev = None
        prev_i = None
        for index in range(len(c)):
            if not active_trajec:
                active_trajec.append(index)
                prev = c[index]
                prev_i = i[index]
            else:
                dist = np.linalg.norm(c[index] - prev)
                # print(dist)
                if (i[index] != prev_i) or (dist > 5e-4):
                    trajects.append(active_trajec)
                    # print(active_trajec)
                    # print(dist)
                    active_trajec = [index]
                    prev = c[index]
                    prev_i = i[index]
                else:
                    active_trajec.append(index)
                    prev = c[index]
                    prev_i = i[index]
        trajects.append(active_trajec)
        shuffled_dataset = np.random.choice(np.arange(len(trajects)), len(trajects), replace=False)
        print(shuffled_dataset)

        test_index = int(test_percentage * len(trajects))
        test_dataset = shuffled_dataset[:test_index]
        train_dataset = shuffled_dataset[test_index:]

        split_data(biotac_data, [trajects[k] for k in test_dataset], test_biotac_data)
        split_data(biotac_data, [trajects[k] for k in train_dataset], train_biotac_data)
        print(len(trajects))
        print(trajects)

    print(len(test_biotac_data["contact_loc"]))
    print(len(train_biotac_data["contact_loc"]))

    path_to_res = Path("../across/Data/datasets/BioTac_Pure_Exp_Dataset").resolve()
    if not path_to_res.exists():
        path_to_res.mkdir(parents=True, exist_ok=True)

    with h5py.File(path_to_res / "shuffled_val_data.hdf5", 'w') as f:
        f.create_group("contact_loc")
        f.create_group("electrode_vals")
        f.create_group("force_vec")
        f.create_group("indenter_type")

        f["contact_loc"].create_dataset("data", data=test_biotac_data["contact_loc"])
        f["electrode_vals"].create_dataset("data", data=test_biotac_data["electrode_vals"])
        f["force_vec"].create_dataset("data", data=test_biotac_data["force_vec"])
        f["indenter_type"].create_dataset("data", data=test_biotac_data["indenter_type"])

    with h5py.File(path_to_res / "simulated_shuffled_val_data.hdf5", 'w') as f:
        f.create_dataset("nodal_res", data=test_biotac_data["nodal_res"])
        f.create_dataset("force_res", data=test_biotac_data["force_res"])

    with h5py.File(path_to_res / "shuffled_train_data_2.hdf5", 'w') as f:
        f.create_group("contact_loc")
        f.create_group("electrode_vals")
        f.create_group("force_vec")
        f.create_group("indenter_type")

        f["contact_loc"].create_dataset("data", data=train_biotac_data["contact_loc"])
        f["electrode_vals"].create_dataset("data", data=train_biotac_data["electrode_vals"])
        f["force_vec"].create_dataset("data", data=train_biotac_data["force_vec"])
        f["indenter_type"].create_dataset("data", data=train_biotac_data["indenter_type"])

    with h5py.File(path_to_res / "simulated_shuffled_train_data_2.hdf5", 'w') as f:
        f.create_dataset("nodal_res", data=train_biotac_data["nodal_res"])
        f.create_dataset("force_res", data=train_biotac_data["force_res"])

