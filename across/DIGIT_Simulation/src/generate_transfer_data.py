# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
from tqdm import tqdm
import hydra
import numpy as np

from across.DIGIT_Simulation.renderer.renderer import Renderer

def euler_to_R_matrix(alpha, beta, gamma):
    # degree to radian
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]
                    ])

    R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                    ])

    R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                    ])
    R_matrix = np.dot(R_z, np.dot(R_y, R_x))
    return R_matrix

def transform_points(points, vertices_surface_indices):
    R_matrix = np.dot(euler_to_R_matrix(-90, 0, 0),euler_to_R_matrix(0, 0, 90))
    T_vector = np.array([0.011873, 0, 0.009685-0.0015-0.003441]) # for shifted sensor -0.0015

    T_matrix = np.zeros((4,4))
    T_matrix[:3,:3] = R_matrix
    T_matrix[:3,3] = T_vector
    T_matrix[3,3] = 1

    points_transformed = []
    for point in points[vertices_surface_indices]: 
        #point[1]=point[1]-(5e-02) # the isaac gym has a 5 cm offset in the y axes 
        point = np.append(point, 1)
        # transform point
        point = np.dot(T_matrix, point)
        points_transformed.append(point[:3])
    points_transformed=np.array(points_transformed)

    return points_transformed

# read obj and get vertices
def read_vertices(filename):
    vertices = []
    with open(filename) as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
    return np.array(vertices)


def save_obj(vertices, faces, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for face in faces:
            f.write('f %d %d %d\n' % (face[0], face[1], face[2]))#

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg):
    save_dir = cfg["experiment"]["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    renderer = Renderer(width=cfg["renderer"]["w"], height=cfg["renderer"]["h"],
                        config_path=cfg["renderer"]["config_path"], cfg_renderer=cfg["renderer"], visualizer_pyrenderer=cfg["experiment"]["gui"])

    folder_path = cfg["experiment"]["obj_path"]

    filenames = [f for f in os.listdir(folder_path) if f.startswith("digit_mesh_predicted_")]
    num_steps = len(filenames)

    # sort the filenames based on the indexes in the filename
    filenames = sorted(filenames, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # load indices of the surface vertices
    surface_vertices_path=cfg["renderer"]["surface_vertices_path"]
    vertices_surface_indices = np.load(surface_vertices_path)
    print("vertices_surface_indices loaded: ", vertices_surface_indices.shape)

    # only render once the static color at the beginning for all the steps and env.
    render_static = None

    color_tactile = []
    depth_tactile = []
    for step in tqdm(range(num_steps)):

        nodal_coords= read_vertices(os.path.join(folder_path, filenames[step]))
        nodal_coords=nodal_coords/100 # saved in cm for blender view, convert back to m
        points_transformed = transform_points(nodal_coords, vertices_surface_indices)

        # update the mesh in the renderer
        renderer.update_gelmap_mesh(points_transformed) 
    
        # render the images and save them
        color, depth = renderer.render()
        
        # Remove the depth from curved gel
        for j in range(len(depth)):
            depth[j] = renderer.depth0[j] - depth[j]

        # for noisy images ignore them and set the depth to zero
        render_static = np.max(depth) < 1e-6 # values are normalled to be between 0 and xe-8
        if render_static:
            color, depth = renderer._background_sim, [np.zeros_like(d0) for d0 in renderer.depth0]
            color = [renderer._add_noise(color_img) for color_img in color]

        #color_tactile.append(color[0].copy())
        #depth_tactile.append(depth[0].copy())

        # normalize the image between 0 and 1
        img = depth[0].copy()
        img = img - np.min(img)
        if np.max(img) > 0: # the first image is zero
            img = img / np.max(img)
        img = img * 255.0
        img = img.astype(np.uint8)

        # save color, depth image as npy and jpg
        #if not render_static:
        cv2.imwrite(os.path.join(save_dir, "color_"+str(step)+'.jpg'), color[0].copy())
        #np.save(os.path.join(save_dir, "depth_numpy_"+str(step)+'.npy'), depth[0].copy())
        #cv2.imwrite(os.path.join(save_dir, "depth_"+str(step)+'.jpg'), img)

if __name__ == "__main__":
    main()