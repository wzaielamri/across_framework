from pathlib import Path

import numpy as np


def conv_tet_to_vertices_and_faces(tet_file_path):
    with open(tet_file_path, "r") as f:
        content = f.readlines()

    vertices = []
    faces = []
    for line in content:
        line_content = line.split(" ")
        if line_content[0] == "v":
            vertices.append([float(val) for val in line_content[1:4]])
        elif line_content[0] == "t":
            indices = [int(val) for val in line_content[1:]]

            faces.append([indices[0], indices[1], indices[2]])
            faces.append([indices[0], indices[1], indices[3]])
            faces.append([indices[0], indices[2], indices[3]])
            faces.append([indices[1], indices[2], indices[3]])

    return np.array(vertices), np.array(faces)


if __name__ == '__main__':
    v, f = conv_tet_to_vertices_and_faces(Path("/data/template_mesh.tet").resolve())
    print(v.shape)
    print(f.shape)
