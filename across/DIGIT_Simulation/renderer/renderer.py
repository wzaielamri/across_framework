# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Set backend platform for OpenGL render (pyrender.OffscreenRenderer)
- Pyglet, the same engine that runs the pyrender viewer. This requires an active
  display manager, so you can't run it on a headless server. This is the default option.
- OSMesa, a software renderer. require extra install OSMesa.
  (https://pyrender.readthedocs.io/en/latest/install/index.html#installing-osmesa)
- EGL, which allows for GPU-accelerated rendering without a display manager.
  Requires NVIDIA's drivers.

The handle for EGL is egl (preferred, require NVIDIA driver),
The handle for OSMesa is osmesa.
Default is pyglet, which requires active window
"""

import os
import threading


os.environ["PYOPENGL_PLATFORM"] = "osmesa"
#os.environ["PYOPENGL_PLATFORM"] = "egl"
#os.environ["EGL_DEVICE_ID"] = "0"

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyrender
import trimesh
from omegaconf import OmegaConf
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from across.DIGIT_Simulation.renderer.digit_render import digitRender


def euler2matrix(angles=(0, 0, 0), translation=(0, 0, 0)):
    r = R.from_euler('xyz', angles)
    # Get the rotation matrix
    rotation_matrix = r.as_matrix()
    pose = np.eye(4)
    pose[:3, 3] = translation
    pose[:3, :3] = rotation_matrix
    return pose


class RenderThread(threading.Thread):
    """
    This class implements a HelperThread to allow the rendering of the gui cameras in another thread to avoid
    conflicts with the rendering of main context.
    """

    def __init__(self, scene):
        super().__init__()
        self.scene = scene

    def reload_meshes(self):
        for node in self.scene.nodes:
            if node.mesh is not None:
                for primitive in node.mesh.primitives:
                    if primitive._in_context():
                        primitive._unbind()
                        primitive._remove_from_context()

                        # primitive._add_to_context()
    def run(self) -> None:
        """
        This function renders the gui cameras
        """
        self.reload_meshes()
        pyrender.Viewer(self.scene)


class Renderer:
    def __init__(self, width, height, config_path, cfg_renderer, visualizer_pyrenderer=False, background=None):
        """
        :param width: scalar
        :param height: scalar
        :param background: image
        :param config_path: digit config path
        """
        self._width = width
        self._height = height
        self.digit_render = digitRender(cfg_renderer)

        self.visualizer_pyrenderer = visualizer_pyrenderer
        if background is not None:
            self.set_background(background)
        else:
            self._background_real = None
        self.conf = OmegaConf.load(config_path)
        
        
        self.force_enabled = (
                self.conf.sensor.force is not None and self.conf.sensor.force.enable
        )

        if self.force_enabled:
            if len(self.conf.sensor.force.range_force) == 2:
                self.get_offset = interp1d(self.conf.sensor.force.range_force,
                                           [0, self.conf.sensor.force.max_deformation],
                                           bounds_error=False,
                                           fill_value=(0, self.conf.sensor.force.max_deformation))
            else:
                self.get_offset = interp1d(self.conf.sensor.force.range_force,
                                           self.conf.sensor.force.max_deformation,
                                           bounds_error=False,
                                           fill_value=(0, self.conf.sensor.force.max_deformation[-1]))
        self.step = 0 
        self._init_pyrender()

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def background(self):
        return self._background_real

    def _init_pyrender(self):
        """
        Initialize pyrender
        """
        # Create scene for pybullet sync
        self.scene = pyrender.Scene()

        # Create scene for rendering given depth image
        self.scene_depth = pyrender.Scene()

        """
        objects format:
            {obj_name: pyrender node}
        """
        self.object_nodes = {}
        self.object_depth_nodes = {}
        self.current_object_nodes = {}
        self.object_trimeshes = {}

        self.current_light_nodes = []
        self.cam_light_ids = []

        self._init_gel()
        self._init_camera()
        self._init_light()

        self.r = pyrender.OffscreenRenderer(self.width, self.height)
        colors, depths = self.render(noise=False, calibration=False)


        # self.show_scene()
        self.depth0 = depths
        #np.save("depth0.npy", self.depth0[0])
        self._background_sim = colors

    def show_scene(self):
        scene_visual = pyrender.Scene()

        # add object nodes
        for objname, objnode in self.current_object_nodes.items():
            objTrimesh = self.object_trimeshes[objname]
            pose = objnode.matrix
            mesh = pyrender.Mesh.from_trimesh(objTrimesh)
            obj_node_new = pyrender.Node(mesh=mesh, matrix=pose)
            scene_visual.add_node(obj_node_new)

        # add gel node
        mesh_gel = pyrender.Mesh.from_trimesh(self.gel_trimesh, smooth=False)
        gel_pose = self.gel_node.matrix
        gel_node_new = pyrender.Node(mesh=mesh_gel, matrix=gel_pose)
        scene_visual.add_node(gel_node_new)

        # add light
        for i, light_node in enumerate(self.light_nodes):
            color = self.light_colors[i]
            intensity = self.light_intensities[i]
            light_new = pyrender.PointLight(color=color, intensity=intensity)
            light_pose = light_node.matrix
            light_node_new = pyrender.Node(light=light_new, matrix=light_pose)
            scene_visual.add_node(light_node_new)

        # add camera
        for i, camera_node in enumerate(self.camera_nodes):
            cami = self.conf_cam[i]
            camera = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(cami.yfov), znear=cami.znear,
            )
            pose = camera_node.matrix
            camera_node = pyrender.Node(camera=camera, matrix=pose)
            scene_visual.add_node(camera_node)

        pyrender.Viewer(scene_visual)

    def show_scene_depth(self):
        print("call show_scene_depth")
        self._print_all_pos_depth()
        scene_visual = pyrender.Scene()
        for objname, objnode in self.object_depth_nodes.items():
            objTrimesh = self.object_trimeshes[objname]
            pose = objnode.matrix
            mesh = pyrender.Mesh.from_trimesh(objTrimesh)
            obj_node_new = pyrender.Node(mesh=mesh, matrix=pose)
            scene_visual.add_node(obj_node_new)
        # add light
        for i, light_node in enumerate(self.light_nodes):
            color = self.light_colors[i]
            intensity = self.light_intensities[i]
            light_new = pyrender.PointLight(color=color, intensity=intensity)
            light_pose = light_node.matrix
            light_node_new = pyrender.Node(light=light_new, matrix=light_pose)
            scene_visual.add_node(light_node_new)

        # add camera
        for i, camera_node in enumerate(self.camera_nodes):
            cami = self.conf_cam[i]
            camera = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(cami.yfov), znear=cami.znear,
            )
            pose = camera_node.matrix
            camera_node = pyrender.Node(camera=camera, matrix=pose)
            scene_visual.add_node(camera_node)

        pyrender.Viewer(scene_visual)

    def _init_gel(self):
        """
        Add gel surface in the scene
        """
        # Create gel surface (flat/curve surface based on config file)
        self.gel_trimesh = self._generate_gel_trimesh()
        mesh_gel = pyrender.Mesh.from_trimesh(self.gel_trimesh, smooth=False)
        print("Gel: {}".format(self.gel_trimesh))
        self.gel_pose0 = np.eye(4)
        self.gel_node = pyrender.Node(mesh=mesh_gel, matrix=self.gel_pose0)
        self.scene.add_node(self.gel_node)

    def _generate_gel_trimesh(self):

        # Load config
        g = self.conf.sensor.gel

        if hasattr(g, "mesh") and g.mesh is not None:
            #mesh_dir = os.path.dirname(os.path.realpath(__file__))
            #get execution directory
            mesh_dir = os.getcwd()
            mesh_path = os.path.join(mesh_dir, g.mesh)
            gel_trimesh = trimesh.load(mesh_path)

        elif not g.curvature:
            # Flat gel surface
            origin = g.origin

            X0, Y0, Z0 = origin[0], origin[1], origin[2]
            W, H = g.width, g.height
            gel_trimesh = trimesh.Trimesh(
                vertices=[
                    [X0, Y0 + W / 2, Z0 + H / 2],
                    [X0, Y0 + W / 2, Z0 - H / 2],
                    [X0, Y0 - W / 2, Z0 - H / 2],
                    [X0, Y0 - W / 2, Z0 + H / 2],
                ],
                faces=[[0, 1, 2], [2, 3, 0]],
            )
        else:
            origin = g.origin
            X0, Y0, Z0 = origin[0], origin[1], origin[2]
            W, H = g.width, g.height
            # Curved gel surface
            N = g.countW
            M = int(N * H / W)
            R = g.R
            zrange = g.curvatureMax

            y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
            z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
            yy, zz = np.meshgrid(y, z)

            h = R - np.maximum(0, R ** 2 - (yy - Y0) ** 2 - (zz - Z0) ** 2) ** 0.5
            xx = X0 - zrange * h / h.max()

            gel_trimesh = self._generate_trimesh_from_depth(xx)
        # from datetime import datetime
        # stl_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gelsight_data", "mesh_tmp_{}.stl".format(datetime.now().strftime("%Y%m%d%H%M%S")))
        # gel_trimesh.export(stl_filename)
        print("Gel mesh bounds={}".format(gel_trimesh.bounds))
        return gel_trimesh

    def _generate_trimesh_from_depth(self, depth):
        # Load config
        g = self.conf.sensor.gel
        origin = g.origin

        _, Y0, Z0 = origin[0], origin[1], origin[2]
        W, H = g.width, g.height

        N = depth.shape[1]
        M = depth.shape[0]

        # Create grid mesh
        y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
        z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
        yy, zz = np.meshgrid(y, z)

        # Vertex format: [x, y, z]
        vertices = np.zeros([N * M, 3])

        # Add x, y, z position to vertex
        vertices[:, 0] = depth.reshape([-1])
        vertices[:, 1] = yy.reshape([-1])
        vertices[:, 2] = zz.reshape([-1])

        # Create faces

        faces = np.zeros([(N - 1) * (M - 1) * 6], dtype=np.uint)

        # calculate id for each vertex: (i, j) => i * m + j
        xid = np.arange(N)
        yid = np.arange(M)
        yyid, xxid = np.meshgrid(xid, yid)
        ids = yyid[:-1, :-1].reshape([-1]) + xxid[:-1, :-1].reshape([-1]) * N

        # create upper triangle
        faces[::6] = ids  # (i, j)
        faces[1::6] = ids + N  # (i+1, j)
        faces[2::6] = ids + 1  # (i, j+1)

        # create lower triangle
        faces[3::6] = ids + 1  # (i, j+1)
        faces[4::6] = ids + N  # (i+1, j)
        faces[5::6] = ids + N + 1  # (i+1, j+1)

        faces = faces.reshape([-1, 3])
        gel_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        return gel_trimesh

    def _init_camera(self):
        """
        Set up camera
        """

        self.camera_nodes = []
        self.camera_zero_poses = []
        self.camera_depth_nodes = []

        self.conf_cam = self.conf.sensor.camera
        self.nb_cam = len(self.conf_cam)

        for i in range(self.nb_cam):
            cami = self.conf_cam[i]
            camera = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(cami.yfov), znear=cami.znear,
            )

            camera_zero_pose = euler2matrix(
                angles=np.deg2rad(cami.orientation), translation=cami.position,
            )
            self.camera_zero_poses.append(camera_zero_pose)

            # Add camera node into scene
            camera_node = pyrender.Node(camera=camera, matrix=camera_zero_pose)
            self.scene.add_node(camera_node)
            self.camera_nodes.append(camera_node)

            # Add extra camera node into scene_depth
            camera_node_depth = pyrender.Node(camera=camera, matrix=camera_zero_pose)
            self.scene_depth.add_node(camera_node_depth)
            self.camera_depth_nodes.append(camera_node_depth)

            # Add corresponding light for rendering the camera
            self.cam_light_ids.append(list(cami.lightIDList))

    def _init_light(self):
        """
        Set up light
        """

        # Load light from config file
        light = self.conf.sensor.lights

        origin = np.array(light.origin)

        xyz = []
        if light.polar:
            # Apply polar coordinates
            thetas = light.xrtheta.thetas
            rs = light.xrtheta.rs
            xs = light.xrtheta.xs
            for i in range(len(thetas)):
                theta = np.pi / 180 * thetas[i]
                xyz.append([xs[i], rs[i] * np.cos(theta), rs[i] * np.sin(theta)])
        else:
            # Apply cartesian coordinates
            xyz = np.array(light.xyz.coords)

        self.light_colors = np.array(light.colors)
        self.light_intensities = light.intensities

        # Save light nodes
        self.light_nodes = []
        self.light_poses0 = []
        self.light_depth_nodes = []

        for i in range(len(self.light_colors)):
            color = self.light_colors[i]
            intensity = self.light_intensities[i]
            light_pose_0 = euler2matrix(angles=[0, 0, 0], translation=xyz[i] + origin)

            light = pyrender.PointLight(color=color, intensity=intensity)
            light_node = pyrender.Node(light=light, matrix=light_pose_0)

            self.scene.add_node(light_node)
            self.light_nodes.append(light_node)
            self.light_poses0.append(light_pose_0)
            self.current_light_nodes.append(light_node)

            # Add extra light node into scene_depth
            light_node_depth = pyrender.Node(light=light, matrix=light_pose_0)
            self.scene_depth.add_node(light_node_depth)
            self.light_depth_nodes.append(light_node_depth)


    def update_gelmap_mesh(self, points_transformed):
        self.gel_trimesh.vertices = points_transformed
        mesh_gel = pyrender.Mesh.from_trimesh(self.gel_trimesh, smooth=False)
        self.gel_node.mesh = mesh_gel

    def update_camera_pose(self, position, orientation):
        """
        Update sensor pose (including camera, lighting, and gel surface)
        ### important ###
        call self.update_camera_pose before self.render
        """

        pose = euler2matrix(angles=orientation, translation=position)

        # Update camera
        for i in range(self.nb_cam):
            camera_pose = pose.dot(self.camera_zero_poses[i])
            self.camera_nodes[i].matrix = camera_pose
            # update depth camera
            self.camera_depth_nodes[i].matrix = camera_pose

        # Update gel
        gel_pose = pose.dot(self.gel_pose0)
        self.gel_node.matrix = gel_pose

        # Update light
        for i in range(len(self.light_nodes)):
            light_pose = pose.dot(self.light_poses0[i])
            self.light_nodes[i].matrix = light_pose
            # update depth light
            self.light_depth_nodes[i].matrix = light_pose

    def update_light(self, lightIDList):
        """
        Update the light node based on lightIDList, remove the previous light
        """
        # Remove previous light nodes
        for node in self.current_light_nodes:
            self.scene.remove_node(node)

        # Add light nodes
        self.current_light_nodes = []
        for i in lightIDList:
            light_node = self.light_nodes[i]
            self.scene.add_node(light_node)
            self.current_light_nodes.append(light_node)

    def _add_noise(self, color):
        """
        Add Gaussian noise to the RGB image
        :param color:
        :return:
        """
        # Add noise to the RGB image
        mean = self.conf.sensor.noise.color.mean
        std = self.conf.sensor.noise.color.std

        if mean != 0 or std != 0:
            noise = np.random.normal(mean, std, color.shape)  # Gaussian noise
            color = np.clip(color + noise, 0, 255).astype(
                np.uint8
            )  # Add noise and clip

        return color

    def _calibrate(self, color, camera_index):
        """
        Calibrate simulation wrt real sensor by adding background
        :param color:
        :return:
        """

        if self._background_real is not None:
            # Simulated difference image, with scaling factor 0.5
            diff = (color.astype(np.float) - self._background_sim[camera_index]) * 0.5
            # Add low-pass filter to match real readings
            diff = cv2.GaussianBlur(diff, (7, 7), 0)

            # Combine the simulated difference image with real background image
            color = np.clip((diff[:, :, :3] + self._background_real), 0, 255).astype(
                np.uint8
            )

        return color

    def set_background(self, background):
        self._background_real = cv2.resize(background, (self._width, self._height))
        self._background_real = self._background_real[:, :, ::-1]
        return 0

    def _post_process(self, color, depth, camera_index, noise=True, calibration=True):
        if calibration:
            color = self._calibrate(color, camera_index)
        if noise:
            color = self._add_noise(color)
        return color, depth
    
    def reload_meshes(self):
        for node in self.scene.nodes:
            if node.mesh is not None:
                for primitive in node.mesh.primitives:
                    if primitive._in_context():
                        primitive._unbind()
                        primitive._remove_from_context()
                        primitive._add_to_context()

    @property
    def static(self):
        if self._static is None:
            colors, _ = self.renderer.render(noise=False)
            depths = [np.zeros_like(d0) for d0 in self.depth0]
            self._static = (colors, depths)

        return self._static

    def _render_static(self):
        colors, depths = self.static
        colors = [self._add_noise(color) for color in colors]
        return colors, depths


    def render(self,  noise=False, calibration=True, camera_pos_old=None, camera_ori_old=None):
        """
        :param object_poses:
        :param normal_forces:
        :param noise:
        :return:
        """
        # print("Begin Rendering")
        if camera_pos_old is not None and camera_ori_old is not None:
            self.update_camera_pose(camera_pos_old, camera_ori_old)

        colors, depths = [], []

        for i in range(self.nb_cam):
            # Set the main camera node for rendering
            self.scene.main_camera_node = self.camera_nodes[i]

            # Set up corresponding lights (max: 8)
            self.update_light(self.cam_light_ids[i])

            if self.visualizer_pyrenderer:
                self.reload_meshes()
                visualizer = RenderThread(self.scene)
                visualizer.start()
                visualizer.join()
                self.reload_meshes()

            self.r = pyrender.OffscreenRenderer(self.width, self.height)
            color, depth = self.r.render(self.scene)


            color, depth = self._post_process(color, depth, i, noise, calibration)

            # render color from sensor
            #print("save depth")
            #np.save("test/depth_"+str(self.step)+'.npy', depth)
            color_gel = self.digit_render.render(depth.copy())
            #cv2.imwrite("test/colo_"+str(self.step)+'.jpg', color_gel.copy())
            color = np.clip(color_gel, 0, 255, out=color_gel).astype(np.uint8)

            self.step += 1
            colors.append(color)
            depths.append(depth)
        return colors, depths

    def print_all_pos(self):
        camera_pose = self.camera_nodes[0].matrix
        camera_pos = camera_pose[:3, 3].T
        camera_ori = R.from_matrix(camera_pose[:3, :3]).as_quat()
        # print("camera pos and ori in pyrender=", (camera_pos, camera_ori))

        gel_pose = self.gel_node.matrix
        gel_pos = gel_pose[:3, 3].T
        gel_ori = R.from_matrix(gel_pose[:3, :3]).as_quat()
        # print("Gel pos and ori in pyrender=", (gel_pos, gel_ori))

        obj_pose = self.object_nodes["2_-1"].matrix
        obj_pos = obj_pose[:3, 3].T
        obj_ori = R.from_matrix(obj_pose[:3, :3]).as_quat()
        return camera_pose, gel_pose, obj_pose

    def _print_all_pos_depth(self):
        camera_pose = self.camera_depth_nodes[0].matrix
        camera_pos = camera_pose[:3, 3].T
        camera_ori = R.from_matrix(camera_pose[:3, :3]).as_quat()
        print("depth camera pos and ori in pyrender=", (camera_pos, camera_ori))

        obj_pose = self.object_depth_nodes["2_-1"].matrix
        obj_pos = obj_pose[:3, 3].T
        obj_ori = R.from_matrix(obj_pose[:3, :3]).as_quat()
        print("depth obj pos and ori in pyrender=", (obj_pos, obj_ori))
