#Adapted from RAINBOW: https://github.com/diku-dk/RAINBOW/blob/main/python/rainbow/util/viewer.py
import numpy as np
import pythreejs as p3js
from IPython.display import display

from articulated_dynamics.kinematics import Kinematics

class _ShapeHelper:
    @classmethod
    def make_arrow_node(cls, color, scale=1.0):
        base = p3js.Mesh(p3js.SphereGeometry(0.25/3 * scale, 12, 12), p3js.MeshStandardMaterial(color=color))

        shaft = p3js.Mesh(p3js.CylinderGeometry(0.1/3*scale, 0.1/3*scale, 2.0/3*scale, 12), p3js.MeshStandardMaterial(color=color))
        shaft.position = [0, 1.0 / 3 * scale, 0]

        cone = p3js.Mesh(p3js.CylinderGeometry(0.0, 0.25/3*scale, 1.0/3*scale, 12), p3js.MeshStandardMaterial(color=color))
        cone.position = [0, 2.0 / 3 * scale, 0.0]
        arrow = p3js.Group()
        arrow.add(base)
        arrow.add(shaft)
        arrow.add(cone)
        return arrow
    
    @classmethod
    def make_frame_node(cls, scale=1.0):
        x_arrow = cls.make_arrow_node(color='red', scale=scale)
        y_arrow = cls.make_arrow_node(color='green', scale=scale)
        z_arrow = cls.make_arrow_node(color='blue', scale=scale)
        x_arrow.quaternion = [ 0, 0, -0.7071068, 0.7071068 ]
        z_arrow.quaternion = [ 0.7071068, 0, 0, 0.7071068 ]

        frame = p3js.Group()
        frame.add(x_arrow)
        frame.add(y_arrow)
        frame.add(z_arrow)
        return frame
    
    @classmethod
    def make_sphere_node(cls, name, radius, color):
        sphere = p3js.Mesh(p3js.SphereGeometry(radius, 12, 12), p3js.MeshStandardMaterial(color=color), name=name)
        return sphere
    
    @classmethod
    def make_cylinder_node(cls, name, radius, height, color):
        cylinder = p3js.Mesh(p3js.CylinderGeometry(radius, radius, height, 12), p3js.MeshStandardMaterial(color=color), name=name)
        return cylinder
    
    @classmethod
    def make_box_node(cls, name, width, height, depth, color):
        box = p3js.Mesh(p3js.BoxGeometry(width, height, depth, 1, 1, 1), p3js.MeshStandardMaterial(color=color), name=name)
        return box


class P3JSViewer:
    def __init__(self, width=600, height=480):
        self.shapes = {}
        self.width = int(width)
        self.height = int(height)
        self.aspect = 1.0 * self.width / self.height
        self.fov = 30.0
        self.flash_light = p3js.DirectionalLight(
            color="white", position=[0, 0, 1], intensity=0.6
        )
        self.sun_light = p3js.AmbientLight(intensity=0.5)
        self.camera = p3js.PerspectiveCamera(
            position=[3, 3, 3],
            lookAt=[0, 0, 0],
            fov=self.fov,
            aspect=self.aspect,
            children=[self.flash_light],
        )
        self.orbit = p3js.OrbitControls(controlling=self.camera)
        self.scene = p3js.Scene(
            children=[self.camera, self.sun_light], background=None
        )
        self.renderer = p3js.Renderer(
            camera=self.camera,
            scene=self.scene,
            controls=[self.orbit],
            width=self.width,
            height=self.height,
            clearColor="#53d5fd",
            antialias=True,
        )

        #create a frame
        global_frame = _ShapeHelper.make_frame_node(scale=0.1)
        self.shapes['global_frame'] = global_frame
        self.scene.add(global_frame)
    
    def place_marker(self, name, pos, scale=0.1):
        #place a sphere marker
        if not name in self.shapes:
            mesh = _ShapeHelper.make_sphere_node(name, scale, 'yellow')
            self.scene.add(mesh)
            self.shapes[name] = mesh
        
        mesh = self.shapes[name]
        mesh.position = (pos[0], pos[1], pos[2])
        #mesh.quaternion = (quat[1], quat[2], quat[3], quat[0])
        mesh.quaternion = (0, 0, 0, 1)
        return

    def remove_marker(self, name):
        if name in self.shapes:
            obj = self.shapes[name]
            self.scene.remove(obj)
            self.shapes.pop(name)
        return
    
    def show(self):
        display(self.renderer)

    def _shape_pose_from_qpos(self, sys, qpos):
        poses = Kinematics._forward_pos(sys, qpos)[0]
        #convert format to position + quaternion, note the quaternion follows threejs convention
        return [(p.trans[0], p.trans[1], p.trans[2], p.rot[1], p.rot[2], p.rot[3], p.rot[0]) for p in poses]

    def create_shapes(self, sys):
        #create meshes according system
        for link in sys.links:
            if link.viz_shape is None:
                #create a frame for it
                mesh = _ShapeHelper.make_frame_node(scale=1.0)
            elif link.viz_shape['shape'] == 'box':
                mesh = _ShapeHelper.make_box_node(link.name, link.viz_shape['size'][0], link.viz_shape['size'][1], link.viz_shape['size'][2], link.viz_shape['color'])
            elif link.viz_shape['shape'] == 'cylinder':
                mesh = _ShapeHelper.make_cylinder_node(link.name, link.viz_shape['size'][0], link.viz_shape['size'][1], link.viz_shape['color'])
            elif link.viz_shape['shape'] == 'sphere':
                mesh = _ShapeHelper.make_sphere_node(link.name, link.viz_shape['size'][0], link.viz_shape['color'])
            else:
                return NotImplementedError

            if link.name in self.shapes:
                print("Warning: {0} exists and will be overwritten!".format(link.name))
            else:
                self.scene.add(mesh)

            self.shapes[link.name] = mesh
        return

    def place_shapes(self, sys, state):
        #update all link poses
        for i, link in enumerate(sys.links):
            mesh = self.shapes[link.name]
            mesh.position = (state.x[i].trans[0], state.x[i].trans[1], state.x[i].trans[2])
            #note the difference of quaternion convention between (w, x, y, z) in brax and (x, y, z, w) in p3js
            mesh.quaternion = (state.x[i].rot[1], state.x[i].rot[2], state.x[i].rot[3], state.x[i].rot[0])  
        return
    
    def place_shapes_qpos(self, sys, qpos):
        poses = self._shape_pose_from_qpos(sys, qpos)
        for i, link in enumerate(sys.links):
            mesh = self.shapes[link.name]
            mesh.position = poses[i][:3]
            mesh.quaternion = poses[i][3:7]
        return

    def animate_qpos_traj(self, sys, qpos_traj, dt):
        duration = len(qpos_traj) * dt
        times = list(np.arange(len(qpos_traj)) * dt)

        #use link to index pos and rot traj
        link_pos_traj = [[] for _ in sys.links]
        link_rot_traj = [[] for _ in sys.links]
        for i in range(len(qpos_traj)):
            poses = self._shape_pose_from_qpos(sys, qpos_traj[i])
            for j, p in enumerate(poses):
                link_pos_traj[j].extend(p[:3])
                link_rot_traj[j].extend(p[3:])
        
        link_tracks = \
            [p3js.VectorKeyframeTrack(name='scene/'+sys.links[i].name+'.position', times=times, values=traj) for i, traj in enumerate(link_pos_traj)] \
            + [p3js.VectorKeyframeTrack(name='scene/'+sys.links[i].name+'.quaternion', times=times, values=traj) for i, traj in enumerate(link_rot_traj)]
                
        clip = p3js.AnimationClip(tracks=link_tracks, duration=duration)
        action = p3js.AnimationAction(p3js.AnimationMixer(self.scene), clip, self.scene)
        return action
    