from articulated_dynamics.articulated_system import *

def inertia_from_geom(shape, size, density=1000):
    if shape == 'box':
        mass = size[0] * size[1] * size[2] * density
        it = nplib.array([[size[1]**2+size[2]**2, 0, 0], [0, size[0]**2+size[2]**2, 0], [0, 0, size[0]**2+size[1]**2]]) * mass / 12.
    elif shape == 'cylinder':
        return NotImplementedError
    elif shape == 'sphere':
        return NotImplementedError
    else:
        return NotImplementedError
    return mass, it

class CartPole(ArticulatedSystem):
    def __init__(self, pole_len=0.6, pole_den=10) -> None:
        super().__init__(num_links=2)
        #link 0
        m, it = inertia_from_geom('box', [0.2, 0.05, 0.1], density=10)
        self.links[0].name = 'cart'
        self.links[0].joint_frame = Transform()
        self.links[0].joint_dofs = Prismatic()
        self.links[0].inertia = Inertia.from_it_and_mass(it=it, mass=m)

        self.links[0].viz_shape['size'] = [0.2, 0.1, 0.4]

        #link 1
        m, it = inertia_from_geom('box', [0.1, pole_len, 0.1], density=pole_den)
        self.links[1].name = 'pole'
        self.links[1].prev_transform = Transform(rot=nplib.array([0.7071068, 0, 0.7071068, 0]))
        self.links[1].joint_frame = Transform(trans=nplib.array([0, -pole_len/2, 0]))
        self.links[1].joint_dofs = Revolute()
        self.links[1].inertia = Inertia.from_it_and_mass(it=it, mass=m)

        self.links[1].viz_shape['size'] = [0.1, pole_len, 0.1]

class ThreeLinks(ArticulatedSystem):
    def __init__(self, link_len=0.3, link_den=2700) -> None:
        super().__init__(num_links=3)
        link_length = link_len
        link_width = 0.05
        m, it = inertia_from_geom('box', [link_width, link_length, link_width], link_den)
        #specify link parameters
        #link 0
        self.links[0].joint_frame = Transform(trans=nplib.array([0, -link_length/2., 0]))
        self.links[0].joint_dofs = Revolute()   #1 DOF revolute joint along z axis
        self.links[0].inertia = Inertia.from_it_and_mass(it=it, mass=m)

        self.links[0].viz_shape['size'] = [link_width, link_length, link_width]

        #link 1
        self.links[1].prev_transform = Transform(trans=nplib.array([0, link_length/2., 0]))
        self.links[1].joint_frame = Transform(trans=nplib.array([0, -link_length/2., 0]))
        self.links[1].joint_dofs = Revolute()   #1 DOF revolute joint along z axis
        self.links[1].inertia = Inertia.from_it_and_mass(it=it, mass=m)

        self.links[1].viz_shape['size'] = [link_width, link_length, link_width]

        #link 2
        self.links[2].prev_transform = Transform(trans=nplib.array([0, link_length/2., 0]))
        self.links[2].joint_frame = Transform(trans=nplib.array([0, -link_length/2., 0]))
        self.links[2].joint_dofs = Revolute()   #1 DOF revolute joint along z axis
        self.links[2].inertia = Inertia.from_it_and_mass(it=it, mass=m)

        self.links[2].viz_shape['size'] = [link_width, link_length, link_width]
