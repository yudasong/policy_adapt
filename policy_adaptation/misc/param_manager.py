import numpy as np

HC_dict = {
    "mass": [0,1,2,3,4,5,6,7],
    "friction":[8],
    "restitution":[9],
    "gravity":[10]
}

class HalfCheetahParamManager:
    def __init__(self, env):
        self.env = env
        self.mass_range = [0.        , 6.36031332, 1.53524804, 1.58093995, 1.0691906 ,1.42558747, 1.17885117, 0.84986945]
        self.fric_range = [0.5, 2.0]
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.tilt_z_range = [-0.18, 0.18]
        self.g_range = [-9.81]

        self.controllable_param = [10]
        self.activated_param = [10]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1


    def get_params(self):

        mass_param = []
        for bid in range(0, 8):
            cur_mass = self.env.model.body_mass[bid]
            mass_param.append(cur_mass / self.mass_range[bid])


        cur_friction = self.env.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.fric_range[0]) / (self.fric_range[1] - self.fric_range[0])

        cur_restitution = self.env.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_gravity = self.env.model.opt.gravity[-1]
        gravity_param = (cur_gravity / self.g_range[0] )

        params = np.array(mass_param + [friction_param, rest_param ,gravity_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x, task=None):
        if task is not None:
            self.controllable_param = HC_dict[task]
            self.activated_param = HC_dict[task]



        cur_id = 0
        for bid in range(0, 8):
            if bid in self.controllable_param:
                mass = x[cur_id] * self.mass_range[bid]
                self.env.model.body_mass[bid] = mass
                cur_id += 1

        if 8 in self.controllable_param:
            friction = x[cur_id] * (self.fric_range[1] - self.fric_range[0]) + self.fric_range[0]
            self.env.model.geom_friction[-1][0] = friction
            cur_id += 1
        if 9 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.env.model.geom_solref)):
                self.env.model.geom_solref[bn][1] = restitution
            cur_id += 1

        if 10 in self.controllable_param:
            self.env.model.opt.gravity[:] *= x[cur_id]
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_params()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_params()))
        return x

    def get_mask(self, task=None):

        if task is not None:
            self.controllable_param = HC_dict[task]
            self.activated_param = HC_dict[task]

        x = np.ones(len(self.controllable_param))
        return x



Ant_dict = {
    "mass": [1,2,3,4,5,6,7,8,9,10,11,12,13],
    "friction":[14],
    "restitution":[15],
    "gravity":[16]
}



class AntParamManager:
    def __init__(self, env):
        self.env = env
        self.mass_range = [0.        , 0.32724923, 0.03647693, 0.03647693, 0.06491138,0.03647693, 0.03647693, 0.06491138, 0.03647693, 0.03647693, 0.06491138, 0.03647693, 0.03647693, 0.06491138]
        self.fric_range = [0.5, 2.0]
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.tilt_z_range = [-0.18, 0.18]
        self.g_range = [-9.81]

        self.controllable_param = [16]
        self.activated_param = [16]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1


    def get_params(self):

        mass_param = []
        for bid in range(0, 14):
            cur_mass = self.env.model.body_mass[bid]
            mass_param.append(cur_mass / self.mass_range[bid])


        cur_friction = self.env.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.fric_range[0]) / (self.fric_range[1] - self.fric_range[0])

        cur_restitution = self.env.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_gravity = self.env.model.opt.gravity[-1]
        gravity_param = (cur_gravity / self.g_range[0] )

        params = np.array(mass_param + [friction_param, rest_param ,gravity_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x, task=None):
        if task is not None:
            self.controllable_param = Ant_dict[task]
            self.activated_param = Ant_dict[task]

        for bid in range(0, 14):
            if bid in self.controllable_param:
                mass = x[cur_id] * self.mass_range[bid]
                self.env.model.body_mass[bid] = mass
                cur_id += 1

        if 14 in self.controllable_param:
            friction = x[cur_id] * (self.fric_range[1] - self.fric_range[0]) + self.fric_range[0]
            self.env.model.geom_friction[-1][0] = friction
            cur_id += 1
        if 15 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.env.model.geom_solref)):
                self.env.model.geom_solref[bn][1] = restitution
            cur_id += 1

        if 16 in self.controllable_param:
            self.env.model.opt.gravity[:] *= x[cur_id]
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_params()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_params()))
        return x

    def get_mask(self, task=None):

        if task is not None:
            self.controllable_param = Ant_dict[task]
            self.activated_param = Ant_dict[task]

        x = np.ones(len(self.controllable_param))
        return x
