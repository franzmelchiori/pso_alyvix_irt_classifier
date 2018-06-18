#! /usr/bin/python

"""
    PSO for Alyvix IRTC:
        Particle Swarm Optimizer for
        Alyvix Image-Rect-Text Classifier
    Copyright (C) 2018 Francesco Melchiori
    <https://www.francescomelchiori.com/>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import random
import numpy as np
import scipy.special as sps
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource


class ParameterSampling:

    def __init__(self, lowerbound, upperbound, samples_amount=2,
                 sampling_type='linear'):
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        if samples_amount <= 255:
            self.samples_amount = samples_amount
        else:
            self.samples_amount = 255
        self.sampling_type = sampling_type
        if type(float()) in (type(self.lowerbound), type(self.upperbound)):
            self.samples_type = np.float16
        else:
            self.samples_type = np.int16
        self.samples = np.zeros(shape=self.samples_amount,
                                dtype=self.samples_type)
        if sampling_type == 'linear':
            self.linear_sampling()

    def __repr__(self):
        print_message = ''
        print_message += "'{0}'".format(self.samples)
        return print_message

    def __plot__(self):
        plt.plot(np.linspace(start=0, stop=self.samples_amount,
                             num=self.samples_amount, endpoint=False),
                 self.samples, color='black', linestyle='None', marker='o')
        plt.show()

    def linear_sampling(self):
        self.samples = np.linspace(start=self.lowerbound, stop=self.upperbound,
                                   num=self.samples_amount,
                                   dtype=self.samples_type)
        return self.samples


class Particle:

    def __init__(self, gain_function, solution_space_sizes, inertial_weight=1.,
                 cognitive_weight=1., social_weight=1):
        self.gain_function = gain_function
        self.solution_space_sizes = solution_space_sizes
        self.solution_dimensions = len(self.solution_space_sizes)
        self.weight = np.ones([3], dtype=np.float16)
        self.weight[0] = inertial_weight
        self.weight[1] = cognitive_weight
        self.weight[2] = social_weight
        self.random = np.ones([3], dtype=np.float16)
        self.set_random()
        self.speed = self.init_position()
        self.position = self.init_position()
        self.best = self.init_position()
        self.best_swarm = self.init_position()
        self.samples = {}

    def __repr__(self):
        print_message = ''
        print_message += "Particle weights: '{0}'\n".format(self.weight)
        print_message += "Particle randoms: '{0}'\n".format(self.random)
        print_message += "Particle speed: '{0}'\n".format(self.speed)
        print_message += "Particle position: '{0}'\n".format(self.position)
        print_message += "Particle best position: '{0}'".format(self.best)
        return print_message

    def set_random(self):
        self.random[1] *= np.array([random.random()], dtype=np.float16)
        self.random[2] *= np.array([random.random()], dtype=np.float16)

    def init_position(self):
        return np.zeros([self.solution_dimensions], dtype=np.int16)

    def set_position(self, position, init=False):
        self.position = position
        if init:
            self.best_swarm = position

    def set_best_swarm(self, position):
        self.best_swarm = position

    def bouncing(self, component_displace):
        for (dimension_number, dimension_upperbound) in enumerate(
                self.solution_space_sizes):
            intertial_rebound = abs(component_displace[dimension_number])
            (rebound_number, rebound_displace) = divmod(
                intertial_rebound, dimension_upperbound)
            if rebound_number % 2 is not False:
                component_displace[dimension_number] = rebound_displace
            else:
                component_displace[dimension_number] = dimension_upperbound - \
                                                       rebound_displace

    def quantize_position(self):
        """
            integer position
            from position = 0
            to position < self.solution_space_sizes[<dim>]
        """
        self.position = np.array(self.position + 0.5, dtype=np.int16)
        position_control = self.position < self.solution_space_sizes
        position_valid = self.position * np.equal(position_control, True)
        position_correct = (self.solution_space_sizes - 1) * np.equal(
            position_control, False)
        self.position = np.array(position_valid + position_correct,
                                 dtype=np.int16)

    def sample_gain_function(self):
        position = self.position[0], self.position[1]
        value = self.gain_function(*position)
        print(value)
        try:
            if value > max(self.samples):
                self.best = position
        except ValueError:
            pass
        self.samples[value] = position

    def perturb(self):
        intertial_displace = self.speed - 0
        cognitive_displace = self.best - self.position
        social_displace = self.best_swarm - self.position
        intertial_term = self.weight[0] * self.random[0] * intertial_displace
        cognitive_term = self.weight[1] * self.random[1] * cognitive_displace
        social_term = self.weight[2] * self.random[2] * social_displace
        self.bouncing(intertial_term)
        self.bouncing(cognitive_term)
        self.bouncing(social_term)
        self.speed = intertial_term + cognitive_term + social_term
        self.position = self.position + self.speed
        self.quantize_position()
        self.sample_gain_function()


class PSO:

    def __init__(self, gain_function, parameters, particle_amount=3):
        self.gain_function = gain_function
        self.parameters = parameters
        self.particle_amount = particle_amount
        self.parameter_types = [parameter.samples_type
                                for parameter in self.parameters]
        self.solution_space_sizes = np.array(
            [parameter.samples.size for parameter in self.parameters],
            dtype=np.int16)
        self.solution_dimensions = len(self.solution_space_sizes)
        self.particle_space = np.zeros(0, dtype=np.byte)
        self.init_particle_space()
        self.particle_result = Particle(self.gain_function,
                                        self.solution_space_sizes)
        self.result_space = np.zeros(0, dtype=np.byte)
        self.build_result_space()

    def __repr__(self):
        print_message = ''
        # print_message += '{0}\n'.format(self.parameters_types)
        for particle in self.particle_space:
            print_message += '{0}\n\n'.format(particle)
        # print_message += '{0}\n'.format(self.result_space)
        return print_message

    def init_particle_space(self):
        self.particle_space = [Particle(self.gain_function,
                                        self.solution_space_sizes)
                               for particle_number
                               in range(self.particle_amount)]
        for particle in self.particle_space:
            position = np.array([random.randint(0, coordinate_size-1)
                                 for coordinate_size
                                 in self.solution_space_sizes],
                                dtype=np.int16)
            particle.set_position(position, True)
        return self.particle_space

    def build_result_space(self):
        parameters_sizes = [parameter.samples.size
                            for parameter in self.parameters]
        self.result_space = np.zeros(parameters_sizes, dtype=np.float16)
        return self.result_space

    def iter_particle_swarm(self):
        for particle in self.particle_space:
            particle.perturb()
        return True


class Mountain:

    def __init__(self, param_1_dim, param_2_dim):
        filename = cbook.get_sample_data('jacksboro_fault_dem.npz',
                                         asfileobj=False)
        with np.load(filename) as dem:
            z = dem['elevation']
            nrows, ncols = z.shape
            x = np.linspace(dem['xmin'], dem['xmax'], ncols)
            y = np.linspace(dem['ymin'], dem['ymax'], nrows)
            x, y = np.meshgrid(x, y)
            region = np.s_[0:param_1_dim, 0:param_2_dim]
            self.x, self.y, self.z = x[region], y[region], z[region]

    def altitude_function(self, param_1, param_2):
        return self.z[param_1, param_2]

    def surface_plot_2d(self):
        fig, ax = plt.subplots()
        ax.imshow(self.z, interpolation='nearest')
        plt.show()

    def surface_plot_3d(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ls = LightSource(270, 45)
        rgb = ls.shade(self.z, cmap=cm.gist_earth, vert_exag=0.1,
                       blend_mode='soft')
        ax.plot_surface(self.x, self.y, self.z, rstride=1, cstride=1,
                        facecolors=rgb, linewidth=0, antialiased=False,
                        shade=False)
        plt.show()


def main():
    param_1 = ParameterSampling(0, 99, 100)
    param_2 = ParameterSampling(0, 99, 100)
    params = [param_1, param_2]
    # print(params)
    fnc = Mountain(100, 100)
    gain_function = fnc.altitude_function
    fnc.surface_plot_2d()
    pso = PSO(gain_function, params, 3)
    for k in range(10):
        print('*** start iter ***')
        pso.iter_particle_swarm()
        print('')


if __name__ == '__main__':
    main()
