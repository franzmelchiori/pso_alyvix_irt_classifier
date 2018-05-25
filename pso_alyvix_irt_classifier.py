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
import matplotlib.pyplot as plt


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

    def __init__(self, solution_dimensions, inertial_weight=1.,
                 cognitive_weight=1., social_weight=1):
        self.solution_dimensions = solution_dimensions
        self.weight = np.ones([3], dtype=np.float16)
        self.weight[0] = inertial_weight
        self.weight[1] = cognitive_weight
        self.weight[2] = social_weight
        self.random = np.ones([3], dtype=np.float16)
        self.set_random()
        self.speed = self.init_position()
        self.position = self.init_position()
        self.best = self.init_position()

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
        return np.zeros([self.solution_dimensions], dtype=np.uint8)


class PSO:

    def __init__(self, parameters, particle_amount):
        self.parameters = parameters
        self.particle_amount = particle_amount
        self.solution_space_sizes = np.array(
            [parameter.samples.size for parameter in self.parameters],
            dtype=np.int16)
        self.solution_dimensions = len(self.solution_space_sizes)
        self.particle_space = np.zeros(0, dtype=np.byte)
        self.init_particle_space()
        self.particle_result = Particle(self.solution_dimensions)
        self.result_space = np.zeros(0, dtype=np.byte)
        self.build_result_space()

    def __repr__(self):
        print_message = ''
        # print_message += '{0}\n'.format(self.parameters_types)
        for particle in self.particle_space:
            print_message += '{0}\n\n'.format(particle)
        print_message += '{0}\n'.format(self.result_space)
        return print_message

    # def get_parameters_types(self):
    #     parameters_names = ['p{0}'.format(parameter_number)
    #                         for parameter_number in range(len(self.parameters))]
    #     parameters_types = [parameter.samples_type
    #                         for parameter in self.parameters]
    #     self.parameters_types = np.dtype(zip(parameters_names,
    #                                          parameters_types))
    #     return self.parameters_types

    def init_particle_space(self):
        self.particle_space = [Particle(self.solution_dimensions)
                               for particle_number
                               in range(self.particle_amount)]
        return self.particle_space

    def build_result_space(self):
        parameters_sizes = [parameter.samples.size
                            for parameter in self.parameters]
        self.result_space = np.zeros(parameters_sizes, dtype=np.float16)
        return self.result_space

    def iter_particle_swarm(self):
        pass


def main():
    param_1 = ParameterSampling(0, 2, 5)
    param_2 = ParameterSampling(0., 10, 7)
    param_3 = ParameterSampling(0, 1, 2)
    params = [param_1, param_2, param_3]
    pso_test = PSO(params, 2)
    print(pso_test)


if __name__ == '__main__':
    main()
