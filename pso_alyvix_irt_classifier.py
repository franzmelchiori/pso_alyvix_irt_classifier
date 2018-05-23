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


import numpy as np
import matplotlib.pyplot as plt


class ParameterSampling:

    def __init__(self, lowerbound, upperbound, samples_amount=2,
                 sampling_type='linear'):
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.sampling_type = sampling_type
        self.samples_amount = samples_amount
        if type(float()) in (type(self.lowerbound), type(self.upperbound)):
            self.samples_type = np.float64
        else:
            self.samples_type = np.int64
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


class PSO:

    def __init__(self, parameters):
        self.parameters = parameters
        self.solution_space = np.zeros(1, dtype=np.int64)
        self.parameters_types = np.dtype('int64')
        self.get_parameters_types()
        self.build_solution_space()

    def __repr__(self):
        print_message = ''
        print_message += '{0}\n'.format(self.parameters_types)
        print_message += '{0}\n'.format(self.solution_space)
        return print_message

    def get_parameters_types(self):
        parameters_types = ''
        for parameter in self.parameters:
            if parameter.samples.dtype is np.dtype('float'):
                parameters_types += 'f8,'
            else:
                parameters_types += 'i8,'
        self.parameters_types = np.dtype(parameters_types[:-1])
        return self.parameters_types

    def build_solution_space(self):
        parameters_sizes = []
        for parameter in self.parameters:
            parameters_sizes.append(parameter.samples.size)
        self.solution_space = np.zeros(parameters_sizes)
        return self.solution_space


def main():
    # pso_test = PSO()
    # print(pso_test)
    param_1 = ParameterSampling(0, 2, 5)
    param_2 = ParameterSampling(0., 10, 5)
    params = [param_1, param_2]
    pso = PSO(params)
    print(pso)


if __name__ == '__main__':
    main()
    print('end.')
