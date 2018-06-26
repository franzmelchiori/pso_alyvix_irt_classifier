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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from matplotlib import cbook
from matplotlib.colors import LightSource
from matplotlib import cm

import cv2


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

    def __init__(self, gain_function, parameters, solution_space_sizes,
                 inertial_weight=1., cognitive_weight=1., social_weight=1.,
                 serial_number=0):
        self.gain_function = gain_function
        self.parameters = parameters
        self.solution_sizes = solution_space_sizes
        self.weight = np.ones([3], dtype=np.float16)
        self.weight[0] = inertial_weight
        self.weight[1] = cognitive_weight
        self.weight[2] = social_weight
        self.serial_number = serial_number
        self.random = np.ones([3], dtype=np.float16)
        self.speed = self.init_position()
        self.position = self.init_position()
        self.best = self.init_position()
        self.best_value = False
        self.best_swarm = self.init_position()
        self.best_swarm_value = False
        self.samples = []

    def __repr__(self):
        print_message = ''
        if self.serial_number == 0:
            print_message += "    * Swarm\n"
            print_message += "        Best position: {0}\n" \
                             "".format(self.best_swarm)
            print_message += "        Best value: {0}" \
                             "".format(self.best_swarm_value)
        else:
            print_message += "    * Particle {0}\n" \
                             "".format(self.serial_number)
            print_message += "        Weights: {0}\n" \
                             "".format(self.weight)
            print_message += "        Randoms: {0}\n" \
                             "".format(self.random)
            print_message += "        Speed: {0}\n" \
                             "".format(self.speed)
            print_message += "        Position: {0}\n" \
                             "".format(self.position)
            print_message += "        Best position: {0}\n" \
                             "".format(self.best)
            print_message += "        Best value: {0}\n" \
                             "".format(self.best_value)
            print_message += "        Last sample: {0}" \
                             "".format(self.samples[-1])
        return print_message

    def set_random(self):
        self.random = np.ones([3], dtype=np.float16)
        self.random[1:] = np.random.random((1, 2))

    def init_position(self):
        return np.zeros([len(self.solution_sizes)], dtype=np.int16)

    def set_position(self, position):
        self.position = position

    def set_best_swarm(self, position):
        self.best_swarm = position

    def quantize_vector(self, vector):
        vector_expanded = np.array(vector + 0.5, dtype=np.int16)
        position_control = vector_expanded < self.solution_sizes
        position_valid = vector_expanded * np.equal(position_control, True)
        position_correct = (self.solution_sizes - 1) * np.equal(
            position_control, False)
        vector = np.array(position_valid + position_correct, dtype=np.int16)
        return vector

    def sample_gain_function(self):
        function_parameters = []
        sample_coordinate = []
        for dimension, coordinate in enumerate(self.position):
            function_parameters.append(
                self.parameters[dimension].samples[coordinate])
            sample_coordinate.append(coordinate)
        value = self.gain_function(function_parameters)
        sampled_values = [sample[-1] for sample in self.samples]
        if (self.best_value is False) or (value >= max(sampled_values)):
            self.best = self.position
            self.best_value = value
        sample = sample_coordinate
        sample.append(value)
        sample = tuple(sample)
        self.samples.append(sample)
        return sample

    def perturb(self):
        if self.best_value is False:
            intertial_displace = self.init_position()
            cognitive_displace = self.init_position()
            social_displace = self.init_position()
        else:
            intertial_displace = self.speed - 0
            cognitive_displace = self.best - self.position
            social_displace = self.best_swarm - self.position
        self.set_random()
        intertial_term = self.weight[0] * self.random[0] * intertial_displace
        cognitive_term = self.weight[1] * self.random[1] * cognitive_displace
        social_term = self.weight[2] * self.random[2] * social_displace
        self.speed = intertial_term + cognitive_term + social_term
        self.position = self.position + self.speed
        self.position = self.quantize_vector(self.position)
        sample = self.sample_gain_function()
        return sample


class PSO:

    def __init__(self, gain_function, parameters, iterations, particle_amount=3,
                 inertial_weight=1., cognitive_weight=1., social_weight=1.,
                 verbose=False):
        self.gain_function = gain_function
        self.parameters = parameters
        self.iterations = iterations
        self.particle_amount = particle_amount
        self.inertial_weight = inertial_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.solution_sizes = np.array([parameter.samples.size
                                        for parameter in self.parameters],
                                       dtype=np.int16)
        self.particle_space = self.init_particle_space()
        self.particle_result = Particle(
            gain_function=self.gain_function, parameters=self.parameters,
            solution_space_sizes=self.solution_sizes,
            inertial_weight=self.inertial_weight,
            cognitive_weight=self.cognitive_weight,
            social_weight=self.social_weight, serial_number=0)
        self.verbose = verbose

    def __repr__(self):
        print_message = ''
        print_message += "*** PSO\n"
        print_message += "*** Best position: {0}\n" \
                         "".format(self.particle_result.best_swarm)
        print_message += "*** Best value: {0}" \
                         "".format(self.particle_result.best_swarm_value)
        return print_message

    def init_particle_space(self):
        self.particle_space = [Particle(
            gain_function=self.gain_function, parameters=self.parameters,
            solution_space_sizes=self.solution_sizes,
            inertial_weight=self.inertial_weight,
            cognitive_weight=self.cognitive_weight,
            social_weight=self.social_weight, serial_number=particle_number+1)
            for particle_number in range(self.particle_amount)]
        for particle in self.particle_space:
            position = np.array([random.randint(0, coordinate_size - 1)
                                 for coordinate_size
                                 in self.solution_sizes],
                                dtype=np.int16)
            particle.set_position(position)
        return self.particle_space

    def iter_particle_swarm(self):
        particles_data = []
        for particle in self.particle_space:
            particle_data = np.empty((len(self.solution_sizes) + 1,
                                      self.iterations))
            particles_data.append(particle_data)
        for i in range(self.iterations):
            for particle, particle_data in zip(self.particle_space,
                                               particles_data):
                sample = particle.perturb()
                value = sample[-1]
                sampled_best_value = self.particle_result.best_swarm_value
                if value >= sampled_best_value:
                    self.particle_result.best_swarm = particle.position
                    self.particle_result.best_swarm_value = value
                particle_data[:, i] = sample
                if self.verbose:
                    print('** Iteration {0}'.format(i+1))
                    print(particle)
            for particle in self.particle_space:
                particle.set_best_swarm(self.particle_result.best_swarm)
        return particles_data


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

    def altitude_function(self, params):
        return self.z[params[0], params[1]]

    def surface_plot_2d(self):
        fig, ax = plt.subplots()
        ax.imshow(self.z, interpolation='nearest')
        plt.show()

    def particle_trajectory(self, particle_data):
        p = particle_data
        iterations = len(particle_data[0])
        dimensions = 3
        trajectory_data = np.empty((dimensions, iterations))
        for i in range(iterations):
            trajectory_data[:, i] = self.x[int(p[0, i]), int(p[1, i])], \
                                    self.y[int(p[0, i]), int(p[1, i])], \
                                    self.z[int(p[0, i]), int(p[1, i])]+10
        return trajectory_data

    def update_trajectories(self, num, trajectories_data, lines):
        for line, data in zip(lines, trajectories_data):
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
        return lines

    def surface_plot_3d(self, particles_data):
        iterations = len(particles_data[0][0])
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        ls = LightSource(270, 45)
        rgb = ls.shade(self.z, cmap=cm.gist_earth, vert_exag=0.1,
                       blend_mode='soft')

        ax.plot_surface(self.x, self.y, self.z, facecolors=rgb,
                        antialiased=True)

        data = [self.particle_trajectory(particle_data)
                for particle_data in particles_data]
        lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1]+10)[0]
                 for dat in data]
        # points = [ax.scatter3D(self.x[int(p[0][1]), int(p[1][1])],
        #                        self.y[int(p[0][1]), int(p[1][1])],
        #                        self.z[int(p[0][1]), int(p[1][1])]+10,
        #                        s=30, c='r') for p in particles_data]

        animate_trajectories = animation.FuncAnimation(
            fig=fig, func=self.update_trajectories, frames=iterations,
            fargs=(data, lines), interval=24, blit=False)
        plt.show()


def test_ps_optimizer():
    s = 100
    i = 10
    p = 10
    iw = .75
    cw = .5
    sw = .5
    v = True

    param_1 = ParameterSampling(0, s-1, s)
    param_2 = ParameterSampling(0, s-1, s)
    params = [param_1, param_2]
    # print(params)
    fnc = Mountain(s, s)
    gain_function = fnc.altitude_function
    pso = PSO(gain_function=gain_function, parameters=params, iterations=i,
              particle_amount=p, inertial_weight=iw, cognitive_weight=cw,
              social_weight=sw, verbose=v)
    particles_data = pso.iter_particle_swarm()
    print(pso)
    fnc.surface_plot_3d(particles_data)


def test_irt_classifier():
    import alyvix_irt_classifier.contouring as irt

    contouring = irt.Contouring()
    numpy_matrix = contouring.auto_contouring(
        'alyvix_irt_classifier/image_to_classify_01.png', scaling_factor=2.0)
    print(numpy_matrix)

    debug_matrix = contouring.get_debug_matrix()
    debug_image = contouring.get_debug_image()
    cv2.imwrite("alyvix_irt_classifier/debug_matrix.png", debug_matrix)
    cv2.imwrite("alyvix_irt_classifier/debug_image.png", debug_image)


def main():
    """
        https://docs.opencv.org/2.4.9/modules/imgproc/doc/
            feature_detection.html?highlight=canny#canny
            feature_detection.html?highlight=canny#houghlinesp
            filtering.html#getstructuringelement
    """
    import alyvix_irt_classifier.contouring as irt

    i = 10
    p = 10
    iw = .75
    cw = .5
    sw = .5
    v = True

    canny_threshold1 = ParameterSampling(0, 300, 7)
    canny_threshold2 = ParameterSampling(0, 300, 7)
    canny_apertureSize = ParameterSampling(3, 6, 4)
    hough_threshold = ParameterSampling(1, 10, 10)
    hough_minLineLength = ParameterSampling(1, 10, 10)
    hough_maxLineGap = ParameterSampling(1, 10, 10)
    line_angle_cancel = ParameterSampling(0, 9, 10)
    ellipse_width = ParameterSampling(2, 10, 9)
    ellipse_height = ParameterSampling(2, 10, 9)
    text_roi_emptiness = ParameterSampling(0., 1, 11)
    text_roi_proportion = ParameterSampling(0., 3, 11)
    image_roi_emptiness = ParameterSampling(0., 1, 11)
    vline_hw_proportion = ParameterSampling(1, 10, 10)
    vline_w_maxsize = ParameterSampling(1, 10, 10)
    hline_wh_proportion = ParameterSampling(1, 10, 10)
    hline_h_maxsize = ParameterSampling(1, 10, 10)
    rect_w_minsize = ParameterSampling(1, 10, 10)
    rect_h_minsize = ParameterSampling(1, 10, 10)
    rect_w_maxsize_01 = ParameterSampling(600, 1000, 5)
    rect_h_maxsize_01 = ParameterSampling(50, 250, 5)
    rect_w_maxsize_02 = ParameterSampling(50, 250, 5)
    rect_h_maxsize_02 = ParameterSampling(600, 1000, 5)
    rect_hw_proportion = ParameterSampling(1, 10, 10)
    rect_hw_w_maxsize = ParameterSampling(1, 10, 10)
    rect_wh_proportion = ParameterSampling(1, 10, 10)
    rect_wh_h_maxsize = ParameterSampling(1, 10, 10)
    hrect_proximity = ParameterSampling(2, 20, 10)
    vrect_proximity = ParameterSampling(2, 20, 10)
    vrect_others_proximity = ParameterSampling(20, 200, 10)
    hrect_others_proximity = ParameterSampling(20, 200, 10)
    params = [canny_threshold1, canny_threshold2, canny_apertureSize,
              hough_threshold, hough_minLineLength, hough_maxLineGap,
              line_angle_cancel, ellipse_width, ellipse_height,
              text_roi_emptiness, text_roi_proportion, rect_w_minsize,
              rect_h_minsize, rect_w_maxsize_01, rect_h_maxsize_01,
              rect_w_maxsize_02, rect_h_maxsize_02, rect_hw_proportion,
              rect_hw_w_maxsize, rect_wh_proportion, rect_wh_h_maxsize,
              hrect_proximity, vrect_proximity, vrect_others_proximity,
              hrect_others_proximity]

    contouring = irt.Contouring(
        canny_threshold1.samples[0],
        canny_threshold2.samples[0],
        canny_apertureSize.samples[0],
        hough_threshold.samples[0],
        hough_minLineLength.samples[0],
        hough_maxLineGap.samples[0],
        line_angle_cancel.samples[0],
        ellipse_width.samples[0],
        ellipse_height.samples[0],
        text_roi_emptiness.samples[0],
        text_roi_proportion.samples[0],
        image_roi_emptiness.samples[0],
        vline_hw_proportion.samples[0],
        vline_w_maxsize.samples[0],
        hline_wh_proportion.samples[0],
        hline_h_maxsize.samples[0],
        rect_w_minsize.samples[0],
        rect_h_minsize.samples[0],
        rect_w_maxsize_01.samples[0],
        rect_h_maxsize_01.samples[0],
        rect_w_maxsize_02.samples[0],
        rect_h_maxsize_02.samples[0],
        rect_hw_proportion.samples[0],
        rect_hw_w_maxsize.samples[0],
        rect_wh_proportion.samples[0],
        rect_wh_h_maxsize.samples[0],
        hrect_proximity.samples[0],
        vrect_proximity.samples[0],
        vrect_others_proximity.samples[0],
        hrect_others_proximity.samples[0]
    )
    numpy_matrix = contouring.auto_contouring(
        'alyvix_irt_classifier/image_to_classify_01.png')
    # cv2.imwrite("numpy_matrix_i.png", numpy_matrix[:, :, 0]*255)
    # cv2.imwrite("numpy_matrix_r.png", numpy_matrix[:, :, 1]*255)
    # cv2.imwrite("numpy_matrix_t.png", numpy_matrix[:, :, 2]*255)

    # gain_function =

    # pso = PSO(gain_function=gain_function, parameters=params, iterations=i,
    #           particle_amount=p, inertial_weight=iw, cognitive_weight=cw,
    #           social_weight=sw, verbose=v)
    # particles_data = pso.iter_particle_swarm()
    # print(pso)


if __name__ == '__main__':
    test_ps_optimizer()
    # test_irt_classifier()
    # main()
