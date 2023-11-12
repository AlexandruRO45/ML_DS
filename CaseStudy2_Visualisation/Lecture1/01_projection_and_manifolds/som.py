#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Self-organizing map
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np


def fromdistance(fn, shape, center=None, dtype=float):
    def distance(*args):
        d = 0
        for i in range(len(shape)):
            d += ((args[i] - center[i]) / float(max(1, shape[i] - 1))) ** 2
        return np.sqrt(d) / np.sqrt(len(shape))

    if center == None:
        center = np.array(list(shape)) // 2
    return fn(np.fromfunction(distance, shape, dtype=dtype))


def Gaussian(shape, center, sigma=0.5):
    """ """

    def g(x):
        return np.exp(-x ** 2 / sigma ** 2)

    return fromdistance(g, shape, center)


class SOM:
    """ Self-organizing map """

    def __init__(self, *args):
        """ Initialize som """
        self.codebook = np.zeros(args)
        # for interpolation
        self.meshgrid = np.meshgrid(
            np.linspace(0, 1, args[0]), np.linspace(0, 1, args[1])
        )
        self.reset()

    def reset(self):
        """ Reset weights """
        self.codebook = np.random.random(self.codebook.shape)

    def score(self, sample, width=1.0):
        """ score a sample """
        D = ((self.codebook - sample) ** 2).sum(axis=-1)
        return np.exp(-(D.reshape(self.codebook.shape[0:2])) ** 2 / (2 * width ** 2))

    def interpolate(self, sample, width=1.0):
        """ interpolate a sample """
        D = ((self.codebook - sample) ** 2).sum(axis=-1)
        weights = np.exp(-(D.reshape(self.codebook.shape[0:2])) ** 2 / (2 * width ** 2))
        weights = weights / np.sum(weights)
        return np.stack([self.meshgrid[0] / weights, self.meshgrid[1] / weights])

    def classify(self, sample):
        """ classify a sample """
        D = ((self.codebook - sample) ** 2).sum(axis=-1)
        winner = np.unravel_index(np.argmin(D), D.shape)
        return winner

    def density(self, samples):
        w, h, d = self.codebook.shape
        density = np.zeros((w, h))

        for sample in samples:
            D = ((self.codebook - sample) ** 2).sum(axis=-1)
            winner = np.unravel_index(np.argmin(D), D.shape)
            density[winner] += 1

        return density

    def get_nearest(self, x, y, data):
        D = ((self.codebook[x, y, :] - data) ** 2).sum(axis=-1)
        winner = np.argmin(D)
        return data[winner, :], winner

    def get_n_nearest_indices(self, x, y, data):
        D = ((self.codebook[x, y, :] - data) ** 2).sum(axis=-1)
        winner = np.argsort(D)
        return winner, D

    def learn(self, samples, epochs=10000, sigma=(10, 0.001), lrate=(0.5, 0.005)):
        """ Learn samples """
        sigma_i, sigma_f = sigma
        lrate_i, lrate_f = lrate

        lrate = lrate_i
        sigma = sigma_i
        s = samples.shape[0]
        for i in range(epochs):
            if i % 500 == 0:
                print(
                    "Epoch \t %d /\t %d \tLrate:%.2f\t Sigma:%.2f"
                    % (i, epochs, lrate, sigma)
                )
            # Adjust learning rate and neighborhood
            t = i / float(epochs)
            lrate = lrate_i * (lrate_f / float(lrate_i)) ** t
            sigma = sigma_i * (sigma_f / float(sigma_i)) ** t

            # Get random sample
            index = np.random.randint(0, s)
            data = samples[index]

            # Get index of nearest node (minimum distance)

            D = ((self.codebook - data) ** 2).sum(axis=-1)
            winner = np.unravel_index(np.argmin(D), D.shape)

            # Generate a Gaussian centered on winner
            G = Gaussian(D.shape, winner, sigma)
            G = np.nan_to_num(G)

            # Move nodes towards sample according to Gaussian
            delta = self.codebook - data
            for i in range(self.codebook.shape[-1]):
                self.codebook[..., i] -= lrate * G * delta[..., i]

