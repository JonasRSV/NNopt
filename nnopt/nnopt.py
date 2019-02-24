import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time


class Optimizer():
    SCOPE = "nnopt"

    def __init__(
            self,
            F: "F: R^N -> R^1",
            N: int,
            surrogate_hidden_layer: int = 60,
            model_hidden_layer: int = 60,
            memory_hidden_layer: int = 60,
            mem_model_out: int = 5,
            Rn:
            "[Range_0, Range_1...Range_n] N = Range_n = (Min_n, Max_n)" = None,
            seed=None):

        self.N = N

        if Rn and len(Rn) != N:
            raise Exception("Ranges must match domain")

        self.F = F

        self.N_samples = []
        self.M_samples = []

        with tf.variable_scope(Optimizer.SCOPE, reuse=tf.AUTO_REUSE):
            self.N_gen = None
            if Rn:
                self.N_gen = tf.concat(
                    [
                        tf.random_uniform([1], mini, maxi, seed=seed)
                        for mini, maxi in Rn
                    ],
                    axis=0)
            else:
                self.N_gen = tf.concat(
                    [
                        tf.random_uniform([1], -10, 10, seed=seed)
                        for _ in range(N)
                    ],
                    axis=0)

            self.IN = tf.placeholder(tf.float32, shape=[None, N])
            self.OBS = tf.placeholder(tf.float32, shape=[None])

            with tf.variable_scope("surrogate", reuse=tf.AUTO_REUSE):
                hidden = tf.layers.dense(
                    self.IN,
                    surrogate_hidden_layer,
                    activation=tf.math.sigmoid,
                    use_bias=True)
                self.OUT = tf.squeeze(
                    tf.layers.dense(hidden, 1, activation=None, use_bias=True))

            surrogate_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/surrogate".format(Optimizer.SCOPE))

            surrogate_error = tf.losses.mean_squared_error(self.OBS, self.OUT)

            surrogate_grads = tf.gradients(surrogate_error,
                                           surrogate_variables)
            self.surrogate_opt = tf.train.AdamOptimizer(
                learning_rate=0.001).apply_gradients(
                    zip(surrogate_grads, surrogate_variables))

            with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                hidden = tf.layers.dense(
                    self.IN,
                    memory_hidden_layer,
                    activation=tf.math.sigmoid,
                    use_bias=True)
                model_out = tf.layers.dense(
                    hidden, mem_model_out, activation=None, use_bias=True)

            with tf.variable_scope("memory", reuse=tf.AUTO_REUSE):
                hidden = tf.layers.dense(
                    self.IN,
                    memory_hidden_layer,
                    activation=tf.math.sigmoid,
                    use_bias=True)
                memory_out = tf.layers.dense(
                    hidden, mem_model_out, activation=None, use_bias=True)

            memory_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/memory".format(Optimizer.SCOPE))

            self.entropy = tf.losses.mean_squared_error(model_out, memory_out)
            memory_grads = tf.gradients(self.entropy, memory_variables)
            self.memory_opt = tf.train.AdamOptimizer(
                learning_rate=0.001).apply_gradients(
                    zip(memory_grads, memory_variables))

            self.IN_grads = tf.squeeze(tf.gradients(self.OUT, self.IN))
            self.IN_hessian = tf.linalg.inv(
                tf.squeeze(tf.hessians(self.OUT, self.IN)))

        local_init = tf.initializers.variables(
            tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=Optimizer.SCOPE))
        self.S = tf.Session()
        self.S.run(local_init)

    def sample(self, N=[], verbose=True) -> None:
        if len(N) == 0:
            N = self.S.run(self.N_gen)

        M = self.F(*N)
        if verbose:
            print("sample", N, "target", M)

        if self.N == 1:
            self.N_samples.append(np.array([N]))
        else:
            self.N_samples.append(N)

        self.M_samples.append(M)

    def fit(self, epochs=10) -> float:
        for _ in range(epochs):
            self.S.run(
                (self.surrogate_opt, self.memory_opt),
                feed_dict={
                    self.IN: self.N_samples,
                    self.OBS: self.M_samples
                })

    def optimize(self):
        """ Work on heuristics here. """

        suggestions = []
        """ All of these belong to potential regions. """
        for p in range(len(self.N_samples)):

            # newton step is verrry unstable here it seems
            # gradients, inv_hess = self.S.run(
            # (self.IN_grads, self.IN_hessian), feed_dict={
            # self.IN: [self.N_samples[p]]
            # })

            # suggestion = self.N_samples[p] - inv_hess @ gradients

            suggestion = self.N_samples[p]

            # A few gradient steps is more stable
            for _ in range(10):
                gradients = self.S.run(
                    (self.IN_grads), feed_dict={
                        self.IN: [suggestion]
                    })

                suggestion = suggestion + 0.1 * gradients

            pred, entropy = self.S.run(
                (self.OUT, self.entropy), feed_dict={
                    self.IN: [suggestion]
                })

            suggestions.append((suggestion, pred * abs(entropy), pred))

        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[0]

    def run(self, random=100, optimization=100, fitting=100, verbose=True):
        for i in range(random):
            self.sample(verbose=verbose)

        if verbose:
            print("Best random", self.N_samples[np.argmax(self.M_samples)],
                  max(self.M_samples))

        self.fit(fitting)

        for i in range(optimization):
            sugg, _, _ = self.optimize()
            self.sample(sugg, verbose=verbose)
            self.fit(fitting)

        return self.N_samples[np.argmax(self.M_samples)], max(self.M_samples)

    def pred_test(self, samples, obs):
        return samples, obs, self.S.run(
            (self.OUT, self.entropy), feed_dict={
                self.IN: samples
            })
