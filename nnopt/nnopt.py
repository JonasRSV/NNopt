import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.covariance import EmpiricalCovariance
import numpy as np
import time
import random


class Optimizer():

    def __init__(
            self,
            F: "F: R^N -> R^1",
            N: int,
            surrogate_hidden_layer: int = 60,
            kernel_common: int = 60,
            Rn:
            "[Range_0, Range_1...Range_n] N = Range_n = (Min_n, Max_n)" = None,
            convergence_limit: float = 1e-5,
            codomain_norm:
            "F: R^1 x R^1 -> R^1" = lambda x, y: np.mean(np.abs(x - y)),
            scope: str = "nnopt",
            seed=None):

        self.N = N
        self.Rn = Rn

        self.convergence_limit = convergence_limit
        self.codomain_norm = codomain_norm

        self.minis = None
        self.maxis = None
        if Rn and len(Rn) != N and isinstance(Rn, list):
            raise Exception("Ranges must match domain")
        elif isinstance(Rn, tuple):
            self.minis, self.maxis = Rn[0], Rn[1]
        elif Rn:
            self.minis, self.maxis = zip(*Rn)

        self.F = F

        self.N_samples = []
        self.M_samples = []
        self.samples = 0

        self.k1_features = []
        self.k2_features = []
        self.kernel_labels = []

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.kernel_lr = tf.Variable(1e-3)
            self.surrogate_lr = tf.Variable(1e-3)

            self.N_gen = None
            if Rn and isinstance(Rn, list):
                self.N_gen = tf.concat(
                    [
                        tf.random_uniform([1], mini, maxi, seed=seed)
                        for mini, maxi in Rn
                    ],
                    axis=0)
            elif Rn and isinstance(Rn, tuple):
                self.N_gen = tf.concat(
                    [
                        tf.random_uniform([1], Rn[0], Rn[1], seed=seed)
                        for _ in range(N)
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
                hidden = tf.layers.dense(
                    hidden,
                    surrogate_hidden_layer,
                    activation=tf.math.sigmoid,
                    use_bias=True)
                hidden = tf.layers.dense(
                    hidden,
                    surrogate_hidden_layer,
                    activation=tf.math.sigmoid,
                    use_bias=True)
                hidden = tf.layers.dense(
                    hidden,
                    surrogate_hidden_layer,
                    activation=None,
                    use_bias=True)
                self.OUT = tf.squeeze(
                    tf.layers.dense(hidden, 1, activation=None, use_bias=True))

            surrogate_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/surrogate".format(scope))

            self.surrogate_error = tf.losses.mean_squared_error(
                self.OBS, self.OUT) + 1e-6 * tf.reduce_mean(
                    [tf.reduce_mean(var) for var in surrogate_variables])

            surrogate_grads = tf.gradients(self.surrogate_error,
                                           surrogate_variables)

            surrogate_grads = [
                tf.clip_by_value(grad, -10, 10) for grad in surrogate_grads
            ]

            self.surrogate_opt = tf.train.AdamOptimizer(
                learning_rate=self.surrogate_lr).apply_gradients(
                    zip(surrogate_grads, surrogate_variables))

            self.IN_grads = tf.squeeze(tf.gradients(self.OUT, self.IN), [0, 1])
            self.IN_hessian = tf.linalg.inv(
                tf.squeeze(tf.hessians(self.OUT, self.IN), axis=[0, 1, 3]) +
                np.eye(self.N) * 1e-7)

            self.K1 = tf.placeholder(tf.float32, shape=[None, N])
            self.K2 = tf.placeholder(tf.float32, shape=[None, N])

            self.K_labels = tf.placeholder(tf.float32, shape=[None])

        self.local_init = tf.initializers.variables(
            tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))

        self.S = tf.Session()
        self.S.run(self.local_init)

    def sample(self, N=[], verbose=2) -> None:
        if len(N) == 0:
            N = self.S.run(self.N_gen)

        M = self.F(*N)
        if verbose == 2:
            print("I", self.samples, "target", M)

        if verbose == 1 and self.M_samples and M > max(self.M_samples):
            print("I", self.samples, "target", M)

        self.N_samples.append(N)
        self.M_samples.append(M)

        self.samples += 1

    def predict(self, N) -> (np.ndarray, np.ndarray):
        return self.S.run(self.OUT, feed_dict={self.IN: N})

    def fit(self, epochs=10) -> float:

        lr = 1e-3
        self.S.run(self.surrogate_lr.assign(lr))
        adapted = True
        adaptivity = 100
        p = np.zeros(adaptivity)

        mod_2 = 0
        for i in range(epochs):
            _, err = self.S.run(
                (self.surrogate_opt, self.surrogate_error),
                feed_dict={
                    self.IN: self.N_samples,
                    self.OBS: self.M_samples
                })

            # print("I surr", i, "err", err)

            diff = np.mean(p) / err
            if diff > 0.9 and i % adaptivity == adaptivity - 1:
                if not adapted:
                    break
                lr /= 2
                self.S.run(self.surrogate_lr.assign(lr))

                adapted = False

            if diff < 0.9:
                adapted = True

            if err < self.convergence_limit:
                break
            """ Avoid oscillation. """
            if err == mod_2:
                lr /= 2
                self.S.run(self.surrogate_lr.assign(lr))

            if i % 2:
                mod_2 = err

    def calc_suggestions(self):
        """ Used to normalize """

        suggs = []
        preds = []
        ents = []
        """ Quick fix till i come up with an idea to approximate the manifold.."""
        inv_cov = np.linalg.inv(
            EmpiricalCovariance(
                assume_centered=True).fit(self.N_samples, self.M_samples)
            .covariance_ + np.eye(self.N) * 1e-6)

        # Exploitation
        for p in range(self.samples):
            gradients, inv_hess = self.S.run(
                (self.IN_grads, self.IN_hessian),
                feed_dict={
                    self.IN: [self.N_samples[p]]
                })

            suggestion = self.N_samples[p].reshape(-1) - (
                inv_hess @ gradients).reshape(-1)

            if self.minis and self.maxis:
                suggestion = np.clip(suggestion, self.minis, self.maxis)

            pred = self.predict([suggestion])

            ent = 1000000
            for point in self.N_samples:
                ent = min(
                    ent,
                    np.sqrt((point - suggestion).reshape(1, -1) @ inv_cov @ (
                        point - suggestion).reshape(-1, 1)))

            # print("Start", self.N_samples[p], "Score", self.M_samples[p])
            # print("Recommendation", suggestion, "Pred", pred, "Ent", ent)
            # print()
            # print("Score", self.M_samples[p])
            # print("Pred", pred, "Ent", ent)
            # print()

            suggs.append(suggestion)
            preds.append(pred)
            ents.append(ent)

        # Exploration..
        for p in range(10):
            suggestion = self.S.run(self.N_gen).reshape(-1)
            pred = self.predict([suggestion])

            ent = 1000000
            for point in self.N_samples:
                ent = min(
                    ent,
                    np.sqrt((point - suggestion).reshape(1, -1) @ inv_cov @ (
                        point - suggestion).reshape(-1, 1)))

            # print("Recommendation", suggestion, "Pred", pred, "Ent", ent)
            # print()

            suggs.append(suggestion)
            preds.append(pred)
            ents.append(ent)

        suggs = np.array(suggs)
        preds = np.array(preds).reshape(-1)
        ents = np.array(ents).reshape(-1)

        mean_pred = np.mean(preds)
        stdev_pred = np.std(preds)

        mean_ent = np.mean(ents)
        stdev_ent = np.std(ents)

        # print("MEANENT VARENT", np.mean(ents), np.var(ents))
        preds = (preds - mean_pred) / stdev_pred
        ents = (ents - mean_ent) / stdev_ent

        return suggs, preds, ents

    def optimize(self, exploration=0.1):
        """ Work on heuristics here. """
        suggs, preds, ents = self.calc_suggestions()
        scores = preds + exploration * ents

        # print(preds, "\n", ents, "\n", scores)
        # print()
        # print()

        return suggs[np.argmax(scores)]

    def run(self,
            random=100,
            optimization=10,
            fitting=1000,
            exploration=1.0,
            verbose=2):
        for i in range(random):
            self.sample(verbose=verbose)

        if verbose == 2 or verbose == 1:
            print("Best random", self.N_samples[np.argmax(self.M_samples)],
                  max(self.M_samples))

        for i in range(optimization):
            self.fit(fitting)
            sugg = self.optimize(exploration=exploration)
            self.sample(sugg, verbose=verbose)

        return self.N_samples[np.argmax(self.M_samples)], max(self.M_samples)

    def forget(self):
        self.M_samples = []
        self.N_samples = []
        self.k1_features = []
        self.k2_features = []
        self.kernel_labels = []
        self.samples = 0
        self.S.run(self.local_init)

    def close(self):
        self.S.close()
