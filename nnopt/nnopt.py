import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import random


class Optimizer():
    SCOPE = "nnopt"

    def __init__(
            self,
            F: "F: R^N -> R^1",
            N: int,
            surrogate_hidden_layer: int = 60,
            kernel_seperate: int = 60,
            kernel_common: int = 60,
            Rn:
            "[Range_0, Range_1...Range_n] N = Range_n = (Min_n, Max_n)" = None,
            convergence_limit: float = 1e-5,
            codomain_norm:
            "F: R^1 x R^1 -> R^1" = lambda x, y: np.sum(np.abs(x - y)),
            seed=None):

        self.N = N
        self.Rn = Rn

        self.convergence_limit = convergence_limit
        self.codomain_norm = codomain_norm

        self.minis = None
        self.maxis = None
        if Rn and len(Rn) != N:
            raise Exception("Ranges must match domain")
        if Rn:
            self.minis, self.maxis = zip(*Rn)

        self.F = F

        self.N_samples = []
        self.M_samples = []
        self.samples = 0

        self.k1_features = []
        self.k2_features = []
        self.kernel_labels = []

        self.kernel_lr = tf.Variable(1e-3)
        self.surrogate_lr = tf.Variable(1e-3)

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
                scope="{}/surrogate".format(Optimizer.SCOPE))

            self.surrogate_error = tf.losses.mean_squared_error(
                self.OBS, self.OUT)

            surrogate_grads = tf.gradients(self.surrogate_error,
                                           surrogate_variables)
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

            with tf.variable_scope("kernel", reuse=tf.AUTO_REUSE):
                hidden = tf.concat([self.K1, self.K2], 1)
                hidden = tf.layers.dense(
                    hidden,
                    kernel_common,
                    activation=tf.math.sigmoid,
                    use_bias=True)
                hidden = tf.layers.dense(
                    hidden,
                    kernel_common,
                    activation=tf.math.sigmoid,
                    use_bias=True)
                hidden = tf.layers.dense(
                    hidden, kernel_common, activation=None, use_bias=True)
                self.K = tf.squeeze(
                    tf.layers.dense(hidden, 1, activation=None, use_bias=True))

            kernel_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/kernel".format(Optimizer.SCOPE))

            self.kernel_error = tf.losses.mean_squared_error(
                self.K_labels, self.K)
            kernel_grads = tf.gradients(self.kernel_error, kernel_variables)
            self.kernel_opt = tf.train.AdamOptimizer(
                learning_rate=self.kernel_lr).apply_gradients(
                    zip(kernel_grads, kernel_variables))

        self.local_init = tf.initializers.variables(
            tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=Optimizer.SCOPE))

        self.S = tf.Session()
        self.S.run(self.local_init)

    def sample(self, N=[], verbose=True) -> None:
        if len(N) == 0:
            N = self.S.run(self.N_gen)

        M = self.F(*N)
        if verbose:
            print("target", M)

        self.N_samples.append(N)
        self.M_samples.append(M)

        self.samples += 1

        for i in range(self.samples):
            self.k1_features.append(self.N_samples[-1])
            self.k2_features.append(self.N_samples[i])
            self.kernel_labels.append(
                self.codomain_norm(self.M_samples[-1], self.M_samples[i]))

            self.k1_features.append(self.N_samples[i])
            self.k2_features.append(self.N_samples[-1])
            self.kernel_labels.append(
                self.codomain_norm(self.M_samples[i], self.M_samples[-1]))

    def predict(self, N) -> (np.ndarray, np.ndarray):
        return self.S.run(self.OUT, feed_dict={self.IN: N})

    def kernel(self, k1, k2) -> [float]:
        return self.S.run(self.K, feed_dict={self.K1: k1, self.K2: k2})

    def fit(self, epochs=10, lr=1e-3) -> float:
        self.S.run((self.kernel_lr.assign(lr), self.surrogate_lr.assign(lr)))

        for i in range(epochs):
            _, err = self.S.run(
                (self.surrogate_opt, self.surrogate_error),
                feed_dict={
                    self.IN: self.N_samples,
                    self.OBS: self.M_samples
                })

            if err < self.convergence_limit:
                break

        for i in range(min(max(epochs // 5, 400), 500)):
            _, err = self.S.run(
                (self.kernel_opt, self.kernel_error),
                feed_dict={
                    self.K1: self.k1_features,
                    self.K2: self.k2_features,
                    self.K_labels: self.kernel_labels
                })

            if err < self.convergence_limit:
                break

    def calc_suggestions(self):
        """ Used to normalize """

        suggs = []
        preds = []
        ents = []

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

            ent = 1e6
            for k2 in self.N_samples:
                ent = min(self.kernel([suggestion], [k2]), ent)

            # print("Start", self.N_samples[p], "Score", self.M_samples[p])
            # print("Recommendation", suggestion, "Pred", pred, "Ent", ent)
            # print()
            # print("Score", self.M_samples[p])
            # print("Pred", pred, "Ent", ent)
            # print()

            suggs.append(suggestion)
            preds.append(pred)
            ents.append(ent)

        preds = np.array(preds)
        ents = np.array(ents)

        mean_pred = np.mean(preds)
        stdev_pred = np.std(preds)
        mean_ent = np.mean(ents)
        stdev_ent = np.std(ents)

        # print("MEANENT VARENT", np.mean(ents), np.var(ents))
        preds = (preds - mean_pred) / (stdev_pred + 1e-6)
        ents = (ents - mean_ent) / (stdev_ent + 1e-6)

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
            lr=1e-3,
            verbose=True):
        for i in range(random):
            self.sample(verbose=verbose)

        if verbose:
            print("Best random", self.N_samples[np.argmax(self.M_samples)],
                  max(self.M_samples))

        self.fit(fitting, lr=lr)

        for i in range(optimization):
            sugg = self.optimize(exploration=exploration)
            self.sample(sugg, verbose=verbose)
            self.fit(fitting, lr=lr)

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
