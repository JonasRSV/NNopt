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
            memory_hidden_layer: int = 60,
            Rn:
            "[Range_0, Range_1...Range_n] N = Range_n = (Min_n, Max_n)" = None,
            seed=None):

        self.N = N
        self.Rn = Rn

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

            with tf.variable_scope("memory", reuse=tf.AUTO_REUSE):
                hidden = tf.layers.dense(
                    self.IN,
                    memory_hidden_layer,
                    activation=tf.math.sigmoid,
                    use_bias=True)
                memory_out = tf.squeeze(
                    tf.layers.dense(hidden, 1, activation=None, use_bias=True))

            memory_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/memory".format(Optimizer.SCOPE))

            self.mem_out = memory_out
            self.entropy = tf.losses.mean_squared_error(self.OUT, memory_out)
            error = tf.losses.mean_squared_error(self.OBS, memory_out)
            memory_grads = tf.gradients(error, memory_variables)
            self.memory_opt = tf.train.AdamOptimizer(
                learning_rate=0.001).apply_gradients(
                    zip(memory_grads, memory_variables))

            self.IN_grads = tf.squeeze(tf.gradients(self.OUT, self.IN), [0, 1])
            self.IN_hessian = tf.linalg.inv(
                tf.squeeze(tf.hessians(self.OUT, self.IN), axis=[0, 1, 3]))

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

    def predict(self, N) -> (np.ndarray, np.ndarray):
        return self.S.run((self.OUT, self.entropy), feed_dict={self.IN: N})

    def mem_pred(self, N) -> (np.ndarray, np.ndarray):
        return self.S.run((self.mem_out), feed_dict={self.IN: N})

    def fit(self, epochs=10, forgetting=True) -> float:
        if forgetting:
            self.S.run(self.local_init)
        for _ in range(epochs):
            self.S.run(
                (self.surrogate_opt, self.memory_opt),
                feed_dict={
                    self.IN: self.N_samples,
                    self.OBS: self.M_samples
                })

    def optimize(self, exploration=0.1):
        """ Work on heuristics here. """

        POINTS = len(self.N_samples)
        suggestions = []
        score_suggestions = []
        entr_suggestions = []
        """ All of these belong to potential regions. """
        for p in range(POINTS):

            # newton step is verrry unstable here it seems
            gradients, inv_hess = self.S.run(
                (self.IN_grads, self.IN_hessian),
                feed_dict={
                    self.IN: [self.N_samples[p]]
                })

            suggestion = (self.N_samples[p]).reshape(-1)
            step = (inv_hess @ gradients).reshape(-1)

            # print("Suggestion", suggestion)
            # print("S", step)

            if self.Rn:
                sugg = []
                for s, (mn, mx) in zip(suggestion - step, self.Rn):
                    sugg.append(np.clip(s, mn, mx))

                suggestion = np.array(sugg)

            # suggestion = self.N_samples[p]

            # A few gradient steps is more stable
            # for _ in range(10):
            # gradients = self.S.run(
            # (self.IN_grads), feed_dict={
            # self.IN: [suggestion]
            # })

            # # print("Suggestion clipped", suggestion)
            # suggestion = suggestion + gradients

            pred, entropy = self.predict([suggestion])
            # print("Pred", pred)
            # print("Ent", entropy)

            # print()
            # print()

            suggestions.append(suggestion)
            score_suggestions.append((p, pred))
            entr_suggestions.append((p, entropy))

        score_suggestions.sort(key=lambda x: x[1], reverse=True)
        entr_suggestions.sort(key=lambda x: x[1], reverse=True)

        ranking = [0] * POINTS
        for r, (p, _) in enumerate(score_suggestions):
            ranking[p] += (POINTS - r)

        for r, (p, _) in enumerate(entr_suggestions):
            ranking[p] += (POINTS - r) * exploration

        return suggestions[np.argmax(ranking)]

    def run(self,
            random=100,
            optimization=10,
            fitting=1000,
            exploration=1.0,
            forgetting=False,
            verbose=True):
        for i in range(random):
            self.sample(verbose=verbose)

        if verbose:
            print("Best random", self.N_samples[np.argmax(self.M_samples)],
                  max(self.M_samples))

        self.fit(fitting)

        for i in range(optimization):
            sugg = self.optimize(exploration=exploration)
            self.sample(sugg, verbose=verbose)
            self.fit(fitting, forgetting=forgetting)

        return self.N_samples[np.argmax(self.M_samples)], max(self.M_samples)

    def forget(self):
        self.M_samples = []
        self.N_samples = []
        self.S.run(self.local_init)

    def close(self):
        self.S.close()
