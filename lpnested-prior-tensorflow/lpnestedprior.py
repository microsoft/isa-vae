# Copyright 2019 Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
import scipy.special

class LpNestedPrior:
    def __init__(self, p=[0.5, [[1.0], [2.0, [ [1.0], [1.0]]]]], scale=1.0):
        self.p = p
        self.scale = scale
        self.n = self.count_leaves(self.p)
        self.Sf1_log = self.Sf_log(self.p)

    def dimz(self):
        return self.n

    def pradial_log_prob(self, v0, p0, n, s=1.0):
        u = n / p0
        a = tf.convert_to_tensor(tf.log(p0)) + tf.log(v0)*(n-1)
        b = scipy.special.gammaln(u) + tf.log(s)*(u)
        c = (- (v0**p0) / s)
        val = a - tf.convert_to_tensor(b) + c
        return val

    def count_leaves(self, p):
        if len(p)<=1:
            return 1
        else:
            val = 0
            for p_Ik in p[1]:
                val = val + self.count_leaves(p_Ik)
            return val

    def betaFunction(self, a, b):
        return scipy.special.gamma(a) * scipy.special.gamma(b) / scipy.special.gamma(a+b)

    def betaFunctionLog(self, a, b):
        return tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a+b)

    def Sf_recursive(self, p):
        if len(p) <= 1:
            # leaf node
            return 1
        else:
            p_I = p[0]
            n_I = self.count_leaves(p)
            p_children = p[1]
            l_I = len(p_children)
            val = 1.0 / p_I**(l_I-1)
            n_Ik_sum = 0
            for k in range(len(p_children)-1):
                n_Ik = self.count_leaves(p_children[k])
                n_Ik_sum = n_Ik_sum + n_Ik
                n_Ik_plus = self.count_leaves(p_children[k+1])
                val = val * self.betaFunction(n_Ik_sum / p_I, n_Ik_plus / p_I)
            for p_child in p_children:
                val = val * self.Sf_recursive(p_child)
            return val


    # compute the surface of a unit ball
    def Sf(self, p):
        n = self.count_leaves(p)
        return 2**n * self.Sf_recursive(p)

    def Sf_log_recursive(self, p):
        if len(p) <= 1:
            # leaf node
            return 0
        else:
            p_I = p[0]
            n_I = self.count_leaves(p)
            p_children = p[1]
            l_I = len(p_children)
            val = - tf.log(p_I)*(l_I-1)
            n_Ik_sum = 0
            for k in range(len(p_children)-1):
                n_Ik = self.count_leaves(p_children[k])
                n_Ik_sum = n_Ik_sum + n_Ik
                n_Ik_plus = self.count_leaves(p_children[k+1])
                val = val + self.betaFunctionLog(n_Ik_sum / p_I, n_Ik_plus / p_I)
            for p_child in p_children:
                val = val + self.Sf_log_recursive(p_child)
            return val

    # compute the logarithm of the surface of a unit ball
    def Sf_log(self, p):
        n = self.count_leaves(p)
        return n * tf.log(2.0) + self.Sf_log_recursive(p)

    def f_recursive(self, z, p, k):
        if len(p) <= 1:
            # leaf node
            return tf.abs(z[:,k]), k+1, 1.0
        else:
            p_I = p[0]
            p_children = p[1]
            val = 0
            for p_child in p_children:
                op, k, p_Iplus = self.f_recursive(z, p_child, k)
                val = val + op**(p_I/p_Iplus)
            return val, k, p_I

    def f(self, z, p):
        val, k, p_I = self.f_recursive(z, p, k=0)
        return val**(1.0/p_I)

    def log_prob(self, z):
        v0 = self.f(z, self.p)
        p0 = self.p[0]

        res1 = self.pradial_log_prob(v0, p0, self.n, self.scale)

        log_divisor = tf.log(v0)*(self.n-1) + tf.convert_to_tensor(self.Sf1_log)
        res = res1 - log_divisor
        return res
