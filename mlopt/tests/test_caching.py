import unittest
import cvxpy as cp
import numpy as np
import mlopt
from mlopt.sampling import uniform_sphere_sample
import pandas as pd
import numpy.testing as npt
import logging
from mlopt.tests.settings import TEST_TOL as TOL


class TestCaching(unittest.TestCase):

    def setUp(self):
        """Setup simple problem"""
        np.random.seed(1)
        # This needs to work for different
        p = 10
        n = 30
        F = np.random.randn(n, p)
        D = np.diag(np.random.rand(n)*np.sqrt(p))
        Sigma = F.dot(F.T) + D
        gamma = 1.0
        mu = cp.Parameter(n, name='mu')
        x = cp.Variable(n)
        cost = - mu * x + gamma * cp.quad_form(x, Sigma) + .5 * cp.norm(x, 1)
        constraints = [cp.sum(x) == 1, x >= 0]

        # Define optimizer
        # Force mosek to be single threaded
        m = mlopt.Optimizer(cp.Minimize(cost),
                            constraints,
                            log_level=logging.DEBUG)

        '''
        Sample points
        '''
        theta_bar = 10 * np.random.rand(n)
        radius = 5.0

        '''
        Train and solve
        '''

        # Training and testing data
        n_train = 100
        n_test = 10

        # Sample points from multivariate ball
        X_d = uniform_sphere_sample(theta_bar, radius, n=n_train)
        df = pd.DataFrame({'mu': list(X_d)})

        # Train and test using pytorch
        params = {
            'learning_rate': [0.01],
            'batch_size': [100],
            'n_epochs': [200]
        }

        m.train(df, parallel=False, learner=mlopt.PYTORCH, params=params)

        # Testing data
        X_d_test = uniform_sphere_sample(theta_bar, radius, n=n_test)
        df_test = pd.DataFrame({'mu': list(X_d_test)})

        # Store stuff
        self.m = m
        self.df_test = df_test

    def test_solve(self):
        """Solve problem with or without caching"""
        caching = self.m.solve(self.df_test, use_cache=True)
        no_caching = self.m.solve(self.df_test, use_cache=False)

        for i in range(len(self.df_test)):
            npt.assert_array_almost_equal(caching[i]['x'],
                                          no_caching[i]['x'],
                                          decimal=TOL)

            # Compare cost
            npt.assert_array_almost_equal(caching[i]['cost'],
                                          no_caching[i]['cost'],
                                          decimal=TOL)

