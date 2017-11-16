import unittest
from divmachines.mf import MatrixFactorization
from divmachines.layers import TestEmbedding
from divmachines.mf.models import BiasMatrixFactorizationModel, \
    WithoutBiasMatrixFactorizationModel
from divmachines.mf import MatrixFactorization
import numpy as np


class MatrixFactorizationTest(unittest.TestCase):

    def test_predict_BiasMFModel(self):
        model = BiasMatrixFactorizationModel(3, 4, n_factors=2)
        model.x = TestEmbedding(3, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                          [1, 1],
                                          [2, 2]]))
        model.y = TestEmbedding(4, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                          [1, 1],
                                          [2, 2],
                                          [3, 3]]))
        model.user_biases = TestEmbedding(3, 1,
                                sparse=True,
                                embedding_weights=np.array([[0], [1], [2]]))
        model.item_biases = TestEmbedding(4, 1,
                                sparse=True,
                                embedding_weights=np.array([[0], [1], [2], [3]]))

        mf = MatrixFactorization(model=model, n_iter=0, learning_rate=0)
        mf.fit(np.array([[0, 0, 1],
                         [0, 1, 2],
                         [1, 2, 1],
                         [1, 3, 2],
                         [2, 2, 1],
                         [2, 3, 2]]))

        expected = np.array([0, 1, 2, 3, 1, 4, 7, 10, 2, 7, 12, 17])
        actual = mf.predict(np.array([0, 1, 2]))

        self.assertTrue(np.all(expected == actual))

    def test_predict_WOBiasMFModel(self):
        model = WithoutBiasMatrixFactorizationModel(3, 4, n_factors=2)
        model.x = TestEmbedding(3, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                          [1, 1],
                                          [2, 2]]))
        model.y = TestEmbedding(4, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                          [1, 1],
                                          [2, 2],
                                          [3, 3]]))

        mf = MatrixFactorization(model=model, n_iter=0, learning_rate=0)
        mf.fit(np.array([[0, 0, 1],
                         [0, 1, 2],
                         [1, 2, 1],
                         [1, 3, 2],
                         [2, 2, 1],
                         [2, 3, 2]]))

        expected = np.array([0, 0, 0, 0, 0, 2, 4, 6, 0, 4, 8, 12])
        actual = mf.predict(np.array([0, 1, 2]))

        self.assertTrue(np.all(expected == actual))

    def test_predict_WOBiasMFModel_one_user(self):
        model = WithoutBiasMatrixFactorizationModel(3, 4, n_factors=2)
        model.x = TestEmbedding(3, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                          [1, 1],
                                          [2, 2]]))
        model.y = TestEmbedding(4, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                          [1, 1],
                                          [2, 2],
                                          [3, 3]]))

        mf = MatrixFactorization(model=model, n_iter=0, learning_rate=0)
        mf.fit(np.array([[0, 0, 1],
                         [0, 1, 2],
                         [1, 2, 1],
                         [1, 3, 2],
                         [2, 2, 1],
                         [2, 3, 2]]))

        expected = np.array([0, 2, 4, 6])
        actual = mf.predict(1)
        self.assertTrue(np.all(expected == actual))

    def test_predict_BiasMFModel_single_user(self):
        model = BiasMatrixFactorizationModel(3, 4, n_factors=2)
        model.x = TestEmbedding(3, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                          [1, 1],
                                          [2, 2]]))
        model.y = TestEmbedding(4, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                          [1, 1],
                                          [2, 2],
                                          [3, 3]]))

        model.user_biases = TestEmbedding(3, 1,
                                sparse=True,
                                embedding_weights=np.array([[0], [1], [2]]))
        model.item_biases = TestEmbedding(4, 1,
                                sparse=True,
                                embedding_weights=np.array([[0], [1], [2], [3]]))

        mf = MatrixFactorization(model=model, n_iter=0, learning_rate=0)
        mf.fit(np.array([[0, 0, 1],
                         [0, 1, 2],
                         [1, 2, 1],
                         [1, 3, 2],
                         [2, 2, 1],
                         [2, 3, 2]]))

        expected = np.array([1, 4, 7, 10])
        actual = mf.predict(1)

        self.assertTrue(np.all(expected == actual))

    def test_predict_BiasMFModel_single_user_single_item(self):
        model = BiasMatrixFactorizationModel(3, 4, n_factors=2)
        model.x = TestEmbedding(3, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                                            [1, 1],
                                                            [2, 2]]))
        model.y = TestEmbedding(4, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                                            [1, 1],
                                                            [2, 2],
                                                            [3, 3]]))

        model.user_biases = TestEmbedding(3, 1,
                                          sparse=True,
                                          embedding_weights=np.array([[0], [1], [2]]))
        model.item_biases = TestEmbedding(4, 1,
                                          sparse=True,
                                          embedding_weights=np.array([[0], [1], [2], [3]]))

        mf = MatrixFactorization(model=model, n_iter=0, learning_rate=0)
        mf.fit(np.array([[0, 0, 1],
                         [0, 1, 2],
                         [1, 2, 1],
                         [1, 3, 2],
                         [2, 2, 1],
                         [2, 3, 2]]))

        expected = np.array([4])
        actual = mf.predict(1, 1)

        self.assertTrue(np.all(expected == actual))

    def test_predict_WOBiasMFModel_multi_users_multi_items(self):
        model = WithoutBiasMatrixFactorizationModel(3, 4, n_factors=2)
        model.x = TestEmbedding(3, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                                            [1, 1],
                                                            [2, 2]]))
        model.y = TestEmbedding(4, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                                            [1, 1],
                                                            [2, 2],
                                                            [3, 3]]))


        mf = MatrixFactorization(model=model, n_iter=0, learning_rate=0)
        mf.fit(np.array([[0, 0, 1],
                         [0, 1, 2],
                         [1, 2, 1],
                         [1, 3, 2],
                         [2, 2, 1],
                         [2, 3, 2]]))

        expected = np.array([0, 2])
        actual = mf.predict(np.array([0, 1]), np.array([0, 1]))

        self.assertTrue(np.all(expected == actual))

    def test_predict_BiasMFModel_multi_users_multi_items(self):
        model = BiasMatrixFactorizationModel(3, 4, n_factors=2)
        model.x = TestEmbedding(3, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                                            [1, 1],
                                                            [2, 2]]))
        model.y = TestEmbedding(4, 2,
                                sparse=True,
                                embedding_weights=np.array([[0, 0],
                                                            [1, 1],
                                                            [2, 2],
                                                            [3, 3]]))
        model.user_biases = TestEmbedding(3, 1,
                                          sparse=True,
                                          embedding_weights=np.array([[0], [1], [2]]))
        model.item_biases = TestEmbedding(4, 1,
                                          sparse=True,
                                          embedding_weights=np.array([[0], [1], [2], [3]]))

        mf = MatrixFactorization(model=model, n_iter=0, learning_rate=0)
        mf.fit(np.array([[0, 0, 1],
                         [0, 1, 2],
                         [1, 2, 1],
                         [1, 3, 2],
                         [2, 2, 1],
                         [2, 3, 2]]))

        expected = np.array([0, 4])
        actual = mf.predict(np.array([0, 1]), np.array([0, 1]))

        self.assertTrue(np.all(expected == actual))
