import unittest
import numpy as np
import torch
from torch import LongTensor
from torch.autograd import Variable
from divmachines.mf.mf import MatrixFactorization


class MatrixFactorizationTest(unittest.TestCase):

    def test_forward_size_batch(self):
        model = MatrixFactorization(5, 5, n_factors=5)

        user_ids = Variable(LongTensor(np.array([0, 1, 2, 3])))
        item_ids = Variable(LongTensor(np.array([0, 1, 2, 3])))

        pred = model(user_ids, item_ids)

        expected = (model.x(user_ids) * model.y(item_ids)).sum(1)

        self.assertEqual(pred.size(), torch.Size([4]))
        self.assertTrue(np.all(pred.data.numpy() == expected.data.numpy()))

    def test_forward_size_single_pair(self):
        model = MatrixFactorization(5, 5, n_factors=5)

        user_ids = Variable(LongTensor(np.array([0])))
        item_ids = Variable(LongTensor(np.array([0])))

        pred = model(user_ids, item_ids)
        expected = (model.x(user_ids) * model.y(item_ids)).sum()

        self.assertEqual(pred.size(), torch.Size([1]))
        self.assertTrue(np.all(pred.data.numpy() == expected.data.numpy()))


