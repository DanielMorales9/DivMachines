from torch.nn import Module


class PointwiseModel(Module):
    """
    Base Pointwise Model class
    """
    def __init__(self):
        super(PointwiseModel, self).__init__()

    def forward(self, *input):
        pass


class PairwiseModel(Module):
    """
    Base Pairwise Model class
    """

    def forward(self, *input):
        pass


class Classifier(object):
    """
    Base Classifier
    """
    def __init__(self):
        self._n_users = None
        self._n_items = None

    def _init_model(self):
        pass

    def _init_optimization_function(self):
        pass

    def fit(self, *args):
        pass

    def prediction(self, *args):
        pass

    def _check_input(self, user_ids, item_ids, allow_items_none=False):

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._n_users:
            raise ValueError('Maximum user id greater '
                             'than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._n_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')
