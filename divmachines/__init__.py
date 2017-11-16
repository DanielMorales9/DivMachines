class Classifier(object):
    """
    Base class for all classifiers
    """
    def __init__(self):
        super(Classifier, self).__init__()

    def fit(self, *args):
        pass

    def prediction(self, *args):
        pass
