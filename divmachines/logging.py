
class Logger(object):
    """
    Base class for logging.
    """
    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        pass


class TrainingLogger(Logger):
    """
    Training Logger a class that logs the training process.
    It can be configured for storing the losses for each epoch
    and for each batch.

    Parameters
    ----------
    batch: bool, optional
        Flag for logging batch or not
    """
    def __init__(self, batch=False):
        super(TrainingLogger, self).__init__()
        self._batch = batch
        self._logs = []

        self._losses = None
        self._epochs = None
        self._batches = None

    @property
    def losses(self):
        """
        Getter for the losses
        :return: list
            losses
        """
        if self._losses is None:
            if self._batch:
                self._losses = [a for a, _, _ in self._logs]
            else:
                self._losses = [a for a, _ in self._logs]
        return self._losses

    @property
    def epochs(self):
        """
        Getter for the epochs
        :return: list
            epochs
        """
        if self._epochs is None:
            if self._batch:
                self._epochs = [b for _, b, _ in self._logs]
            else:
                self._epochs = [b for _, b in self._logs]
        return self._epochs

    @property
    def batches(self):
        """
        Getter for the Batches
        If batch logging is not enable raise ValueError
        :return: list
            batches
        """
        if not self._batch:
            raise ValueError("Batch logging is disabled")
        if self._batch is None:
            self._batches = [c for _, _, c in self._logs]
        return self._batches

    def log(self, loss, epoch, batch=None):
        """
        Logging function
        :param loss: float
            Loss value for an epoch and/or batch
        :param epoch: int
            Iteration
        :param batch: int, optional
            Batch in the Iteration
        """

        loss = loss.data.numpy()[0]
        if self._batch:
            if batch is None:
                raise ValueError("Batch logging enabled without "
                                 "providing batch value")
            else:
                self._logs.append((loss, epoch, batch))
        else:
            self._logs.append((loss, epoch))
