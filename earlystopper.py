from utils import Monitor
import logging

log = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(self, patience, monitor):

        self.monitor = monitor
        self.patience = patience

        self.previous_metrics = None
        self.patience_left = patience

    def should_stop(self, curr_metrics):
        should_stop = False
        value, name = self.monitor.value, self.monitor.name

        if self.previous_metrics is not None:
            if self.monitor == Monitor.MEAN_RANK_REL or self.monitor == Monitor.FILTERED_MEAN_RANK_REL:
                is_worse = self.previous_metrics[value] < curr_metrics[value]
            else:
                is_worse = self.previous_metrics[value] > curr_metrics[value]

            if self.patience_left > 0 and is_worse:
                self.patience_left -= 1
                log.info(
                    '%s more chances before the trainer stops the training. (prev_%s, curr_%s): (%.4f, %.4f)' %
                    (self.patience_left, name, name, self.previous_metrics[value], curr_metrics[value]))

            elif self.patience_left == 0 and is_worse:
                log.info('Stop the training.')
                should_stop = True

            else:
                log.info('Reset the patience count to %d' % (self.patience))
                self.patience_left = self.patience

        self.previous_metrics = curr_metrics

        return should_stop
