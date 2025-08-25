"""
Early-stopping utility.

Monitors a chosen validation metric and stops training
when the metric has not improved for a given number of epochs (`patience`).
"""

from utils import Monitor
import logging

log = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(self, patience, monitor):
        """
        Args
        ----
        patience : int
            Number of consecutive epochs without improvement
            before training should be stopped.
        monitor : utils.Monitor
            Metric to watch (e.g., Monitor.MEAN_RANK_REL,
            Monitor.FILTERED_MEAN_RANK_REL, etc.).
        """
        self.monitor = monitor
        self.patience = patience

        # Keep the last metric snapshot
        self.previous_metrics = None
        # How many chances are left before stopping
        self.patience_left = patience

    def should_stop(self, curr_metrics):
        """
        Decide whether training should stop based on the latest metric values.

        Args
        ----
        curr_metrics : dict
            Mapping from metric keys to their current numeric values.

        Returns
        -------
        bool
            True  -> stop training.
            False -> keep training.
        """
        should_stop = False
        value_key = self.monitor.value  # metric key to check
        metric_name = self.monitor.name  # human-readable metric name

        # First call: just store the metrics
        if self.previous_metrics is not None:
            # Determine whether the metric got worse
            if self.monitor in (Monitor.MEAN_RANK_REL, Monitor.FILTERED_MEAN_RANK_REL):
                # For rank-based metrics, lower is better
                is_worse = self.previous_metrics[value_key] < curr_metrics[value_key]
            else:
                # For accuracy-like metrics, higher is better
                is_worse = self.previous_metrics[value_key] > curr_metrics[value_key]

            if self.patience_left > 0 and is_worse:
                # Metric degraded but we still have patience left
                self.patience_left -= 1
                log.info(
                    '%d more chances before stopping. (%s) prev=%.4f, curr=%.4f',
                    self.patience_left, metric_name,
                    self.previous_metrics[value_key], curr_metrics[value_key]
                )

            elif self.patience_left == 0 and is_worse:
                # Patience exhausted
                log.info('Stopping training.')
                should_stop = True

            else:
                # Metric improved
                log.info('Patience reset to %d.', self.patience)
                self.patience_left = self.patience

        # Save snapshot for the next call
        self.previous_metrics = curr_metrics

        return should_stop
