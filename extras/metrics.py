import re

import torch
from ignite.metrics import Metric

class Inform(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super(Inform, self).reset()

    def update(self, output):
        y_preds, ys = output
        predicted_slots = list()
        for ypred in y_preds:
            predicted_slots = re.findall(r"\[.+\]", ' '.join(ypred))

        # TODO: How we can calcualte if we have multiple refrences.
        expected_slots = list()
        for y in ys:
            expected_slots = re.findall(r"\[.+\]", ' '.join(y[0]))

        self._num_correct += len(set(expected_slots) & set(predicted_slots))
        self._num_examples += len(set(predicted_slots))

    def compute(self):
        return self._num_correct / self._num_examples


class Success(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        super(Success, self).reset()

    def update(self, output):
        pass

    def compute(self):
        return 0


class Combined(Metric):
    def __init__(self, metrics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = metrics

    def reset(self):
        super(Combined, self).reset()

    def update(self, output):
        pass

    def compute(self):
        bleu = self._metrics['bleu'].compute()
        inform = self._metrics['inform'].compute()
        success = self._metrics['success'].compute()
        return bleu + 0.5 * (inform + success)
