import re
from typing import Any, Union, Dict

from .. import AudioTransformer


class AudioEvaluator:
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self):
        self.greater_is_better = True
        self.primary_metric = None

    def __call__(
        self, model: AudioTransformer, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> Union[float, Dict[str, float]]:
        pass

    def prefix_name_to_metrics(self, metrics: Dict[str, float], name: str) -> Dict[str, float]:
        if not name:
            return metrics
        metrics = {name + "_" + key: float(value) for key, value in metrics.items()}
        if hasattr(self, "primary_metric") and not self.primary_metric.startswith(name + "_"):
            self.primary_metric = name + "_" + self.primary_metric
        return metrics

    def store_metrics_in_model_card_data(self, model: AudioTransformer, metrics: Dict[str, Any]) -> None:
        # model.model_card_data.set_evaluation_metrics(self, metrics)
        pass

    @property
    def description(self) -> str:
        class_name = self.__class__.__name__
        
        try:
            index = class_name.index("Evaluator")
            class_name = class_name[:index]
        except IndexError:
            pass

        return re.sub(r"([a-z])([A-Z])", r"\g<1> \g<2>", class_name)