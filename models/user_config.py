from enum import Enum

from models.baseline_qwen import BaselineQwenModel


class Models(Enum):
    # Add your models here
    baseline_qwen_3_5_9b = BaselineQwenModel

    @staticmethod
    def get_model(model_name: str):
        try:
            return Models[model_name].value
        except KeyError:
            raise ValueError(f"Model `{model_name}` not found.")
