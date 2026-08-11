"""Model exports with lazy loading for optional Torch-backed policies."""

from .traffic_predictor import RidgeMultiOutput, TrafficPredictorSuite

__all__ = [
    "ECDUGNNLayer",
    "MomentEncoder",
    "RidgeMultiOutput",
    "SlotLevelPolicy",
    "TrafficPredictorSuite",
]


def __getattr__(name):
    if name == "ECDUGNNLayer":
        from .gnn_layers import ECDUGNNLayer

        return ECDUGNNLayer
    if name == "MomentEncoder":
        from .moment_encoder import MomentEncoder

        return MomentEncoder
    if name == "SlotLevelPolicy":
        from .slot_policy import SlotLevelPolicy

        return SlotLevelPolicy
    raise AttributeError(name)
