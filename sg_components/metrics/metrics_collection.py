"""
This module provides functionalities to set up evaluation metrics used in a neural network evaluation process.

Functions:
    get_metrics: Generates a collection of essential evaluation metrics.
"""

from torchmetrics import MetricCollection

from sg_components.metrics.highlights.metrics import (
    AveragePrecision,
    HIT1Coef,
)
from sg_components.metrics.moments.metrics import MRAveragePrecision, MRRecallAt1


def get_metrics(**kwargs) -> MetricCollection:
    """Generate a collection of essential evaluation metrics.

    This function creates a collection of metrics HL/MR tasks.

    Args:
        kwargs: Arbitrary keyword arguments that are forwarded to the initialization of each metric.

    Returns:
        MetricCollection: A collection of initialized metrics.
    """
    return MetricCollection(
        {
            "HL-HIT@1-Fair": HIT1Coef(threshold="Fair", **kwargs),
            "HL-HIT@1-Good": HIT1Coef(threshold="Good", **kwargs),
            "HL-HIT@1-VeryGood": HIT1Coef(threshold="VeryGood", **kwargs),
            "HL-mAP-Fair": AveragePrecision(threshold="Fair", **kwargs),
            "HL-mAP-Good": AveragePrecision(threshold="Good", **kwargs),
            "HL-mAP-VeryGood": AveragePrecision(threshold="VeryGood", **kwargs),
            "MR-mAP-Short": MRAveragePrecision(window_range="short", **kwargs),
            "MR-mAP-Middle": MRAveragePrecision(window_range="middle", **kwargs),
            "MR-mAP-Long": MRAveragePrecision(window_range="long", **kwargs),
            "MR-mAP-Full": MRAveragePrecision(window_range="full", **kwargs),
            "MR-R1-Short": MRRecallAt1(window_range="short", **kwargs),
            "MR-R1-Middle": MRRecallAt1(window_range="middle", **kwargs),
            "MR-R1-Long": MRRecallAt1(window_range="long", **kwargs),
            "MR-R1-Full": MRRecallAt1(window_range="full", **kwargs),
        },
    )
