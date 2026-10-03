from app.pipeline.anomaly import AnomalyStage
from app.pipeline.base import Pipeline
from app.pipeline.classify import ClassifyStage
from app.pipeline.decision import DecisionStage
from app.pipeline.duplicates import DuplicateStage
from app.pipeline.extract import ExtractStage
from app.pipeline.risk import RiskStage
from app.pipeline.validate import ValidateStage


def build_pipeline() -> Pipeline:
    """The production pipeline, in architecture order."""
    return Pipeline([ClassifyStage(), ExtractStage(), ValidateStage(), DuplicateStage(),
                     AnomalyStage(), RiskStage(), DecisionStage()])
