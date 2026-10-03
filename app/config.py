from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///opspilot.db"
    classifier_model_path: str = "models/email_classifier.joblib"
    classifier_min_confidence: float = 0.70
    anomaly_model_path: str = "models/anomaly_detector.joblib"
    # Decision thresholds (tuned in Phase 6)
    auto_approve_max_risk: float = 0.30
    auto_reject_min_risk: float = 0.85
    auto_approve_max_amount: float = 10000.0

    model_config = {"env_prefix": "OPSPILOT_"}


settings = Settings()
