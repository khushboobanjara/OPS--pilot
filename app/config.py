from pydantic import SecretStr
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
    new_vendor_review_amount: float = 2000.0

    # Hardening / public demo
    api_key: SecretStr = SecretStr("")      # when set, every API call must send header X-API-Key
    max_request_bytes: int = 1_000_000      # larger request bodies get 413
    demo_mode: bool = False                 # public demo: seed sample data, rate-limit writes, reset on a schedule
    demo_writes_per_minute: int = 60
    demo_reset_minutes: int = 60
    # Mailbox integration (see .env.example). Credentials come from the environment or a git-ignored .env file.
    imap_host: str = ""
    imap_port: int = 993
    imap_user: str = ""
    imap_password: SecretStr = SecretStr("")
    imap_folder: str = "INBOX"
    api_url: str = "http://127.0.0.1:8000"
    poll_interval_seconds: int = 60
    smtp_host: str = "smtp.gmail.com"      # only used by scripts/send_test_emails.py
    smtp_port: int = 465

    model_config = {"env_prefix": "OPSPILOT_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
