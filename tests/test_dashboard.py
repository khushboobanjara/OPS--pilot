from fastapi.testclient import TestClient

from app.main import create_app
from app.pipeline.base import Pipeline


def test_dashboard_is_served_at_root():
    with TestClient(create_app(lambda: Pipeline([]))) as c:
        r = c.get("/")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert "OpsPilot AI" in r.text and "/review-queue" in r.text
