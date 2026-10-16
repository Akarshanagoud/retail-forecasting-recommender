"""The studio's HTTP API, on a small seeded dataset so the suite stays fast."""
import pytest
from fastapi.testclient import TestClient

from retailml.server import Studio, create_app


@pytest.fixture(scope="module")
def studio(tmp_path_factory):
    return Studio(n_skus=3, n_stores=1, days=320, reco_customers=300, workspace=tmp_path_factory.mktemp("studio"))


@pytest.fixture()
def client(studio):
    return TestClient(create_app(studio))


def test_ui_is_served_and_root_redirects(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"] == "/ui/"
    assert "Demand Planning Studio" in client.get("/ui/").text


def test_endpoints_wait_for_the_backtests(client, studio):
    studio.ready.clear()
    try:
        response = client.get("/api/models")
        assert response.status_code == 503
        assert response.headers["retry-after"] == "2"
        assert client.get("/health").json()["status"] == "warming"
    finally:
        studio.ready.set()
    assert client.get("/api/models").status_code == 200


def test_leaderboard_is_ranked_and_complete(client, studio):
    data = client.get("/api/models").json()
    wapes = [c["wape"] for c in data["comparisons"] if c["wape"] is not None]
    assert wapes == sorted(wapes)
    assert len(data["matrix"]["series"]) == len(studio.series)
    assert all(len(row) == len(data["matrix"]["models"]) for row in data["matrix"]["wape"])


def test_series_view_lines_up_history_backtests_and_forecast(client, studio):
    sku, store = sorted(studio.series)[0]
    view = client.get(f"/api/series?sku={sku}&store={store}&model=ridge").json()
    assert len(view["dates"]) == len(view["actual"]) + len(view["forecast"]) == len(view["actual"]) + 28
    for window in view["windows"]:
        assert window["start_index"] >= 0
        assert window["start_index"] + len(window["predicted"]) <= len(view["actual"])
    assert all(lo <= f <= hi for lo, f, hi in zip(view["lower"], view["forecast"], view["upper"], strict=True))
    assert min(view["forecast"]) > 0, "a 28-day forecast must not collapse once lags reach the horizon"


def test_recommendations_explain_themselves_from_the_customers_history(client):
    customer = client.get("/api/customers").json()[0]["customer"]
    data = client.get(f"/api/recommend?customer={customer}").json()
    bought = {h["sku"] for h in data["history"]}
    assert data["cf"]
    for rec in data["cf"]:
        assert rec["sku"] not in bought
        assert set(rec["because_of"]) <= bought
    assert set(data["scores"]) == {"item-item CF", "popularity"}


def test_drift_report_and_retraining_decision(client, studio):
    sku, store = sorted(studio.series)[0]
    data = client.get(f"/api/drift?sku={sku}&store={store}&model_age_days=90").json()
    assert data["report"]["verdict"] in {"stable", "watch", "drifted"}
    assert data["decision"]["should_retrain"] is True
    assert "schedule" in data["decision"]["reasons"]


def test_registry_promotes_only_material_improvements(client):
    registry = client.get("/api/registry").json()
    assert registry["production"]["params"]["model"] == "seasonal_naive"
    ridge = client.post("/api/registry/train", json={"model": "ridge"}).json()["registered"]
    assert ridge["stage"] == "staging"
    outcome = client.post("/api/registry/promote", json={"version": ridge["version"]}).json()
    assert outcome["promoted"] is True
    assert outcome["registry"]["production"]["version"] == ridge["version"]
    again = client.post("/api/registry/train", json={"model": "ridge"}).json()["registered"]
    assert client.post("/api/registry/promote", json={"version": again["version"]}).json()["promoted"] is False


def test_unknown_models_and_series_are_rejected(client):
    assert client.get("/api/series?sku=SKU-000&store=ST-00&model=prophet").status_code == 422
    assert client.get("/api/series?sku=NOPE&store=ST-00&model=ridge").status_code == 404
