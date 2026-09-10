"""CLI behaviour and exit codes."""
import json

from retailml.cli import main

SMALL = ["--skus", "4", "--stores", "2", "--days", "500", "--customers", "200"]


def test_generate_reports_a_summary(capsys):
    assert main(["generate", *SMALL]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["series"] == 8
    assert payload["rows"] == 8 * 500


def test_generate_writes_files(tmp_path, capsys):
    main(["generate", *SMALL, "--out", str(tmp_path)])
    capsys.readouterr()
    assert (tmp_path / "sales.jsonl").exists()
    assert (tmp_path / "interactions.jsonl").exists()
    assert (tmp_path / "specs.json").exists()


def test_leakage_check_passes(capsys):
    assert main(["leakage-check", *SMALL, "--series", "3"]) == 0
    assert "no feature row changed" in capsys.readouterr().out


def test_backtest_ranks_models(capsys):
    code = main([
        "backtest", *SMALL, "--models", "seasonal_naive,ridge",
        "--origins", "3", "--json",
    ])
    assert code == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["model"] for r in rows} == {"seasonal_naive", "ridge"}
    assert rows[0]["wape"] <= rows[1]["wape"]


def test_backtest_feature_model_beats_the_baseline(capsys):
    main(["backtest", *SMALL, "--models", "seasonal_naive,ridge", "--origins", "3"])
    out = capsys.readouterr().out
    assert "better WAPE than seasonal naive" in out


def test_backtest_gate_fails_on_an_impossible_threshold(capsys):
    code = main([
        "backtest", *SMALL, "--models", "naive", "--origins", "2", "--max-wape", "0.001",
    ])
    capsys.readouterr()
    assert code == 1


def test_backtest_rejects_an_unknown_model(capsys):
    assert main(["backtest", *SMALL, "--models", "prophet"]) == 2
    assert "unknown model" in capsys.readouterr().err


def test_backtest_records_runs_and_registers(tmp_path, capsys):
    runs = tmp_path / "runs.jsonl"
    registry = tmp_path / "registry.json"
    main([
        "backtest", *SMALL, "--models", "seasonal_naive", "--origins", "2",
        "--tracking-path", str(runs), "--registry-path", str(registry), "--register",
    ])
    capsys.readouterr()
    assert runs.exists() and registry.exists()

    assert main(["runs", "--tracking-path", str(runs), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and "wape" in rows[0]["metrics"]

    assert main(["registry", "--registry-path", str(registry)]) == 0
    assert "demand_forecast" in capsys.readouterr().out


def test_recommend_explains_its_suggestions(capsys):
    assert main(["recommend", *SMALL, "-k", "5"]) == 0
    out = capsys.readouterr().out
    assert "because you bought" in out or "cold start" in out


def test_recommend_json(capsys):
    main(["recommend", *SMALL, "-k", "3", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["recommendations"]) <= 3
    assert payload["model"]["name"] == "item_item_cf"


def test_evaluate_reco_beats_popularity(capsys):
    code = main([
        "evaluate-reco", "--skus", "15", "--stores", "2", "--days", "400",
        "--customers", "600", "--min-ndcg-lift", "0.05",
    ])
    out = capsys.readouterr().out
    assert "popularity" in out
    assert code == 0, "the collaborative filter must earn its complexity"


def test_drift_reports_per_feature_statistics(capsys):
    assert main(["drift", *SMALL, "--reference-days", "200"]) == 0
    out = capsys.readouterr().out
    assert "PSI" in out and "VERDICT" in out


def test_drift_detects_injected_performance_decay(capsys):
    code = main([
        "drift", *SMALL, "--reference-days", "200",
        "--baseline-error", "0.20", "--recent-error", "0.45", "--fail-on-drift",
    ])
    out = capsys.readouterr().out
    assert "performance" in out
    assert code == 1


def test_retrain_holds_back_when_nothing_drifted(tmp_path, capsys):
    registry = tmp_path / "registry.json"
    # Seed a production model so the "no production model" trigger is inactive.
    from retailml import ModelRegistry

    reg = ModelRegistry(registry)
    model = reg.register("demand_forecast", "seed", "wape", 0.05)
    reg.promote("demand_forecast", model.version)

    code = main([
        "retrain", *SMALL, "--registry-path", str(registry),
        "--tracking-path", str(tmp_path / "runs.jsonl"), "--model-age-days", "1",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "no retraining needed" in out or "held back" in out


def test_retrain_runs_when_there_is_no_production_model(tmp_path, capsys):
    code = main([
        "retrain", *SMALL, "--registry-path", str(tmp_path / "registry.json"),
        "--tracking-path", str(tmp_path / "runs.jsonl"), "--model-age-days", "1",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "no_production_model" in out


def test_version(capsys):
    assert main(["--version"]) == 0
    assert "retailml" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()
