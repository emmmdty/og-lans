import importlib


def test_json_parser_runs_without_dirtyjson(monkeypatch):
    module = importlib.import_module("oglans.utils.json_parser")
    monkeypatch.setattr(module, "_DIRTYJSON_MODULE", None)

    parser = module.RobustJSONParser(enable_dirtyjson=True)
    assert parser._dirtyjson is None

    parsed, diagnostics = parser.parse('```json\n[{"event_type":"X","arguments":[]}]\n```')
    assert diagnostics["success"] is True
    assert isinstance(parsed, list)
