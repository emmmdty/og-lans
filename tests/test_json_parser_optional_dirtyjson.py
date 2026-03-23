import importlib


def test_json_parser_runs_without_dirtyjson(monkeypatch):
    module = importlib.import_module("oglans.utils.json_parser")
    monkeypatch.setattr(module, "_DIRTYJSON_MODULE", None)

    parser = module.RobustJSONParser(enable_dirtyjson=True)
    assert parser._dirtyjson is None

    parsed, diagnostics = parser.parse('```json\n[{"event_type":"X","arguments":[]}]\n```')
    assert diagnostics["success"] is True
    assert isinstance(parsed, list)


def test_normalize_parsed_events_supports_wrapped_events():
    module = importlib.import_module("oglans.utils.json_parser")

    parsed = module.normalize_parsed_events(
        {"events": [{"event_type": "中标", "arguments": []}], "ignored": True}
    )

    assert parsed == [{"event_type": "中标", "arguments": []}]


def test_parse_event_list_with_diagnostics_returns_normalized_event_list():
    module = importlib.import_module("oglans.utils.json_parser")

    events, diagnostics = module.parse_event_list_with_diagnostics(
        '```json\n{"events":[{"event_type":"中标","arguments":[]}]}\n```'
    )

    assert diagnostics["success"] is True
    assert events == [{"event_type": "中标", "arguments": []}]
