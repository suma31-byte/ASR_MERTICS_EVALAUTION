"""The checks that stop a run producing numbers nobody can trust."""


def test_a_changed_output_device_is_detected(ev, monkeypatch):
    state = {"name": "Speakers (Realtek)"}
    monkeypatch.setattr(ev, "current_output_device",
                        lambda: {"index": 1, "name": state["name"], "samplerate": 48000.0})
    ev.pin_output_device()
    assert ev.output_device_changed() is None
    state["name"] = "Headphones (USB)"
    assert "changed" in ev.output_device_changed()


def test_resource_limits(ev, monkeypatch):
    monkeypatch.setattr(ev, "phone_battery_pct", lambda s: 80)
    monkeypatch.setattr(ev, "phone_free_mb", lambda s: 4000)
    monkeypatch.setattr(ev, "pc_free_mb", lambda: 50000)
    assert ev.check_resources(None, loud=False) is None

    monkeypatch.setattr(ev, "phone_battery_pct", lambda s: 5)
    assert "battery" in ev.check_resources(None, loud=False)

    monkeypatch.setattr(ev, "phone_battery_pct", lambda s: 80)
    monkeypatch.setattr(ev, "phone_free_mb", lambda s: 100)
    assert "phone has" in ev.check_resources(None, loud=False)


def test_a_result_that_cannot_be_saved_raises(ev, monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "RESULTS_CSV", tmp_path / "no-such-dir" / "out.csv")
    monkeypatch.setattr(ev, "load_all_results", lambda: {})
    row = {k: "" for k in ev.RESULT_FIELDS} | {"id": "1"}
    try:
        ev.append_result(row)
    except ev.ResultWriteError:
        return
    raise AssertionError("a failed save must stop the run, not pass silently")


def test_adb_retries_then_succeeds(ev, monkeypatch):
    import types
    attempts = []

    def fake_once(args, serial, timeout):
        attempts.append(args)
        if len(attempts) < 3:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="device offline")
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(ev, "_adb_once", fake_once)
    monkeypatch.setattr(ev, "restart_adb_server", lambda s: True)
    assert ev.adb(["shell", "echo"]).strip() == "ok"
    assert len(attempts) == 3
