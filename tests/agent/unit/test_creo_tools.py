"""Creo review tools: behaviour flags, reply parsing, settings and dry runs. No network is used."""

from __future__ import annotations

import json
import sys

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))


def step(t, x, y, a=0.0, b=("fwd",), enemies=0, front=500, dmg=0.0, ev=()):
    return dict(t=t, x=x, y=y, a=a, b=list(b), hp=100, enemies=enemies, front=front, dmg=dmg, ev=list(ev),
                goal=None, notes=0)


def kinds(flags):
    return {f["kind"] for f in flags}


def test_walking_straight_raises_no_flags():
    from episode_trace import behaviour_flags

    steps = [step(i, 20.0 * i, 0.0) for i in range(200)]
    assert behaviour_flags(steps) == []


def test_standing_still_is_stuck_but_not_while_fighting():
    from episode_trace import behaviour_flags

    idle = [step(i, 100.0, 100.0, b=("turnL",)) for i in range(40)]
    assert "stuck" in kinds(behaviour_flags(idle))
    fight = [step(i, 100.0, 100.0, b=("attack",), enemies=2) for i in range(40)]
    assert "stuck" not in kinds(behaviour_flags(fight))


def test_spinning_wall_hugging_weapon_cycling_and_passive_fire():
    from episode_trace import behaviour_flags

    spin = [step(i, 50.0, 50.0, a=(i * 45.0) % 360, b=("turnR",)) for i in range(20)]
    assert "spinning" in kinds(behaviour_flags(spin))
    hug = [step(i, 10.0 + 0.1 * i, 0.0, front=20) for i in range(30)]
    assert "wall_hugging" in kinds(behaviour_flags(hug))
    cycle = [step(i, 20.0 * i, 0.0, b=("fwd", "weapon")) for i in range(40)]
    assert "weapon_cycling" in kinds(behaviour_flags(cycle))
    passive = [step(i, 20.0 * i, 0.0, enemies=1, dmg=2.0) for i in range(30)]
    assert "passive_fire" in kinds(behaviour_flags(passive))


def test_no_new_area_is_skipped_when_the_run_ends_with_exit():
    from episode_trace import behaviour_flags

    loop = [step(i, 64.0 * ((i // 2) % 2), 0.0) for i in range(300)]
    assert "no_new_area" in kinds(behaviour_flags(loop))
    loop[-1]["ev"] = ["EXIT"]
    flags = [f for f in behaviour_flags(loop) if f["kind"] == "no_new_area"]
    assert all(f["end"] < len(loop) - 1 for f in flags)


def test_summary_mentions_outcome_flags_and_buttons():
    from episode_trace import behaviour_flags, summarize

    steps = [step(i, 100.0, 100.0, b=("turnL", "weapon")) for i in range(60)]
    steps[-1]["ev"] = ["DIED"]
    trace = dict(map="maps/x.wad:MAP01", seed=0, tier=2, kind="rooms", episode_ticks=2800, steps=steps,
                 result=dict(exit=False, dead=True, keys=0, doors=0, locked_doors=0, areas=1, kills=0, damage=100,
                             health=0, score=-15.0, weighted_score=-30.0))
    text = summarize(trace, behaviour_flags(steps))
    assert "DIED" in text and "stuck" in text and "weapon 100%" in text


def test_test_maps_are_refused():
    from episode_trace import resolve_jobs

    with pytest.raises(SystemExit, match="test map"):
        resolve_jobs(None, ["maps/curated/test/t1-arena-200000.wad:MAP01"], 1)


def test_parse_json_reply_handles_fences_and_noise():
    from creo import parse_json_reply

    assert parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_reply('Sure! {"verdict": "mixed"} Hope this helps.') == {"verdict": "mixed"}
    assert "_unparsed" in parse_json_reply("no json here")


def test_settings_file_then_environment(tmp_path, monkeypatch):
    from creo import load_settings

    env = tmp_path / ".creo.env"
    env.write_text('CREO_API_KEY="from-file"\nCREO_BASE_URL=https://example.invalid/v1\n# comment\n', encoding="utf-8")
    monkeypatch.delenv("CREO_API_KEY", raising=False)
    monkeypatch.setenv("CREO_MODEL", "m-env")
    s = load_settings(env)
    assert s["CREO_API_KEY"] == "from-file" and s["CREO_BASE_URL"].endswith("/v1") and s["CREO_MODEL"] == "m-env"


def test_dry_run_writes_request_without_secrets(tmp_path, monkeypatch):
    import creo

    monkeypatch.setattr(creo, "DRY_DIR", tmp_path)
    client = creo.Creo(model="m", dry_run=True, settings={"CREO_API_KEY": "sk-secret-123", "CREO_BASE_URL": "https://x/v1"})
    assert client.chat([{"role": "user", "content": "hi"}], json_mode=True, tag="t") == "{}"
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    body = files[0].read_text(encoding="utf-8")
    assert "sk-secret-123" not in body and json.loads(body)["response_format"] == {"type": "json_object"}


def test_missing_settings_fail_clearly():
    import creo

    with pytest.raises(creo.CreoError, match="CREO_API_KEY"):
        creo.Creo(settings={}, dry_run=False)


def test_mock_review_context_respects_budget():
    import mock_review

    ctx = mock_review.build_context(20000)
    assert "docs/brief/BRIEF.md" in ctx and len(ctx) <= 21000
