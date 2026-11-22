import wave
from types import SimpleNamespace

import numpy as np
import pytest

from shorts_clipper.captions import Caption, CaptionStyle, to_ass
from shorts_clipper.director import (DirectorOptions, cut_source, effects_for, hook_title_ass, loudness, mix_audio,
                                     mix_command, plan_edit, quality_check, remap_captions)
from shorts_clipper.story import (classify_moment, game_name, hook_language, intro_length, story_arc, streamer_name,
                                  template_hooks, write_hooks)
from shorts_clipper.video import make_vertical_clip


def word(start, text, length=0.35):
    return SimpleNamespace(start=start, end=start + length, word=text, probability=1.0)


def levels(duration, loud=(), quiet=()):
    data = np.full(int(duration / 0.1), -24.0) + np.sin(np.arange(int(duration / 0.1))) * 2
    for a, b in quiet:
        data[int(a / .1):int(b / .1)] = -80
    for a, b in loud:
        data[int(a / .1):int(b / .1)] = -6
    return data


def speech(spans, text="jalan terus"):
    words = []
    for a, b in spans:
        t = a
        while t + 0.35 <= b:
            words.append(word(t, text.split()[len(words) % len(text.split())]))
            t += 0.45
    return words


def captions(words):
    return [Caption(chunk[0].start, chunk[-1].end, " ".join(w.word for w in chunk), chunk)
            for chunk in (words[i:i + 4] for i in range(0, len(words), 4))]


def test_streamer_game_and_language_from_metadata():
    assert streamer_name({"channel": "Windah Basudara", "uploader": "x"}) == "Windah Basudara"
    assert streamer_name({"uploader": "MiawAug Official"}) == "MiawAug"
    assert streamer_name({"channel": "pewdiepie"}) == "Pewdiepie"
    assert streamer_name({"channel": "1234"}) is None and streamer_name({}) is None
    assert game_name("WINDAH MAIN GTA 6 PERTAMA KALI") == "GTA VI"
    assert game_name("Minecraft Hardcore day 100") == "Minecraft" and game_name("Just chatting") is None
    assert hook_language("id") == "id" and hook_language("en", "main bareng kalian") == "en"
    assert hook_language(None, "Main GTA bareng kalian sampai tamat") == "id"


def test_moment_type_follows_dialogue_and_chat():
    near = [word(10, "eh"), word(10.5, "kena"), word(11, "batunya"), word(11.5, "wkwk")]
    assert classify_moment(near, ["KARMA", "wkwkwk"], 11) == "karma"
    assert classify_moment([word(10, "AAAAH"), word(10.5, "takut")], ["jumpscare lol"], 10) == "scare"
    assert classify_moment([word(10, "let's go"), word(11, "clutch")], ["W", "GG"], 10) == "win"
    assert classify_moment([word(10, "hmm")], [], 10) == "shock"


def test_hooks_name_the_streamer_tease_and_differ_per_clip():
    hooks = template_hooks("karma", "id", "Windah Basudara", "GTA VI", seed="a")
    assert "Momen Ketika Windah Basudara Terkena Karma di GTA VI!" in hooks
    assert all(len(h) <= 64 and "Windah Basudara" in h for h in hooks)
    english = template_hooks("karma", "en", "Windah Basudara", None, seed="a")
    assert "The Moment Windah Basudara Gets Instant Karma!" in english
    first = template_hooks("fail", "en", "Ninja", "Fortnite", seed="clip1")[0]
    assert template_hooks("fail", "en", "Ninja", "Fortnite", seed="clip1", used=[first])[0] != first
    assert template_hooks("unknown", "en")  # unknown types fall back to a generic tease


def test_write_hooks_uses_llm_when_configured(monkeypatch):
    import io
    import json
    monkeypatch.setenv("CLIPPER_LLM_API_KEY", "test")
    calls = []

    def fake(request, timeout):
        calls.append(json.loads(request.data))
        body = {"choices": [{"message": {"content": json.dumps({"hooks": ["Windah Kira Dia Aman... Ternyata", "x" * 90]})}}]}
        return io.BytesIO(json.dumps(body).encode())
    monkeypatch.setattr("urllib.request.urlopen", fake)
    result = write_hooks({"name": "Windah", "language": "id"}, [word(5, "aduh")], 5)
    assert result["writer"] == "llm" and result["options"] == ["Windah Kira Dia Aman... Ternyata"]
    assert "Indonesian" in calls[0]["messages"][0]["content"]
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert write_hooks({"name": "Windah", "language": "id"}, [word(5, "aduh")], 5)["writer"] == "local"


def test_story_arc_builds_up_before_the_event_and_ends_after_the_reaction():
    words = speech([(0.5, 4), (5, 19.5), (20.5, 26)]) + speech([(31, 36)])
    data = levels(40, loud=[(20, 23)], quiet=[(26.5, 30.8)])
    arc = story_arc(data, words, 40, peak=21)
    assert 19.5 <= arc["event"] <= 21
    assert 5 <= arc["event"] - arc["start"] <= 14.5
    assert 26 <= arc["end"] <= 27.5
    assert not any(w.start < arc["end"] < w.end for w in words)
    # A long reaction keeps going until it settles instead of a fixed length.
    long = story_arc(levels(40, loud=[(20, 23)], quiet=[(33.5, 40)]), speech([(5, 33)]), 40, peak=21)
    assert long["end"] > 33
    # Non-stop talking: both boundaries still fall between words.
    words = speech([(0, 40)])
    nonstop = story_arc(levels(40, loud=[(20, 23)]), words, 40, peak=21)
    assert not any(w.start + 0.01 < t < w.end - 0.01 for w in words for t in (nonstop["start"], nonstop["end"]))


def test_intro_length_fits_text_and_voice():
    assert 2.5 <= intro_length("Short one") <= 3.3
    assert intro_length("x" * 60) == 3.3
    assert intro_length("Short", voice=3.2) == 3.6


def test_plan_opens_with_story_intro_then_build_up_event_and_reaction():
    words = speech([(0.5, 4), (5, 19.5), (20.5, 26)]) + speech([(31, 36)])
    context = {"name": "Windah Basudara", "game": "GTA VI", "language": "id", "messages": ["KARMA wkwk"],
               "seed": "c1"}
    plan = plan_edit(levels(40, loud=[(20, 23)], quiet=[(26.5, 30.8)]), captions(words), 40, DirectorOptions(),
                     peak=21, story=context)
    intro, *rest = plan.segments
    assert intro.role == "intro" and 2.5 <= intro.length <= 3.31
    assert intro.start == rest[0].start and all(s.role == "main" for s in rest)
    hook = plan.hook
    assert hook["style"] == "story" and "Windah Basudara" in hook["text"] and hook["category"] == "karma"
    assert hook["text"] == hook["options"][0] and hook["build_up"] >= 5 and hook["payoff"] >= 3.5
    assert plan.map_time(hook["event"]) > intro.length
    assert [kind for _, kind in plan.sfx][:2] == ["hit", "boing"]
    assert plan.score["hook"] == 80  # scored on the written hook, not the original opening
    shown = remap_captions(captions(words), plan)
    assert shown[0].start >= intro.length - 0.01  # no captions over the hook card
    custom = plan_edit(levels(40, loud=[(20, 23)]), captions(words), 40,
                       DirectorOptions(hook_text="Windah Pikir Dia Aman..."), peak=21, story=context)
    assert custom.hook["text"] == "Windah Pikir Dia Aman..." and len(custom.hook["options"]) >= 2
    teaser = plan_edit(levels(40, loud=[(20, 23)]), captions(words), 40, DirectorOptions(story=False), peak=21,
                       story=context)
    assert teaser.segments[0].role == "hook"
    plain = plan_edit(levels(40, loud=[(20, 23)]), captions(words), 40, DirectorOptions(hook=False), peak=21,
                      story=context)
    assert plain.hook is None and all(s.role == "main" for s in plain.segments)


def test_story_title_pops_words_and_highlights_the_name():
    ass = hook_title_ass(to_ass([], CaptionStyle()), "Momen Ketika Windah Terkena Karma!", 3, top=700, story=True,
                         highlight="Windah")
    line = ass.splitlines()[-1]
    assert line.count(r"\alpha&HFF&") == 5 and r"\1c&H00D7FF&\t" in line.split("WINDAH")[0][-40:]
    assert "Hook,Montserrat Black,96" in ass


def test_voiceover_is_mixed_in(tmp_path):
    plan = plan_edit(levels(10), [], 10, DirectorOptions(), story={"language": "en"})
    plan.voiceover = str(tmp_path / "v.wav")
    command = mix_command(tmp_path / "in.mp4", tmp_path / "out.mp4", plan)
    assert command[command.index("-i", command.index("-i") + 1) + 1] == plan.voiceover
    assert "[1:a]" in command[command.index("-filter_complex") + 1]


@pytest.fixture
def source(tmp_path):
    import subprocess
    path = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=20",
                    "-f", "lavfi", "-i", "sine=frequency=330:duration=20", "-filter_complex",
                    "[1:a]volume='if(between(t,12,14),1,0.08)':eval=frame[a]", "-map", "0:v", "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def test_full_story_render_with_voiceover(source, tmp_path):
    words = speech([(0.5, 11.5), (12.2, 15.5)], "no way bro")
    voice = tmp_path / "voice.wav"
    with wave.open(str(voice), "wb") as wav:
        wav.setnchannels(1), wav.setsampwidth(2), wav.setframerate(22050)
        wav.writeframes((np.sin(np.arange(22050 * 2) * 0.05) * 8000).astype("<i2").tobytes())
    context = {"name": "Ninja", "language": "en", "voice": lambda text, language: (voice, 2.0)}
    plan = plan_edit(loudness(source, 20), captions(words), 20, DirectorOptions(voiceover=True), peak=12.5,
                     story=context)
    assert plan.segments[0].role == "intro" and plan.hook["voiceover"] and plan.voiceover == str(voice)
    directed = cut_source(source, tmp_path / "cut.mp4", plan)
    plan.focal = [.5] * len(plan.segments)
    shown = remap_captions(captions(words), plan)
    subtitles = tmp_path / "c.ass"
    subtitles.write_text(hook_title_ass(to_ass(shown, CaptionStyle()), plan.hook["text"], plan.hook["duration"],
                                        top=700, story=True, highlight="Ninja"), encoding="utf-8")
    video = make_vertical_clip(directed, tmp_path / "clip.mp4", 0, plan.duration, "crop", subtitles,
                               effects=effects_for(plan))
    mixed = mix_audio(video, tmp_path / "mixed.mp4", plan)
    checks = {c["check"]: c for c in quality_check(mixed, plan, shown)}
    assert checks["Format"]["status"] == "pass" and checks["Subtitle timing"]["status"] == "pass"
    assert checks["Visual consistency"]["status"] == "pass"
