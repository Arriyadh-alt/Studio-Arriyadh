import re
import sys
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from shorts_clipper.captions import (
    Caption, CaptionStyle, FONTS, Transcript, _classic_ass, ass_color,
    chunk_words, emphasized_words, to_ass, transcribe,
)


def word(text, start, end):
    return SimpleNamespace(word=text, start=start, end=end)


@pytest.mark.parametrize('value', [None, [], 'pop', 1])
def test_style_invalid_container(value):
    assert CaptionStyle.from_dict(value) == CaptionStyle()


def test_style_validation_and_clamping():
    style = CaptionStyle.from_dict({'preset':'unknown','font':'evil,Font','size':999,
        'position':'left','text_color':'{bad}','highlight_color':'#aBcD12',
        'uppercase':'false','emphasis':False,'unknown':1})
    assert style == CaptionStyle(size=130, highlight_color='#ABCD12', emphasis=False)
    assert CaptionStyle.from_dict({'size':-8}).size == 40
    for bad in (True, None, '90', float('nan'), float('inf')):
        assert CaptionStyle.from_dict({'size':bad}).size == 80
    for font in FONTS:
        assert CaptionStyle.from_dict({'font':font}).font == font
    assert CaptionStyle.from_dict({'uppercase':True}).uppercase


def test_ass_color():
    assert ass_color('#12abEF') == '&HEFAB12&'
    assert ass_color('red') == '&HFFFFFF&'


def test_pop_contiguous_windows_and_fixed_wrap():
    words = [word(' Incredible', 0, .2), word(' wonderful', .3, .5),
             word(' opportunity!', .55, .7)]
    captions = chunk_words(words, clip_end=1)
    output = to_ass(captions, {'preset':'pop','size':100})
    events = [line.split(',', 9) for line in output.splitlines() if line.startswith('Dialogue:')]
    assert len(events) == 3
    assert events[0][1] == '0:00:00.00'
    assert events[-1][2] == '0:00:00.80'
    assert all(a[2] == b[1] for a, b in zip(events, events[1:]))
    plain = [re.sub(r'\{[^}]*\}', '', event[9]) for event in events]
    assert len(set(plain)) == 1 and r'\N' in plain[0]
    for event, token in zip(events, ('Incredible','wonderful','opportunity!')):
        highlighted = re.findall(r'\{[^}]*\\c&H4DE3FF&[^}]*\}([^{}]*)', event[9])
        assert ''.join(highlighted).replace(r'\N', '').strip() == token
        assert r'\t(0,100,\fscx100\fscy100)' in event[9]


def test_classic_is_exactly_legacy_and_static():
    captions = chunk_words([word(' Hello', 0, .4), word(' world!', .5, 1)])
    assert to_ass(captions, {'preset':'classic'}) == _classic_ass(captions)
    assert r'\t(' not in to_ass(captions, {'preset':'classic'})


def test_styles_empty_literal_uppercase_position_and_minimal():
    assert 'Dialogue:' not in to_ass([], {})
    captions = [Caption(0, 1, '{hi}\\N world')]
    output = to_ass(captions, {'uppercase':True,'position':'top','font':'Liberation Sans','text_color':'#123456'})
    assert ',8,90,190,260,1' in output
    assert 'Liberation Sans,80,&H00563412' in output
    assert '｛HI｝＼N WORLD' in output
    minimal = to_ass(captions, {'preset':'minimal','position':'middle'})
    assert r'\fad(120,80)' in minimal and r'\t(' not in minimal
    assert ',5,90,190,0,1' in minimal
    assert 'DejaVu Sans,64,' in minimal


@pytest.mark.parametrize('tokens,expected', [
    (['the','amazing','WIN!','100%'], {2,3}),
    (['yang','dan','banget','keren!'], {2,3}),
    (['di','ke','itu','ini','aja','sih','dong','nih','kok','udah','gak','nggak'], set()),
    (['number','123'], {0,1}),
])
def test_emphasis_en_and_id(tokens, expected):
    assert emphasized_words(tokens) == expected


def test_emphasis_can_be_disabled():
    captions = chunk_words([word(' AMAZING',0,.2),word('100%',.3,.6)])
    assert '&HB5F567&' in to_ass(captions, {})
    assert '&HB5F567&' not in to_ass(captions, {'emphasis':False})


def test_transcription_language_env_and_list_compatibility(monkeypatch):
    recognize = Mock()
    recognize.transcribe.return_value = ([SimpleNamespace(words=[word(' Halo!',0,1)])], SimpleNamespace(language='id'))
    model = Mock(return_value=recognize)
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=model))
    monkeypatch.setenv('CLIPPER_WHISPER_MODEL','small')
    result = transcribe('source.mp4', language='auto', clip_end=2)
    assert isinstance(result, list) and isinstance(result, Transcript)
    assert result == [Caption(0,1,'Halo!')] and result.language == 'id'
    assert result[0].words[0].word == ' Halo!'
    model.assert_called_once_with('small', device='cpu', compute_type='int8')
    recognize.transcribe.assert_called_once_with('source.mp4', language=None, word_timestamps=True)
    transcribe('source.mp4', model='base', language='id')
    assert model.call_args.args == ('base',)
    assert recognize.transcribe.call_args.kwargs['language'] == 'id'


def test_transcription_empty_and_error(monkeypatch):
    recognize = Mock()
    recognize.transcribe.return_value = ([], SimpleNamespace(language='en'))
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=Mock(return_value=recognize)))
    assert transcribe('silent.mp4') == []
    recognize.transcribe.side_effect = RuntimeError('model unavailable')
    with pytest.raises(RuntimeError, match='model unavailable'):
        transcribe('source.mp4')


def test_whisper_audio_decode_dependency_compatibility(tmp_path):
    from faster_whisper.audio import decode_audio
    source = tmp_path / 'silence.wav'
    with wave.open(str(source), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b'\0\0' * 1600)
    assert len(decode_audio(str(source))) == 1600
