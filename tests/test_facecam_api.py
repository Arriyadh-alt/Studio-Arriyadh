import json
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from clipper_studio.api import create_app, Store, FacecamSpec
from shorts_clipper.facecam import SourceBox, layout_facecam


@pytest.mark.parametrize('spec', [
    {'enabled':'yes'}, {'size':.1}, {'size':.7}, {'size':'0.4'}, {'position':'middle'},
    {'box':{'x':.9,'y':0,'w':.2,'h':.2}}, {'box':{'x':0,'y':0,'w':0,'h':.2}},
    {'box':{'x':0,'y':0,'w':.2,'h':.2,'extra':1}}, {'extra':True},
])
def test_validation(tmp_path, spec):
    with TestClient(create_app(tmp_path)) as client:
        assert client.post('/api/jobs', json={'url':'https://youtu.be/abcdefghijk','facecam':spec}).status_code == 422
        assert client.post('/api/jobs/missing/clips/missing/render', json={'start':0,'end':1,'facecam':spec}).status_code == 422


def test_legacy_defaults(tmp_path):
    store=Store(tmp_path)
    store.save({'id':'old','request':{},'clips':[{'id':'clip'}]})
    job=store.get('old')
    assert job['request']['facecam'] == FacecamSpec().model_dump()
    assert job['clips'][0]['facecam'] == FacecamSpec().model_dump()


@pytest.mark.parametrize('scenario', ['detected','manual','none','failure','render_failure','disabled','still_failure'])
def test_worker(tmp_path, monkeypatch, scenario):
    monkeypatch.setattr('clipper_studio.api.fetch_metadata', lambda *a: {'duration':60})
    monkeypatch.setattr('clipper_studio.api.fetch_chat', lambda *a: ({'duration':60}, []))
    monkeypatch.setattr('clipper_studio.api.detect_spikes', lambda *a,**k: ([{'start':0,'end':10}], []))
    monkeypatch.setattr('clipper_studio.api.download_section', lambda url,work,*a: work/'source.mp4')
    monkeypatch.setattr('clipper_studio.api.video_duration', lambda *a: 10)
    def still(source, output, *a):
        if scenario=='still_failure':
            raise RuntimeError('still')
        output.write_bytes(b'jpeg')
    monkeypatch.setattr('clipper_studio.api.save_source_still', still)
    box=SourceBox(.7,.1,.2,.2)
    detection=MagicMock(return_value=None if scenario=='none' else box)
    if scenario=='failure':
        detection.side_effect=RuntimeError('detection')
    monkeypatch.setattr('clipper_studio.api.detect_facecam', detection)
    overlay=layout_facecam(box,(640,360),'top-right')
    auto=MagicMock(return_value=overlay)
    monkeypatch.setattr('clipper_studio.api.auto_layout_facecam', auto)
    calls=[]
    def render(source, output, *a, **kwargs):
        calls.append(kwargs.get('facecam'))
        if scenario=='render_failure' and kwargs:
            raise RuntimeError('overlay filter')
        output.write_bytes(b'video')
    monkeypatch.setattr('clipper_studio.api.make_vertical_clip', render)
    monkeypatch.setattr('clipper_studio.api.subprocess.run', lambda cmd,**k: Path(cmd[-1]).write_bytes(b'thumb'))
    spec={'enabled':scenario!='disabled'}
    if scenario=='manual':
        spec['box']=vars(box)
    app=create_app(tmp_path)
    with TestClient(app) as client:
        jid=client.post('/api/jobs',json={'url':'https://youtu.be/abcdefghijk','captions':False,'facecam':spec}).json()['id']
        app.state.pending.join()
        job=client.get(f'/api/jobs/{jid}').json()
        clip=job['clips'][0]
        assert clip['status']=='ready'
        assert clip['facecam']==job['request']['facecam']
        if scenario in ('detected','manual','still_failure'):
            assert calls==[overlay]
            assert clip['facecam_result']['position']=='top-right'
            assert clip['facecam_result']['origin']==('manual' if scenario=='manual' else 'detected')
        elif scenario=='render_failure':
            assert calls==[overlay,None]
        else:
            assert calls==[None]
        if scenario in ('manual','disabled'):
            detection.assert_not_called()
        if scenario=='still_failure':
            assert 'source_still_url' not in clip
        else:
            response=client.get(clip['source_still_url'])
            assert response.content==b'jpeg' and response.headers['content-type']=='image/jpeg'
        base=f'/api/jobs/{jid}/clips/{clip["id"]}'
        assert client.post(base+'/render',json={'start':0,'end':5,'captions':False,'facecam':{'enabled':False}}).status_code==202
        app.state.pending.join()
        assert client.get(f'/api/jobs/{jid}').json()['clips'][0]['facecam']['enabled'] is False
