import {runInNewContext} from 'node:vm';
import app from '../clipper_studio/static/app.js' with {type:'text'};

Deno.test('changing the speech language previews a sample instead of a stale story hook', async () => {
  const listeners=[], elements=new Map(), requests=[];
  const clip={
    id:'clip', language:'id', caption_language:'id', narration_url:'/clip/narration',
    director_result:{edits:{hook:{style:'story', language:'id', text:'Momen ketika dia jatuh!'}}},
  };
  const element=selector=>{
    if(!elements.has(selector)) elements.set(selector,{
      value:selector==='#editing-mode'?'full':'', parentElement:{}, removeAttribute(){}, append(){},
      querySelector:()=>null,
      addEventListener(event, handler) {
        if(selector==='#clips' && event==='click') listeners.push(handler);
      },
    });
    return elements.get(selector);
  };
  const language={value:'id'}, hookInput={value:''};
  const node={dataset:{id:clip.id},querySelector(selector) {
    if(selector==='[data-caption=language]') return language;
    if(selector==='[data-director=hook_text]') return hookInput;
    if(selector==='.voice-status') return element(selector);
    if(selector==='[data-field=editing_mode]') return {value:'full'};
    if(['[data-director=voice]','[data-director=delivery]'].includes(selector)) return {value:'auto'};
    return {type:'checkbox', checked:['[data-director=enabled]','[data-director=hook]','[data-director=story]'].includes(selector)};
  }};
  runInNewContext(app+'\ncurrent={id:"job",clips:[clip]};', {
    clip, document:{querySelector:element}, location:{hash:''},
    CaptionControls:{html(){},preview(){}}, PodcastControls:{read(){},defaults(){}},
    ResizeObserver:class {observe(){}}, addEventListener(){}, scrollTo(){},
    setInterval(){}, clearInterval(){}, setTimeout(){}, clearTimeout(){}, AbortController,
    Audio:class {async play(){}}, URL:{createObjectURL(){}},
    fetch(url, options) {
      if(url==='./api/voices/preview' || url==='./clip/narration') {
        requests.push({url,body:options?.body && JSON.parse(options.body)});
        return Promise.resolve({ok:true,blob:()=>Promise.resolve(null),headers:{get:()=>''}});
      }
      return Promise.resolve({ok:false,json:()=>Promise.resolve(null)});
    },
  });
  const button={dataset:{action:'voice-preview'},closest:()=>node};
  const preview=async()=>{
    for(const listener of listeners) await listener({target:{closest:selector=>selector==='button'?button:null}});
    return requests.at(-1);
  };
  if((await preview())?.url!=='./clip/narration') throw Error('Expected the rendered narration');
  language.value='en';
  const sample=await preview();
  if(sample?.body?.language!=='en' || sample.body.text!==null) {
    throw Error(`Expected an English sample request with no hook text; got ${JSON.stringify(sample)}`);
  }
  hookInput.value='Watch this!';
  if((await preview())?.body?.text!=='Watch this!') throw Error('Expected the explicit hook text');
});
