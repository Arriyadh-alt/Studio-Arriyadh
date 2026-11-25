globalThis.MomentReview = (() => {
  const panel = document.querySelector('#moments-panel');
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const time = value => {
    const seconds = Math.max(0, Math.round(value));
    return `${Math.floor(seconds/3600)}:${String(Math.floor(seconds%3600/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
  };
  const parse = value => {
    if (!/^\d+(?:\.\d+)?$|^\d+(?::[0-5]\d){1,2}(?:\.\d+)?$/.test(value.trim())) throw Error('Use seconds, MM:SS or H:MM:SS.');
    return value.trim().split(':').reduce((total, part) => total*60 + Number(part), 0);
  };
  const filters = {all:'Ranked (all)',ai:'AI-detected',chat:'Chat-detected',both:'Both'};
  const podcast = () => job?.request?.content_type==='podcast';
  const filterSet = () => podcast() ? PodcastControls.filters : filters;
  const matches = (m, filter) => filter==='all' ? (m.sources.includes('ai') || ['selected','rendered'].includes(m.status) ||
    !job.moments.some(item=>item.score_breakdown)) : podcast() ? PodcastControls.dimension(m,filter)!=null : filter==='both' ? m.sources.length>1 : m.sources.includes(filter);
  const ordered = () => [...job.moments].sort((a,b)=>podcast() && filter!=='all' ? (PodcastControls.dimension(b,filter)??0)-(PodcastControls.dimension(a,filter)??0) : a.rank-b.rank);
  let job, filter='all', key='', active=null, pending=false, showHidden=false;
  const drafts = new Map();
  const errors = new Map();
  const source = m => m.podcast ? 'Conversation AI' : m.sources.length>1?'AI + Chat':m.sources.includes('ai')?'AI Analysis':'Chat Analysis';
  const range = m => `${time(m.start)}–${time(m.end)}`;
  const tiers = {high_priority:'High priority',review:'Recommended for review',needs_improvement:'Needs improvement',hidden:'Below recommendation threshold'};
  const components = {hook:'Hook',story:'Context',emotion:'Emotion',visual:'Visual relevance',payoff:'Payoff',retention:'Retention estimate'};
  const visible = m => showHidden || m.score_tier!=='hidden' || ['selected','rendered'].includes(m.status) ||
    (m.reviewed && m.status==='suggested');
  const originalRange = m => m.alternatives?.ai || m.original_ai_range || m.original_chat_range || m;
  const option = (value,label,selected) => `<option value="${esc(value)}" ${value===selected?'selected':''}>${esc(label)}</option>`;
  const seconds = value => `${Number(value).toFixed(1)} s`;
  function evidence(m, clip) {
    const reaction=m.reaction, penalties=Object.entries(m.score_penalties || {});
    const checks=Array.isArray(clip?.quality_validation)?clip.quality_validation:clip?.director_result?.quality;
    const resolution=clip?.export_resolution;
    const format=checks?.find(q=>q.check==='Format')?.detail;
    const actual=clip?.rendered_duration!=null ? seconds(clip.rendered_duration) : format?.match(/[\d.]+ s/)?.[0];
    const dimensions=resolution?.width && resolution?.height ? `${resolution.width}×${resolution.height}` : format?.match(/\d+×\d+/)?.[0];
    const outside=reaction?.verified && (reaction.trigger_time<m.start || reaction.trigger_time>=m.end);
    return `<dl class="moment-facts"><div><dt>Original source</dt><dd>${range(originalRange(m))}</dd></div>
      <div><dt>Proposed edit</dt><dd>${seconds(m.duration)}</dd></div>
      ${actual? `<div><dt>Rendered duration</dt><dd>${esc(actual)}</dd></div>` : ''}
      ${dimensions? `<div><dt>Export resolution</dt><dd>${esc(dimensions)}</dd></div>` : ''}</dl>
      ${m.score_breakdown?`<dl class="moment-score-grid">${Object.entries(components).map(([id,label])=>`<div><dt>${label}</dt><dd>${esc(m.score_breakdown[id])}</dd></div>`).join('')}</dl>
      <details class="moment-explanation"><summary>Why this score?</summary><ul>${Object.entries(components).map(([id,label])=>`<li><strong>${label}:</strong> ${esc(m.score_explanation?.[id])}</li>`).join('')}</ul>
      ${penalties.length?`<p>Penalties: ${penalties.map(([name,value])=>`${esc(name.replaceAll('_',' '))} −${esc(value)}`).join(' · ')}</p>`:''}
      ${m.score_penalties?.misleading_hook?'<p>The edited range excludes the evidence for the selected hook.</p>':''}</details>`:''}
      ${reaction?`<div class="moment-trigger"><strong>Reaction trigger</strong>
        <p>${reaction.verified?`${time(reaction.trigger_time)} · ${esc(reaction.subject || reaction.evidence?.join(', '))}`:'Not verified. Needs manual review.'}</p>
        <p>Noticed ${time(reaction.notice_time)} · Peak ${time(reaction.peak_time)}${reaction.conclusion_time!=null?` · Conclusion ${time(reaction.conclusion_time)}`:''}</p>
        ${outside?'<p class="moment-warning">The edited range excludes the identified trigger.</p>':''}</div>`:''}
      ${checks?.length?`<details class="moment-explanation"><summary>Quality validation</summary><ul>${checks.map(q=>`<li><strong>${esc(q.check)} (${esc(q.status)}):</strong> ${esc(q.detail)}</li>`).join('')}</ul></details>`:''}`;
  }
  function renderFields(m, draft) {
    if(podcast()) return '';
    const settings=m.render_options || {}, request=job.request || {};
    const current=(k,fallback)=>draft[k] ?? settings[k] ?? fallback;
    const framing=Object.hasOwn(settings,'framing')?settings.framing:request.tracking?.x;
    return `<details class="moment-advanced"><summary>Subtitles, framing, layout &amp; voiceover</summary>
      <label>Frame fit<select name="mode">${option('crop','Crop',current('mode', request.mode || 'crop'))}${option('blur','Blur background',current('mode', request.mode || 'crop'))}</select></label>
      <label>Layout<select name="layout">${Object.entries({auto:'Automatic',gameplay:'Gameplay focus',reaction:'Reaction focus',split:'Split screen'}).map(([v,l])=>option(v,l,current('layout',request.tracking?.layout || 'auto'))).join('')}</select></label>
      <label>Horizontal framing (%)<input name="framing" type="number" min="0" max="100" step="any" placeholder="Automatic" value="${esc(draft.framing ?? (framing==null?'':(framing*100).toFixed(3).replace(/\.?0+$/,'')))}"></label>
      <label>Subtitles<select name="captions">${option('true','On',String(current('captions',request.captions ?? true)))}${option('false','Off',String(current('captions',request.captions ?? true)))}</select></label>
      <label>AI hook voiceover<select name="voiceover">${option('false','Off (text only)',String(current('voiceover',request.director?.voiceover ?? false)))}${option('true','On',String(current('voiceover',request.director?.voiceover ?? false)))}</select></label>
      </details>`;
  }
  const previewUrl = m => {
    if(!job.url)return null;
    const url=new URL(job.url);
    const id=url.hostname==='youtu.be'?url.pathname.slice(1):url.searchParams.get('v') || url.pathname.split('/').pop();
    if(!/^[\w-]{11}$/.test(id))return null;
    return `https://www.youtube-nocookie.com/embed/${id}?start=${Math.floor(m.start)}&end=${Math.ceil(m.end)}&autoplay=1&playsinline=1`;
  };
  function card(m) {
    const draft=drafts.get(m.id) || {};
    const clip=job.clips.find(c=>c.id===m.clip_id);
    const rendered=m.status==='rendered';
    const scoreLabel=m.sources.includes('ai')?'Viral potential':'Chat reaction strength';
    return `<article class="moment-card ${m.status==='rejected'?'is-rejected':''}" data-moment="${esc(m.id)}">
      <div class="moment-meta"><strong class="moment-rank">#${m.rank}</strong><span>${range(m)} · ${Math.round(m.duration)}s</span><span class="moment-source">${source(m)}</span></div>
      <h3>${esc(m.title || 'Stream reaction')}</h3><p class="moment-description">${esc(m.description)}</p>${PodcastControls.momentHtml(m,rendered,job)}
      <div class="potential"><span>${scoreLabel}${tiers[m.score_tier]?` · ${tiers[m.score_tier]}`:''}</span><strong>${Math.round(m.score)}<small> / 100</small></strong></div><progress max="100" value="${m.score}" aria-label="${scoreLabel} ${m.score} out of 100"></progress>
      <p class="moment-reason">${esc(m.reason)}</p>
      ${evidence(m,clip)}
      ${m.alternatives?`<details class="moment-alternatives"><summary>Compare AI &amp; chat ranges</summary>${['ai','chat'].map(s=>`<div><span>${s==='ai'?'AI':'Chat'}: ${range(m.alternatives[s])}</span><button type="button" data-moment-action="${s}" ${rendered?'disabled':''}>Apply ${s==='ai'?'AI':'chat'}</button></div>`).join('')}</details>`:''}
      <div class="moment-player"></div><button type="button" class="moment-preview" data-moment-action="preview">${active===m.id?'■ Close preview':'▶ Preview'}</button>
      ${rendered?`${m.hooks?.length?`<ul class="moment-hook-list">${m.hooks.map(h=>`<li>${esc(h)}${h===m.selected_hook?' (selected)':''}</li>`).join('')}</ul>`:''}<a class="moment-rendered" href="#clip-${esc(m.clip_id)}" data-moment-action="highlight">Go to highlight #${clip?.number ?? job.clips.indexOf(clip)+1}</a>`:`<form class="moment-times">
      <label>Start<input name="start" aria-label="Moment ${m.rank} start" value="${esc(draft.start ?? time(m.start))}" inputmode="text"></label>
      <label>End<input name="end" aria-label="Moment ${m.rank} end" value="${esc(draft.end ?? time(m.end))}" inputmode="text"></label>
      <label class="moment-wide">Suggested title<input name="title" required maxlength="120" value="${esc(draft.title ?? m.title)}"></label>
      ${m.hooks?.length?`<fieldset class="moment-hooks moment-wide"><legend>Choose a hook</legend>${m.hooks.map(h=>`<label><input type="radio" name="selected_hook" value="${esc(h)}" ${h===(draft.selected_hook ?? m.selected_hook)?'checked':''}>${esc(h)}</label>`).join('')}</fieldset>`:''}
      ${renderFields(m,draft)}<button type="submit">Save review settings</button></form>
      <div class="moment-actions"><button type="button" data-moment-action="select" aria-pressed="${m.status==='selected'}">${m.status==='selected'?'✓ Approved':'Approve'}</button><button type="button" data-moment-action="reject" aria-pressed="${m.status==='rejected'}">${m.status==='rejected'?'↶ Undo reject':'Reject'}</button><button type="button" class="primary" data-moment-action="render">Render ↗</button></div>`}
      <p class="error moment-error" role="status" ${errors.has(m.id)?'':'hidden'}>${esc(errors.get(m.id))}</p></article>`;
  }
  function draw(value, force=false) {
    job=value;
    panel.hidden=!job.moments?.length;
    if(panel.hidden) {panel.innerHTML='';key='';active=null;return;}
    const next=JSON.stringify([job.id,job.moments,job.ai_error,filter,showHidden,job.clips.map(c=>[c.id,c.status,c.rendered_duration,c.export_resolution,c.director_result?.quality])]);
    if(!force && next===key) return;
    key=next;
    const iframe=panel.querySelector('.moment-player > *');
    const selected=job.moments.filter(m=>m.status==='selected').length;
    panel.innerHTML=`<div class="section-heading"><div><span class="eyebrow">Find the standout</span><h2>Detected moments <span class="hint">· ${job.moments.length}</span></h2></div></div>
      ${job.ai_error?`<p class="moment-warning">AI analysis unavailable — chat analysis only. ${esc(job.ai_error)}</p>`:''}
      <div class="moment-toolbar"><button class="primary" type="button" data-moment-action="batch" ${!selected?'disabled':''}>Render approved (${selected}) ↗</button><button type="button" data-moment-action="top">Approve top 5</button></div>
      <p class="hint">${podcast()?'Rendered with the Podcast Director: hook teaser, natural pauses kept, voice clarity, speaker framing, captions — original voices are never replaced.':'Rendered with AI Director: hook, voice-over, facecam on top, captions, zooms, loop — according to this job’s settings.'}</p>
      <p class="hint">Editorial estimates, not guaranteed views or measured audience retention.</p>
      ${job.moments.some(m=>m.score_tier==='hidden')?`<label class="check moment-hidden"><input type="checkbox" data-show-hidden ${showHidden?'checked':''}> Show scores below 55</label>`:''}
      <div class="moment-filters" role="group" aria-label="Filter moments">${Object.entries(filterSet()).map(([id,label])=>`<button type="button" data-filter="${id}" aria-pressed="${filter===id}">${label} <span>${job.moments.filter(m=>matches(m,id)).length}</span></button>`).join('')}</div>
      <p class="error moment-batch-error" role="status" hidden></p><div class="moment-grid">${ordered().filter(m=>visible(m)&&matches(m,filter)).map(card).join('') || '<p class="empty">No recommended moments in this filter.</p>'}</div>`;
    if(iframe && active) {
      const target=[...panel.querySelectorAll('[data-moment]')].find(n=>n.dataset.moment===active);
      if(target) {
        const moment=job.moments.find(m=>m.id===active);
        const url=previewUrl(moment);
        if(iframe.tagName==='IFRAME' && iframe.getAttribute('src')!==url)iframe.src=url;
        target.querySelector('.moment-player').append(iframe);
      }
      else active=null;
    }
    if(pending) panel.querySelectorAll('button').forEach(b=>b.disabled=true);
  }
  async function request(path, method, data) {
    const response=await fetch(`./api/jobs/${job.id}/moments${path}`,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const body=await response.json();
    if(!response.ok) throw Error(typeof body.detail==='string'?body.detail:body.detail?.map(d=>d.msg).join(' ') || 'Could not update moment.');
    return body;
  }
  async function action(node, fn) {
    if(pending) return;
    const id=node?.dataset.moment;
    const jid=job.id;
    const errorNode=node?.querySelector('.moment-error') || panel.querySelector('.moment-batch-error');
    errorNode.hidden=true;
    pending=true;
    panel.querySelectorAll('button').forEach(b=>b.disabled=true);
    try {
      await fn();
      if(job?.id!==jid) return;
      errors.delete(id);
      const response=await fetch(`./api/jobs/${jid}`,{cache:'no-store'});
      if(!response.ok) throw Error('Saved, but could not refresh. Please retry refresh.');
      const updated=await response.json();
      if(job?.id===jid) draw(updated,true);
    } catch(error) {
      if(job?.id!==jid) return;
      if(id) errors.set(id,error.message);
      errorNode.textContent=error.message;errorNode.hidden=false;
    } finally {
      pending=false;
      if(job?.id===jid) {
        // Redraw only after successful saves; failed edits keep focus and inline feedback.
        if(!errorNode.hidden && errorNode.isConnected) panel.querySelectorAll('button').forEach(b=>b.disabled=false);
        else draw(job,true);
      }
    }
  }
  panel.addEventListener('input',event=>{
    const node=event.target.closest('[data-moment]');
    if(!node || !event.target.matches('input,select'))return;
    if(!event.target.closest('.moment-times'))return;
    drafts.set(node.dataset.moment,Object.fromEntries(new FormData(node.querySelector('.moment-times'))));
  });
  panel.addEventListener('change',event=>{
    if(!event.target.matches('[data-show-hidden]'))return;
    showHidden=event.target.checked;
    draw(job,true);
    panel.querySelector('[data-show-hidden]')?.focus();
  });
  panel.addEventListener('submit',event=>{
    if(event.target.matches('.moment-podcast')) {
      event.preventDefault();
      const node=event.target.closest('[data-moment]');
      action(node,()=>request(`/${node.dataset.moment}`,'PATCH',{podcast:PodcastControls.readMoment(event.target)}));
      return;
    }
    if(!event.target.matches('.moment-times'))return;
    event.preventDefault();
    const node=event.target.closest('[data-moment]');
    action(node,async()=>{
      const data=new FormData(event.target);
      const moment=job.moments.find(m=>m.id===node.dataset.moment);
      const changes={};
      if(['start','end'].some(field=>data.get(field)!==time(moment[field]))) {
        for(const field of ['start','end']) changes[field]=parse(data.get(field));
      }
      for(const field of ['title','selected_hook']) if(data.has(field) && data.get(field)!==moment[field]) changes[field]=data.get(field);
      for(const field of ['captions','voiceover','mode','layout','framing']) {
        if(!data.has(field)) continue;
        const value=data.get(field);
        const fallback=field==='layout'?(job.request.tracking?.layout || 'auto'):field==='framing'?(job.request.tracking?.x ?? null):
          field==='voiceover'?(job.request.director?.voiceover ?? false):field==='mode'?(job.request.mode || 'crop'):(job.request.captions ?? true);
        const old=Object.hasOwn(moment.render_options || {},field)?moment.render_options[field]:fallback;
        const next=field==='framing'?(value===''?null:Number(value)/100):field==='captions'||field==='voiceover'?value==='true':value;
        if(next!==old) changes[field]=next;
      }
      await request(`/${node.dataset.moment}`,'PATCH',changes);
      drafts.delete(node.dataset.moment);
    });
  });
  panel.addEventListener('click',event=>{
    const button=event.target.closest('button,a');
    if(!button)return;
    if(button.dataset.filter){filter=button.dataset.filter;draw(job,true);return;}
    const kind=button.dataset.momentAction;
    if(!kind)return;
    const node=button.closest('[data-moment]');
    const m=job.moments.find(m=>m.id===node?.dataset.moment);
    if(kind==='highlight') {
      event.preventDefault();document.getElementById(`clip-${m.clip_id}`)?.scrollIntoView({behavior:'smooth',block:'start'});return;
    }
    if(kind==='preview') {
      const close=active===m.id;
      panel.querySelectorAll('.moment-player > *').forEach(frame=>frame.remove());
      active=close?null:m.id;
      panel.querySelectorAll('.moment-preview').forEach(b=>b.textContent='▶ Preview');
      if(close)return;
      if(job.upload) {
        const video=document.createElement('video');
        video.src=`./api/jobs/${job.id}/source#t=${m.start},${m.end}`;video.controls=video.autoplay=video.playsInline=true;
        video.addEventListener('timeupdate',()=>{if(video.currentTime>=m.end)video.pause();});
        node.querySelector('.moment-player').append(video);button.textContent='■ Close preview';return;
      }
      const url=previewUrl(m);
      if(!url){active=null;return;}
      const frame=document.createElement('iframe');
      frame.src=url;
      frame.title=`Preview: ${m.title}`;frame.allow='autoplay; encrypted-media; picture-in-picture';frame.allowFullscreen=true;
      node.querySelector('.moment-player').append(frame);button.textContent='■ Close preview';return;
    }
    action(node,async()=>{
      if(kind==='batch' || kind==='render')await request('/render','POST',kind==='render'?{ids:[m.id]}:{});
      else if(kind==='top') {
        const top=ordered().filter(m=>visible(m)&&matches(m,filter)&&!['rendered','rejected'].includes(m.status)).slice(0,5);
        for(const item of top)await request(`/${item.id}`,'PATCH',{status:'selected'});
      } else if(kind==='ai' || kind==='chat') {
        await request(`/${m.id}`,'PATCH',m.alternatives[kind]);drafts.delete(m.id);
      } else await request(`/${m.id}`,'PATCH',{status:kind==='select'?(m.status==='selected'?'suggested':'selected'):(m.status==='rejected'?'suggested':'rejected')});
    });
  });
  return {draw,reset(){job=null;key='';filter='all';active=null;showHidden=false;drafts.clear();errors.clear();panel.innerHTML='';panel.hidden=true;}};
})();
