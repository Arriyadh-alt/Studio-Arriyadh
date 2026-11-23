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
  const matches = (m, filter) => filter==='all' || (filter==='both' ? m.sources.length>1 : m.sources.includes(filter));
  let job, filter='all', key='', active=null, pending=false;
  const drafts = new Map();
  const errors = new Map();
  const source = m => m.sources.length>1?'AI + Chat':m.sources.includes('ai')?'AI Analysis':'Chat Analysis';
  const range = m => `${time(m.start)}–${time(m.end)}`;
  const previewUrl = m => {
    const url=new URL(job.url);
    const id=url.hostname==='youtu.be'?url.pathname.slice(1):url.searchParams.get('v') || url.pathname.split('/').pop();
    if(!/^[\w-]{11}$/.test(id))return null;
    return `https://www.youtube-nocookie.com/embed/${id}?start=${Math.floor(m.start)}&end=${Math.ceil(m.end)}&autoplay=1&playsinline=1`;
  };
  function card(m) {
    const draft=drafts.get(m.id) || {start:time(m.start),end:time(m.end)};
    const clip=job.clips.find(c=>c.id===m.clip_id);
    const rendered=m.status==='rendered';
    return `<article class="moment-card ${m.status==='rejected'?'is-rejected':''}" data-moment="${esc(m.id)}">
      <div class="moment-meta"><strong class="moment-rank">#${m.rank}</strong><span>${range(m)} · ${Math.round(m.duration)}s</span><span class="moment-source">${source(m)}</span></div>
      <h3>${esc(m.title || 'Stream reaction')}</h3><p class="moment-description">${esc(m.description)}</p>
      <div class="potential"><span>Viral Potential</span><strong>${Math.round(m.score)}<small> / 100</small></strong></div><progress max="100" value="${m.score}" aria-label="Viral Potential ${m.score} out of 100"></progress>
      <p class="moment-reason">${esc(m.reason)}</p>
      ${m.alternatives?`<details class="moment-alternatives"><summary>Compare AI &amp; chat ranges</summary>${['ai','chat'].map(s=>`<div><span>${s==='ai'?'AI':'Chat'}: ${range(m.alternatives[s])}</span><button type="button" data-moment-action="${s}" ${rendered?'disabled':''}>Apply ${s==='ai'?'AI':'chat'}</button></div>`).join('')}</details>`:''}
      <div class="moment-player"></div><button type="button" class="moment-preview" data-moment-action="preview">${active===m.id?'■ Close preview':'▶ Preview'}</button>
      ${rendered?`<a class="moment-rendered" href="#clip-${esc(m.clip_id)}" data-moment-action="highlight">Rendered → highlight #${clip?.number ?? job.clips.indexOf(clip)+1}</a>`:`<form class="moment-times"><label>Start<input name="start" aria-label="Moment ${m.rank} start" value="${esc(draft.start)}" inputmode="text"></label><label>End<input name="end" aria-label="Moment ${m.rank} end" value="${esc(draft.end)}" inputmode="text"></label><button type="submit">Save times</button></form><div class="moment-actions"><button type="button" data-moment-action="select" aria-pressed="${m.status==='selected'}">${m.status==='selected'?'✓ Selected':'Select'}</button><button type="button" data-moment-action="reject" aria-pressed="${m.status==='rejected'}">${m.status==='rejected'?'↶ Undo reject':'Reject'}</button><button type="button" class="primary" data-moment-action="render">Render ↗</button></div>`}
      <p class="error moment-error" role="status" ${errors.has(m.id)?'':'hidden'}>${esc(errors.get(m.id))}</p></article>`;
  }
  function draw(value, force=false) {
    job=value;
    panel.hidden=!job.moments?.length;
    if(panel.hidden) {panel.innerHTML='';key='';active=null;return;}
    const next=JSON.stringify([job.id,job.moments,job.chat_error,job.ai_error,filter]);
    if(!force && next===key) return;
    key=next;
    const iframe=panel.querySelector('iframe');
    const selected=job.moments.filter(m=>m.status==='selected').length;
    panel.innerHTML=`<div class="section-heading"><div><span class="eyebrow">Find the standout</span><h2>Detected moments <span class="hint">· ${job.moments.length}</span></h2></div></div>
      ${job.chat_error?`<p class="moment-warning">${job.moments.some(m=>m.sources.includes('chat'))?'Chat replay partially unavailable — ':job.moments.some(m=>m.sources.includes('ai'))?'Chat replay unavailable — AI analysis only. ':'Chat replay unavailable — '}${esc(job.chat_error)}</p>`:''}
      ${job.ai_error?`<p class="moment-warning">AI analysis unavailable — chat analysis only. ${esc(job.ai_error)}</p>`:''}
      <div class="moment-toolbar"><button class="primary" type="button" data-moment-action="batch" ${!selected?'disabled':''}>Render selected (${selected}) ↗</button><button type="button" data-moment-action="top">Select top 5</button></div>
      <p class="hint">Rendered with AI Director: hook, voice-over, facecam on top, captions, zooms, loop — according to this job’s settings.</p>
      <p class="hint">Estimated content-quality score, not a guarantee of views</p>
      <div class="moment-filters" role="group" aria-label="Filter moments">${Object.entries(filters).map(([id,label])=>`<button type="button" data-filter="${id}" aria-pressed="${filter===id}">${label} <span>${job.moments.filter(m=>matches(m,id)).length}</span></button>`).join('')}</div>
      <p class="error moment-batch-error" role="status" hidden></p><div class="moment-grid">${[...job.moments].sort((a,b)=>a.rank-b.rank).filter(m=>matches(m,filter)).map(card).join('') || '<p class="empty">No moments in this filter.</p>'}</div>`;
    if(iframe && active) {
      const target=[...panel.querySelectorAll('[data-moment]')].find(n=>n.dataset.moment===active);
      if(target) {
        const moment=job.moments.find(m=>m.id===active);
        const url=previewUrl(moment);
        if(iframe.getAttribute('src')!==url)iframe.src=url;
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
    if(!node || !event.target.matches('input'))return;
    drafts.set(node.dataset.moment,Object.fromEntries(new FormData(node.querySelector('form'))));
  });
  panel.addEventListener('submit',event=>{
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
      panel.querySelectorAll('iframe').forEach(frame=>frame.remove());
      active=close?null:m.id;
      panel.querySelectorAll('.moment-preview').forEach(b=>b.textContent='▶ Preview');
      if(close)return;
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
        const top=[...job.moments].sort((a,b)=>a.rank-b.rank).filter(m=>!['rendered','rejected'].includes(m.status)).slice(0,5);
        for(const item of top)await request(`/${item.id}`,'PATCH',{status:'selected'});
      } else if(kind==='ai' || kind==='chat') {
        await request(`/${m.id}`,'PATCH',m.alternatives[kind]);drafts.delete(m.id);
      } else await request(`/${m.id}`,'PATCH',{status:kind==='select'?(m.status==='selected'?'suggested':'selected'):(m.status==='rejected'?'suggested':'rejected')});
    });
  });
  return {draw,reset(){job=null;key='';filter='all';active=null;drafts.clear();errors.clear();panel.innerHTML='';panel.hidden=true;}};
})();
