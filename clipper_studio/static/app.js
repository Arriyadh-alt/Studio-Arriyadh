const $ = (selector, root = document) => root.querySelector(selector);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const busy = job => ['queued', 'running'].includes(job.status);
let current = null;
let timer;
let poll = null;
const drafts = new Map();
const keeping = new Map();
let chartKey;
const locked = clip => ['queued','rendering'].includes(clip.status);
const queueLabel = position => position ? `Queued (position ${position})` : 'Queued';
const clipVersion = clip => `${clip.status}:${clip.start}:${clip.end}:${clip.mode}:${clip.captions}:${clip.attempt || 0}:${clip.queue_position || 0}:${JSON.stringify([clip.language,clip.caption_style,clip.caption_language])}`;
$('#create-caption-style').innerHTML = CaptionControls.html();
CaptionControls.preview($('#create-caption-style'));

function timestamp(value) {
  const total = Math.max(0, Math.round(value));
  const h = Math.floor(total / 3600), m = Math.floor(total % 3600 / 60), s = total % 60;
  return `${h ? h + ':' : ''}${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}
function parseTime(value) {
  const text = value.trim();
  if (!/^\d+(?::[0-5]\d){1,2}(?:\.\d+)?$/.test(text)) throw Error('Use MM:SS or HH:MM:SS for clip times.');
  return text.split(':').reduce((sum, part) => sum * 60 + Number(part), 0);
}
function notice(message, selector = '#notice') {
  $(selector).textContent = message;
  $(selector).hidden = !message;
}
async function api(url, method = 'GET', data, signal) {
  const response = await fetch(url, {method, cache:'no-store', signal, headers: data ? {'Content-Type':'application/json'} : {}, body: data ? JSON.stringify(data) : undefined});
  const body = await response.json().catch(() => null);
  if (!response.ok || body === null) {
    const detail = body?.detail;
    throw Error(Array.isArray(detail) ? detail.map(item => item.msg.replace(/^Value error, /,'')).join(' ') : typeof detail === 'string' ? detail : 'Studio didn’t respond. Please try again.');
  }
  return body;
}
function badge(status, position) { return `<span class="badge ${escapeHtml(status)}">${escapeHtml(status==='queued' ? queueLabel(position) : status.replaceAll('_', ' '))}</span>`; }
function safeImage(url) {
  try { const parsed = new URL(url); return parsed.protocol === 'https:' ? escapeHtml(parsed.href) : ''; } catch { return ''; }
}
function date(job) {
  return job.created_at ? new Date(job.created_at).toLocaleDateString(undefined, {month:'short',day:'numeric',year:'numeric'}) : 'Imported stream';
}
function jobTitle(job) {
  if (job.metadata?.title) return job.metadata.title;
  try {
    const url = new URL(job.url);
    const id = url.searchParams.get('v') || url.pathname.split('/').filter(Boolean).pop();
    if (/^[a-zA-Z0-9_-]{11}$/.test(id)) return `YouTube video ${id}`;
  } catch {}
  return 'Untitled stream';
}
function drawHome(jobs) {
  jobs.sort((a,b) => (b.created_at || '').localeCompare(a.created_at || ''));
  $('#jobs').innerHTML = jobs.length ? jobs.map(job => `<a href="#jobs/${job.id}" class="job-card">
    ${safeImage(job.metadata?.thumbnail) ? `<img src="${safeImage(job.metadata.thumbnail)}" alt="" loading="lazy">` : ''}
    <div class="body">${badge(job.status, job.queue_position)}<h3>${escapeHtml(jobTitle(job))}</h3><div class="card-meta"><span>${date(job)}</span><span>${job.clip_count ?? job.clips?.length ?? 0} clips ↗</span></div></div></a>`).join('') : '<p class="empty">Your next great moment starts here. Paste a stream above.</p>';
}
function chart(job) {
  const containerWidth = $('#chart').clientWidth;
  const key = JSON.stringify([containerWidth,job.id,job.activity,job.clips.map((c,i)=>[c.id,c.peak,c.number ?? i+1])]);
  if (key===chartKey)return;
  chartKey=key;
  const data = job.activity;
  if (!data.length) { $('#chart').innerHTML = '<p class="hint">Chat activity will appear after the replay is analyzed.</p>'; return; }
  const width = Math.max(260, containerWidth), height = 125, duration = job.metadata.duration;
  const max = Math.max(1, ...data.map(p => p.count));
  // Group only for display; the original five-second data remains in the API.
  const stride = Math.max(1, Math.ceil(data.length / 250));
  let bars = '';
  for (let i=0; i<data.length; i+=stride) {
    const count = Math.max(...data.slice(i,i+stride).map(p => p.count));
    const h = Math.max(1, count / max * 100);
    bars += `<rect x="${data[i].time / duration * width}" y="${height-h}" width="${Math.max(1,stride*5/duration*width-1)}" height="${h}" rx="1" fill="#64774e"/>`;
  }
  const markerScale = containerWidth / (width+44) * 138 / 150;
  const marks = job.clips.map((clip,index) => `<g class="chart-mark" tabindex="0" role="button" aria-label="Review highlight ${clip.number ?? index+1} at ${timestamp(clip.peak)}" data-clip="${clip.id}"><title>Highlight ${clip.number ?? index+1} · ${timestamp(clip.peak)}</title><rect x="${clip.peak/duration*width-26}" y="0" width="52" height="138" fill="transparent"/><line x1="${clip.peak/duration*width}" x2="${clip.peak/duration*width}" y1="17" y2="125" stroke="#b8f477" stroke-dasharray="3 4"/><g transform="translate(0 13) scale(1 ${markerScale}) translate(0 -13)"><circle cx="${clip.peak/duration*width}" cy="13" r="11" fill="#b8f477"/><text x="${clip.peak/duration*width}" y="17" text-anchor="middle" fill="#172210" font-size="11" font-weight="bold">${clip.number ?? index+1}</text></g></g>`).join('');
  $('#chart').innerHTML = `<svg viewBox="-22 0 ${width+44} 138" preserveAspectRatio="none" aria-label="Chat activity over time">${bars}${marks}</svg><div class="axis"><span>00:00</span><span>${timestamp(duration/2)}</span><span>${timestamp(duration)}</span></div>`;
}
function clipHtml(clip,index,job) {
  const draft = drafts.get(clip.id) || {start:timestamp(clip.start),end:timestamp(clip.end),mode:clip.mode,captions:clip.captions,...CaptionControls.options(clip)};
  const disabled = locked(clip) ? 'disabled' : '';
  const number = clip.number ?? index+1;
  const timeField = field => `<label>${field === 'start' ? 'Start' : 'End'}<input class="time" data-field="${field}" value="${escapeHtml(draft[field])}" aria-label="${field} time" inputmode="text" autocomplete="off"></label><div class="nudges">${[-5,-1,1,5].map(n=>`<button type="button" data-nudge="${n}" data-field="${field}" aria-label="${field} ${n>0?'plus':'minus'} ${Math.abs(n)} seconds">${n>0?'+':''}${n}s</button>`).join('')}</div>`;
  return `<article id="clip-${clip.id}" class="clip" data-id="${clip.id}" data-version="${escapeHtml(clipVersion(clip))}">
    <div class="clip-top"><div><h3>Highlight ${String(number).padStart(2,'0')}</h3>${badge(clip.status, clip.queue_position)}</div><button class="keep" aria-label="Keep highlight ${number}" aria-pressed="${!!clip.kept}" data-action="keep">${clip.kept?'★ Kept':'☆ Keep'}</button></div>
    <div class="preview">${clip.status==='ready' ? `<video controls playsinline preload="metadata" poster=".${clip.thumbnail_url}"><source src=".${clip.file_url}?v=${clip.rendered_at || 0}-${clip.start}-${clip.end}-${clip.mode}-${clip.captions}" type="video/mp4"></video>` : `<div class="placeholder"><span style="font-size:30px">${clip.status==='failed'?'!':'✂'}</span><span>${clip.status==='failed'?'This clip needs another try':clip.status==='cancelled'?'Cancelled — re-render to try again':clip.status==='queued'?escapeHtml(queueLabel(clip.queue_position)):`Creating your highlight…<br>Attempt ${clip.attempt || 1} of 3`}</span></div>`}</div>
    <div class="clip-body"><div class="clip-meta"><span class="score">↗ ${clip.score.toFixed(2)} spike score</span><span>${timestamp(clip.end-clip.start)} clip</span></div><p class="hint">Onset ${timestamp(clip.onset)} · Peak ${timestamp(clip.peak)}</p>
    <div class="samples">${clip.message_sample.length ? clip.message_sample.map(m=>`<span class="sample">${escapeHtml(m)}</span>`).join('') : 'No sample messages'}</div>
    ${clip.error ? `<p class="error">${escapeHtml(clip.error)}${clip.status==='failed'?'<br>Adjust the settings below or try Re-render again.':''}</p>` : ''}
    <form class="review"><div class="time-grid"><div>${timeField('start')}</div><div>${timeField('end')}</div></div>
      <div class="render-options"><label>Layout<select data-field="mode"><option value="crop" ${draft.mode==='crop'?'selected':''}>Center crop</option><option value="blur" ${draft.mode==='blur'?'selected':''}>Blurred background</option></select></label><label class="check"><input type="checkbox" data-field="captions" ${draft.captions?'checked':''}> Captions</label></div>
      ${CaptionControls.html(draft)}
      ${clip.caption_language?`<p class="hint">Transcription language: ${escapeHtml(clip.caption_language)}</p>`:''}
      <p class="draft-note hint" hidden><span>Changes not applied — Re-render to update</span> <button type="button" class="quiet" data-action="reset">Reset</button></p><p class="draft-error error" role="status" hidden></p>
      ${locked(clip)?'<p class="hint">This highlight is queued or rendering. Review actions unlock when it finishes.</p>':''}
      <div class="actions"><button type="submit" class="primary" ${disabled}>Re-render ↻</button>${clip.status==='ready'?`<a class="button" href=".${clip.download_url}">Download ↓</a>`:''}<button class="delete" type="button" data-action="delete" aria-label="Delete highlight ${number}" ${disabled}>Delete</button></div>
    </form></div></article>`;
}
function drawJob(job) {
  job.clips.forEach(c=>{if(keeping.has(c.id))c.kept=keeping.get(c.id);});
  current = job;
  const image = safeImage(job.metadata?.thumbnail);
  $('#job-heading').innerHTML = `<div class="job-header">${image?`<img src="${image}" alt="Stream thumbnail">`:''}<div>${badge(job.status, job.queue_position)}<h1>${escapeHtml(jobTitle(job))}</h1><p>${date(job)} ${job.metadata?`· ${timestamp(job.metadata.duration)} stream`:''} · ${escapeHtml(job.request.mode)} layout</p></div></div>`;
  $('#stage').textContent = job.stage;
  $('#progress').value = job.progress;
  $('#progress-label').textContent = `${Math.round(job.progress*100)}%`;
  const cancellable = busy(job) && 'queue_position' in job;
  $('#cancel-job').hidden = !cancellable;
  $('#cancel-hint').hidden = !cancellable;
  $('#job-error').hidden = !job.error;
  $('#job-error').textContent = job.error || '';
  chart(job);
  $('#clips-title').textContent = `Your highlights${job.clips.length ? ' · '+job.clips.length : ''}`;
  const kept = job.clips.filter(c=>c.kept && c.status==='ready').length;
  $('#download-kept').hidden = !kept;
  $('#download-kept').href = `./api/jobs/${job.id}/download-kept`;
  $('#download-kept').textContent = `Download kept clips (${kept}) ↓`;
  $('#keep-hint').hidden = kept>0 || !job.clips.some(c=>c.status==='ready');
  const unkept=job.clips.filter(c=>!c.kept && c.status!=='rendering');
  $('#delete-unkept').hidden = !job.clips.some(c=>c.kept) || !unkept.length;
  $('#delete-unkept').textContent = `Delete ${unkept.length} unkept`;
  const container = $('#clips');
  container.querySelector('.empty')?.remove();
  const ids = new Set(job.clips.map(c=>c.id));
  container.querySelectorAll('.clip').forEach(node=>{if(!ids.has(node.dataset.id)) node.remove();});
  job.clips.forEach((clip,index)=>{
    const existing = document.getElementById(`clip-${clip.id}`);
    const html = clipHtml(clip,index,job);
    const version = clipVersion(clip);
    if (!existing) container.insertAdjacentHTML('beforeend',html);
    else if (existing.dataset.version !== version) existing.outerHTML = html;
    const node = document.getElementById(`clip-${clip.id}`);
    CaptionControls.preview(node);
    const keep = $('[data-action=keep]',node);
    keep.setAttribute('aria-pressed',String(!!clip.kept));
    keep.textContent = clip.kept ? '★ Kept' : '☆ Keep';
    node.querySelectorAll('[type=submit],[data-action=delete]').forEach(button=>button.disabled=locked(clip));
    draftFeedback(node);
  });
  if (!job.clips.length) container.innerHTML = `<p class="empty">${busy(job)?'Your clips will appear here as they finish.':job.status==='cancelled'?'This job was cancelled. Return to All streams to start another.':job.error?'The stream could not be processed. Return to All streams to try another URL.':'No significant chat spikes found. Try a livelier replay.'}</p>`;
}
async function refresh() {
  if (poll) return;
  const request = new AbortController();
  poll = request;
  const timeout = setTimeout(() => request.abort(), 10000);
  const hash = location.hash;
  const match = hash.match(/^#jobs\/([a-zA-Z0-9]+)$/);
  try {
    const result = await api(match ? `./api/jobs/${match[1]}` : './api/jobs?summary=true', 'GET', undefined, request.signal);
    if (poll !== request || location.hash !== hash) return;
    if (match) drawJob(result); else drawHome(result);
    notice('', '#poll-notice');
  } catch (error) {
    if (poll === request && location.hash === hash) {
      const message = error.message === 'Job not found' ? error.message : 'Couldn’t refresh status. Retrying…';
      notice(message, '#poll-notice');
    }
  } finally {
    clearTimeout(timeout);
    if (poll === request) poll = null;
  }
}
function route() {
  poll?.abort(); poll=null;
  notice(''); notice('', '#poll-notice'); current=null; drafts.clear();
  const jobPage = /^#jobs\//.test(location.hash);
  $('#home').hidden=jobPage; $('#job-view').hidden=!jobPage;
  $('#clips').innerHTML=''; $('#job-heading').innerHTML='';
  $('#stage').textContent='Loading job…'; $('#progress-label').textContent='';
  $('#cancel-job').hidden=true; $('#cancel-hint').hidden=true;
  $('#progress').removeAttribute('value'); $('#job-error').hidden=true;
  chartKey=undefined; $('#chart').innerHTML='';
  $('#download-kept').hidden=true; $('#delete-unkept').hidden=true; $('#keep-hint').hidden=true;
  window.scrollTo(0,0);
  clearInterval(timer); refresh(); timer=setInterval(refresh,2000);
}
$('#lead').addEventListener('input',event=>$('#lead-value').textContent=`${event.target.value} seconds`);
$('#refresh-jobs').addEventListener('click',()=>refresh());
$('#start-form').addEventListener('submit',async event=>{
  event.preventDefault(); notice(''); const button=$('#start-button'); button.disabled=true;
  try {
    const job=await api('./api/jobs','POST',{url:$('#url').value.trim(),clips:Number($('#clip-count').value),lead:Number($('#lead').value),mode:$('#mode').value,captions:$('#captions').checked,...CaptionControls.read($('#start-form'))});
    location.hash=`jobs/${job.id}`;
  } catch(error) {
    notice(error instanceof TypeError ? 'Couldn’t confirm the submission. Check Recent streams before trying again.' : error.message);
  } finally { button.disabled=false; }
});
function saveDraft(node) {
  drafts.set(node.dataset.id,{start:$('[data-field=start].time',node).value,end:$('[data-field=end].time',node).value,mode:$('[data-field=mode]',node).value,captions:$('[data-field=captions]',node).checked,...CaptionControls.read(node)});
  draftFeedback(node);
}
function draftFeedback(node) {
  const clip=current.clips.find(c=>c.id===node.dataset.id), draft=drafts.get(clip.id);
  let error='',changed=false,length=clip.end-clip.start;
  if(draft) {
    changed=draft.start!==timestamp(clip.start)||draft.end!==timestamp(clip.end)||draft.mode!==clip.mode||draft.captions!==clip.captions;
    changed=changed||JSON.stringify({language:draft.language,caption_style:draft.caption_style})!==JSON.stringify(CaptionControls.options(clip));
    try {
      const start=parseTime(draft.start),end=parseTime(draft.end); length=end-start;
      if(end<=start || length>60 || end>current.metadata.duration)throw Error('Choose a window within the stream, with end after start and a maximum of 60 seconds.');
    } catch(e) {error=e.message;length=null;}
  }
  $('.clip-meta span:last-child',node).textContent=length===null?'Invalid clip length':`${timestamp(length)} clip`;
  $('.draft-note',node).hidden=!changed;
  $('.draft-error',node).hidden=!error;
  $('.draft-error',node).textContent=error;
  $('[type=submit]',node).disabled=locked(clip)||!!error;
}
$('#cancel-job').addEventListener('click',async event=>{
  if(!current || !busy(current) || !confirm('Cancel this job? Ready clips will be kept.'))return;
  const jid=current.id;
  event.target.disabled=true;
  try {notice('');await api(`./api/jobs/${jid}/cancel`,'POST');await refresh();}
  catch(error){notice(error.message);}finally{event.target.disabled=false;}
});
$('#delete-unkept').addEventListener('click',async event=>{
  const clips=current.clips.filter(c=>!c.kept && c.status!=='rendering');
  if(!confirm(`Delete ${clips.length} unkept ${clips.length===1?'highlight and its files':'highlights and their files'}?`))return;
  event.target.disabled=true;
  try {for(const clip of clips){await api(`./api/jobs/${current.id}/clips/${clip.id}`,'DELETE');drafts.delete(clip.id);}}
  catch(error){notice(error.message);}finally{event.target.disabled=false;await refresh();}
});
$('#clips').addEventListener('input',event=>{const node=event.target.closest('.clip');if(node)saveDraft(node);});
$('#clips').addEventListener('change',event=>{const node=event.target.closest('.clip');if(node)saveDraft(node);});
$('#clips').addEventListener('click',async event=>{
  const button=event.target.closest('button'); if(!button || !current)return;
  const node=button.closest('.clip'), cid=node.dataset.id;
  const base=`./api/jobs/${current.id}/clips/${cid}`;
  try {
    notice('');
    if(button.dataset.nudge) {
      const input=$(`input[data-field=${button.dataset.field}]`,node);
      input.value=timestamp(Math.min(current.metadata.duration,Math.max(0,parseTime(input.value)+Number(button.dataset.nudge)))); saveDraft(node);
      const video = $('video',node), clip = current.clips.find(clip=>clip.id===cid);
      if (video && clip && Number.isFinite(video.duration)) {
        const offset = parseTime(input.value) - clip.start - (button.dataset.field==='end' ? 2 : 0);
        video.pause();
        video.currentTime = Math.min(video.duration,Math.max(0,offset));
      }
    } else if(button.dataset.action==='keep') {
      const clip=current.clips.find(c=>c.id===cid), previous=!!clip.kept, kept=!previous;
      keeping.set(cid,kept); clip.kept=kept; button.disabled=true; drawJob(current);
      try {await api(base,'PATCH',{kept});}catch(error){const latest=current.clips.find(c=>c.id===cid);if(latest)latest.kept=previous;throw error;}
      finally {keeping.delete(cid);drawJob(current);}
      await refresh();
    } else if(button.dataset.action==='reset') {
      drafts.delete(cid); node.outerHTML=clipHtml(current.clips.find(c=>c.id===cid),current.clips.findIndex(c=>c.id===cid),current);
    } else if(button.dataset.action==='delete' && confirm('Delete this clip and its files?')) {
      button.disabled=true; await api(base,'DELETE'); drafts.delete(cid); await refresh();
    }
  } catch(error) { notice(error.message); } finally { if(button.dataset.action==='keep')button.disabled=false; }
});
$('#clips').addEventListener('submit',async event=>{
  event.preventDefault(); if(!current)return;
  const node=event.target.closest('.clip'), cid=node.dataset.id; saveDraft(node);
  const draft=drafts.get(cid), button=$('[type=submit]',node);
  try {
    notice(''); button.disabled=true;
    const start=parseTime(draft.start),end=parseTime(draft.end);
    if(end<=start || end-start>60 || end>current.metadata.duration)throw Error('Choose a window within the stream, with end after start and a maximum of 60 seconds.');
    await api(`./api/jobs/${current.id}/clips/${cid}/render`,'POST',{start,end,mode:draft.mode,captions:draft.captions,language:draft.language,caption_style:draft.caption_style}); drafts.delete(cid); await refresh();
  } catch(error) { notice(error.message); button.disabled=false; }
});
function jump(event) {const mark=event.target.closest('[data-clip]');if(mark)document.getElementById(`clip-${mark.dataset.clip}`)?.scrollIntoView({behavior:'smooth',block:'start'});}
$('#chart').addEventListener('click',jump);
$('#chart').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();jump(event);}});
new ResizeObserver(()=>{if(current && !$('#job-view').hidden)chart(current);}).observe($('#chart'));
window.addEventListener('hashchange',route);
route();
