const $ = (selector, root = document) => root.querySelector(selector);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const busy = job => ['queued', 'running'].includes(job.status);
const canCancel = job => busy(job) || job.chat_pending || job.chat_retrying;
let current = null;
let timer;
let poll = null;
const drafts = new Map();
const keeping = new Map();
let chartKey;
const locked = clip => ['queued','rendering'].includes(clip.status);
const queueLabel = position => position ? `Queued (position ${position})` : 'Queued';
const clipVersion = clip => `${clip.status}:${clip.start}:${clip.end}:${clip.mode}:${clip.captions}:${clip.attempt || 0}:${clip.queue_position || 0}:${JSON.stringify([current?.rights?.confirmed,clip.podcast,clip.facecam,clip.facecam_result,clip.source_still_url,clip.rendered_at,clip.language,clip.caption_style,clip.caption_language,clip.editing_mode,clip.authenticity,clip.director,clip.director_result,clip.tracking,clip.tracking_result,clip.tracking_still_url,clip.needs_review])}`;
const facecamDefaults = value => ({enabled:true,position:'auto',size:.42,box:null,...value});
const renderedOffset = (clip, offset) => {
  const timeline=clip.director_result?.edits?.timeline;
  if(!timeline?.length) return offset;
  const shot=timeline.find(([,end])=>offset<end) || timeline[timeline.length-1];
  return shot[2] + Math.min(shot[1]-shot[0], Math.max(0, offset-shot[0]));
};
const editingModes = {
  full:['Full AI Director Mode','The original AI Director: storytelling openers, stronger effects and seamless loops.'],
  natural:['Natural Editor Mode','Original footage and sound, adaptive pacing, subtle reframing and text hooks. AI voice-over is off by default.'],
  enhanced:['AI Enhanced Mode','Keep events in order, with tighter pacing, stronger emphasis and optional music, effects or narration.']
};
const modeOptions = selected => Object.entries(editingModes).map(([value,[label]])=>`<option value="${value}" ${selected===value?'selected':''}>${label}</option>`).join('');
const editingMode = clip => clip.editing_mode || 'full';
const directorDefaults = (value, mode='full') => ({enabled:true,hook:true,edits:true,sfx:mode==='full',music:mode==='full'?'auto':'off',story:true,voiceover:false,hook_text:null,voice:'auto',delivery:'auto',loop:mode==='full',...value});
const directorKeys = ['enabled','hook','edits','sfx','music','story','voiceover','hook_text','voice','delivery','loop'];
const readDirector = node => Object.fromEntries(directorKeys.map(k=>{
  const el=$(`[data-director=${k}]`,node);
  return [k,k==='hook_text'?(el.value.trim()||null):el.type==='checkbox'?el.checked:el.value];
}));
function syncPodcastMode(root, mode, selector) {
  const version=$(selector,root);
  if(!version) return;
  version.disabled=mode==='natural';
  const note=$('[data-commentary-note]',version.parentElement);
  note.hidden=mode!=='natural';
}
let voiceCatalog = {voices:[],deliveries:['excited','surprise','suspense','humor','warm']};
const deliveryLabels = {excited:'Excited',surprise:'Surprised',suspense:'Suspense',humor:'Funny',warm:'Warm'};
const voiceOptions = selected => `<option value="auto">Auto (match the scene)</option>${voiceCatalog.voices.map(v=>`<option value="${escapeHtml(v.id)}" title="${escapeHtml(v.description)}" ${selected===v.id?'selected':''}>${escapeHtml(v.name)}</option>`).join('')}`;
const deliveryOptions = selected => `<option value="auto">Auto (from the scene)</option>${voiceCatalog.deliveries.map(m=>`<option value="${m}" ${selected===m?'selected':''}>${deliveryLabels[m]||m}</option>`).join('')}`;
const speechLanguage = (root, hook) => {
  const chosen=$('[data-caption=language]',root)?.value;
  if(chosen && chosen!=='auto') return ['id','ms','jw','su'].includes(chosen) ? 'id' : 'en';
  return ['id','ms','jw','su'].includes(hook?.language) ? 'id' : 'en';
};
async function previewVoice(body, button, status, narrationUrl) {
  const audio=new Audio();
  audio.play().catch(()=>{}); // unlocks playback inside the tap, before the fetch (iOS Safari)
  button.disabled=true; status.textContent='Generating preview…';
  try {
    const response=narrationUrl?await fetch(`.${narrationUrl}`):await fetch('./api/voices/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!response.ok) throw new Error((await response.json().catch(()=>({}))).detail || 'Voice preview failed');
    audio.src=URL.createObjectURL(await response.blob());
    status.textContent=`${response.headers.get('X-Voice-Name')} · ${deliveryLabels[response.headers.get('X-Voice-Mood')]||''}${response.headers.get('X-Voice-Engine')==='offline'?' · offline voice':''}`;
    audio.controls=true; audio.className='voice-audio';
    status.append(audio);
    await audio.play().catch(()=>{});
  } catch(error) { status.textContent=error.message; }
  finally { button.disabled=false; }
}
const directorLabels = {hook:'Hook',pacing:'Pacing',emotion:'Emotion',dialogue:'Dialogue',chat:'Chat'};
function directorHtml(clip, draft) {
  const mode=draft.editing_mode, d=directorDefaults(draft.director,mode), r=clip.director_result;
  const restricted=mode==='natural'?'disabled':'';
  const edits=r?.edits, loop=clip.status==='ready' ? edits?.loop : null;
  const report=r?.status==='used' ? `<div class="director-report">
    <p class="director-score"><strong>${r.score.overall}</strong><span>viral score</span></p>
    <ul class="director-bars">${Object.entries(directorLabels).map(([k,l])=>`<li><span>${l}</span><meter min="0" max="100" value="${r.score[k]}"></meter><span>${r.score[k]}</span></li>`).join('')}</ul>
    <p class="hint">${[edits.hook?.style==='story'?`${edits.hook.duration.toFixed(1)} s story hook · ${edits.hook.build_up.toFixed(1)} s build-up · ${edits.hook.payoff.toFixed(1)} s payoff & reaction`:edits.hook?.style==='source'?`${edits.hook.duration.toFixed(1)} s text hook`:edits.hook?`${edits.hook.duration.toFixed(1)} s hook from ${timestamp(edits.hook.source_start)}`:'No teaser hook',`${edits.cuts} cuts`,`${edits.removed_seconds.toFixed(1)} s trimmed`,`${edits.zooms} zooms`,`${edits.sound_effects} sound effects`,edits.music?'music bed':''].filter(Boolean).map(escapeHtml).join(' · ')}</p>
    ${edits.beats?.length?`<p class="hint">Story: ${edits.beats.map(b=>`${escapeHtml(b.name)} ${timestamp(b.start)}`).join(' → ')}</p>`:''}
    <ul class="quality">${r.quality.map(q=>`<li class="${escapeHtml(q.status)}"><strong>${escapeHtml(q.check)}</strong> ${escapeHtml(q.detail)}</li>`).join('')}</ul>
    ${loop?.status==='applied'?'<p class="hint">Seamless loop applied. Play the preview to hear and see the replay.</p>':loop?.reason?`<p class="hint">Seamless loop skipped: ${escapeHtml(loop.reason)}.</p>`:''}
    ${r.suggestions.length?`<ul class="suggestions">${r.suggestions.map(t=>`<li>${escapeHtml(t)}</li>`).join('')}</ul>`:''}</div>`
    : `<p class="hint">${r?.status==='unavailable'?'AI Director couldn’t edit this clip; it was rendered unedited.':d.enabled?'Re-render to let AI Director edit this clip.':'AI Director off'}</p>`;
  return `<fieldset class="director-options" ${locked(clip)?'disabled':''}><legend>AI Director</legend>
    <label>Editing mode<select data-field="editing_mode">${modeOptions(mode)}</select><small class="hint" data-editing-hint>${editingModes[mode][1]}</small></label>
    <label class="check"><input type="checkbox" data-director="enabled" ${d.enabled?'checked':''}> Enabled</label>
    <label class="check"><input type="checkbox" data-director="hook" ${d.hook?'checked':''}> Hook opener</label>
    <label class="check"><input type="checkbox" data-director="edits" ${d.edits?'checked':''}> Cuts &amp; zooms</label>
    <label class="check"><input type="checkbox" data-director="sfx" ${d.sfx?'checked':''} ${restricted}> Sound effects</label>
    <label>Music<select data-director="music" ${restricted}>${['auto','on','off'].map(m=>`<option value="${m}" ${d.music===m?'selected':''}>${m==='auto'?'Auto':m==='on'?'Always':'Off'}</option>`).join('')}</select></label>
    <label class="check"><input type="checkbox" data-director="story" ${d.story?'checked':''}> <span data-automatic-hook>${mode==='full'?'Storytelling hook':'Quote the original speech'}</span></label>
    <label class="check"><input type="checkbox" data-director="loop" ${d.loop?'checked':''} ${restricted}> Seamless loop</label>
    <label class="check"><input type="checkbox" data-director="voiceover" ${d.voiceover?'checked':''}> AI voice-over</label>
    <div class="voice-picker"><label>Voice<select data-director="voice">${voiceOptions(d.voice)}</select></label><label>Delivery<select data-director="delivery">${deliveryOptions(d.delivery)}</select></label><button type="button" data-action="voice-preview">▶ Preview voice</button><span class="hint voice-status" aria-live="polite">${escapeHtml(narrationLabel(clip))}</span></div>
    ${storyHtml(clip,d)}
    ${report}</fieldset>`;
}
function narrationLabel(clip) {
  const n=clip.director_result?.edits?.hook?.narration;
  return n ? `Last render: ${n.voice_name} · ${deliveryLabels[n.mood]||n.mood}${n.engine==='offline'?' · offline voice':''}` : '';
}
function storyHtml(clip, d) {
  const hook=clip.director_result?.edits?.hook, story=hook?.style==='story';
  const text=d.hook_text || (story || hook?.style==='source' ? hook.text : '');
  const still=clip.source_still_url ? `.${escapeHtml(clip.source_still_url)}?v=${clip.rendered_at||0}` : '';
  return `<div class="story-hook">
    <div class="story-preview" ${still?`style="background-image:url('${still}')"`:''}><span class="story-preview-text">${escapeHtml(text || 'Your hook appears here')}</span></div>
    <label>Hook text<input data-director="hook_text" maxlength="64" value="${escapeHtml(d.hook_text||'')}" placeholder="${escapeHtml(hook?.text || 'Chosen when the clip renders')}" autocomplete="off"></label>
    ${story && hook.options?.length ? `<div class="story-options" role="group" aria-label="Suggested hooks">${hook.options.map(o=>`<button type="button" data-hook-option="${escapeHtml(o)}">${escapeHtml(o)}</button>`).join('')}</div>` : ''}
    <p class="hint">${story ? escapeHtml([hook.streamer, hook.game, hook.category, hook.language==='id'?'Indonesian':'English', hook.writer==='llm'?'written by LLM':'written offline'].filter(Boolean).join(' · ')) + '. ' : ''}Leave blank for the automatic hook; re-render to apply.</p></div>`;
}
const authenticityLabels={context:'Context',subtitles:'Subtitles',title:'Title',hook:'Hook',thumbnail:'Thumbnail'};
function authenticityHtml(report) {
  if(!report) return '';
  const p=report.provenance, o=report.operations;
  const source=p.channel?[`${p.title || 'Original recording'} · ${p.channel}`]:[p.title||p.filename||'Original recording'];
  source.push(`${timestamp(p.start)}–${timestamp(p.end)}`);
  const link = /^https:\/\/www\.youtube\.com\/watch\?v=[\w-]{11}$/.test(p.url||'') ?
    ` · <a href="${escapeHtml(p.url)}" target="_blank" rel="noopener noreferrer">Original video ↗</a>` : '';
  return `<section class="authenticity" aria-label="Content authenticity">
    <h4>Content authenticity</h4>
    ${report.style?`<p class="hint">Adaptive style: ${escapeHtml(report.style)}</p>`:''}
    <ul class="quality">${Object.entries(report.checks).map(([key,c])=>`<li class="${c.status==='review'?'warn':escapeHtml(c.status)}"><strong>${escapeHtml(authenticityLabels[key]||key)}</strong> ${c.status==='review'?'Review: ':''}${escapeHtml(c.detail)}</li>`).join('')}</ul>
    <p class="hint">Synthetic audio: ${o.synthetic_audio?'added':'none'} · Generated footage: ${o.generated_footage?'added':'none'}${o.source_freeze_frame?' · Held source frame':''}${o.loop_blend?' · Loop blend':''}</p>
    <p><strong>Disclosure:</strong> ${escapeHtml(report.disclosure)}</p>
    <p class="hint"><strong>Source:</strong> ${escapeHtml(source.join(' · '))}${link}</p>
    <ul class="quality">${report.flags.map(f=>`<li class="warn">${escapeHtml(f)}</li>`).join('')}</ul>
    </section>`;
}
function facecamHtml(clip, draft) {
  const f=facecamDefaults(draft.facecam), result=clip.facecam_result, box=f.box || result?.box;
  const status=result?.status==='used' ? `${result.origin==='manual'?'Manual camera area':'Facecam detected'} · ${result.position}` : result?.status==='disabled' ? 'Facecam overlay off' : 'No facecam found — draw the camera area';
  return `<fieldset class="facecam-options" ${locked(clip)?'disabled':''}><legend>Facecam overlay</legend>
    <label class="check"><input type="checkbox" data-facecam="enabled" ${f.enabled?'checked':''}> Enabled</label>
    <label>Corner<select data-facecam="position">${['auto','top-left','top-right','bottom-left','bottom-right'].map(p=>`<option value="${p}" ${f.position===p?'selected':''}>${p==='auto'?'Auto':p.replace('-',' ')}</option>`).join('')}</select></label>
    <label>Size<input type="range" data-facecam="size" min="0.25" max="0.6" step="0.01" value="${f.size}"></label>
    <p class="hint">${escapeHtml(status)}</p><details><summary>Adjust camera area</summary>
    ${clip.source_still_url?`<p class="hint">Draw an area, drag to move, or use a corner to resize. Re-render applies your changes.</p><div class="camera-editor"><img draggable="false" alt="Original source frame for camera area" src=".${escapeHtml(clip.source_still_url)}?v=${clip.rendered_at||0}"><div class="camera-box" ${box?'':'hidden'} style="left:${(box?.x||0)*100}%;top:${(box?.y||0)*100}%;width:${(box?.w||0)*100}%;height:${(box?.h||0)*100}%">${['nw','ne','sw','se'].map(c=>`<span class="camera-handle ${c}" data-corner="${c}"></span>`).join('')}</div></div>`:'<p class="hint">Source preview available after rendering.</p>'}
    <button type="button" data-action="facecam-reset">Reset to auto-detect</button></details></fieldset>`;
}
$('#create-caption-style').innerHTML = CaptionControls.html();
CaptionControls.preview($('#create-caption-style'));
fetch('./api/voices').then(r=>r.ok?r.json():null).then(catalog=>{
  if(!catalog) return;
  voiceCatalog=catalog;
  $('#director-voice').innerHTML=voiceOptions($('#director-voice').value);
  $('#director-delivery').innerHTML=deliveryOptions($('#director-delivery').value);
  document.querySelectorAll('.clip').forEach(node=>{node.dataset.version='';});
  if(current) drawJob(current);
}).catch(()=>{});
$('#create-voice-preview').addEventListener('click',event=>previewVoice({voice:$('#director-voice').value,delivery:$('#director-delivery').value,language:speechLanguage($('#start-form'))},event.currentTarget,$('#create-voice-status')));
const jobMode = $('#editing-mode');
jobMode.innerHTML=modeOptions('full');
const jobDirectorKeys = ['enabled','music','story','voiceover','voice','delivery','loop'];
const jobDirectorField = k => $(k==='enabled'?'#director':`#director-${k}`);
const readJobDirector = () => Object.fromEntries(jobDirectorKeys.map(k=>{
  const el=jobDirectorField(k);
  return [k,el.type==='checkbox'?el.checked:el.value];
}));
const jobProfiles={};
let lastJobMode='full';
function syncJobMode() {
  const mode=jobMode.value, natural=mode==='natural';
  $('#editing-mode-hint').textContent=editingModes[mode][1];
  $('#director-story-label').textContent=mode==='full'?'Storytelling hook':'Quote the original speech';
  for(const key of ['loop','music']) jobDirectorField(key).disabled=natural;
  for(const key of ['story','voiceover']) jobDirectorField(key).parentElement.hidden=mode==='full' &&
    $('[name=content-type]:checked').value==='podcast';
  syncPodcastMode($('#start-form'),mode,'#podcast-version');
}
jobMode.addEventListener('change',()=>{
  jobProfiles[lastJobMode]=readJobDirector();
  CaptionControls.applyProfile($('#create-caption-style'),jobMode.value,lastJobMode);
  lastJobMode=jobMode.value;
  const profile=jobProfiles[lastJobMode] || directorDefaults(null,lastJobMode);
  for(const key of jobDirectorKeys) {
    const el=jobDirectorField(key);
    if(el.type==='checkbox') el.checked=profile[key];
    else el.value=profile[key];
  }
  syncJobMode();
});
$('#start-form').addEventListener('change',event=>{if(event.target.name==='content-type') syncJobMode();});
syncJobMode();
const thumbnailUrl = clip => clip.thumbnail_url ? `.${clip.thumbnail_url}?v=${clip.thumbnail_version || clip.rendered_at || 0}` : '';

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
  if (job.filename) return job.filename;
  try {
    const url = new URL(job.url);
    const id = url.searchParams.get('v') || url.pathname.split('/').filter(Boolean).pop();
    if (/^[a-zA-Z0-9_-]{11}$/.test(id)) return `YouTube video ${id}`;
  } catch { return 'Untitled stream'; }
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
function clipHtml(clip,index) {
  const draft = drafts.get(clip.id) || {start:timestamp(clip.start),end:timestamp(clip.end),mode:clip.mode,captions:clip.captions,facecam:facecamDefaults(clip.facecam),tracking:TrackingControls.defaults(clip.tracking),editing_mode:editingMode(clip),director:directorDefaults(clip.director,editingMode(clip)),podcast:PodcastControls.defaults(clip),...CaptionControls.options(clip)};
  const disabled = locked(clip) ? 'disabled' : '';
  const number = clip.number ?? index+1;
  const timeField = field => `<label>${field === 'start' ? 'Start' : 'End'}<input class="time" data-field="${field}" value="${escapeHtml(draft[field])}" aria-label="${field} time" inputmode="text" autocomplete="off"></label><div class="nudges">${[-5,-1,1,5].map(n=>`<button type="button" data-nudge="${n}" data-field="${field}" aria-label="${field} ${n>0?'plus':'minus'} ${Math.abs(n)} seconds">${n>0?'+':''}${n}s</button>`).join('')}</div>`;
  return `<article id="clip-${clip.id}" class="clip" data-id="${clip.id}" data-version="${escapeHtml(clipVersion(clip))}">
    <div class="clip-top"><div><h3>Highlight ${String(number).padStart(2,'0')}</h3>${badge(clip.status, clip.queue_position)}${clip.needs_review?'<span class="badge review-flag">Needs review</span>':''}</div><button class="keep" aria-label="Keep highlight ${number}" aria-pressed="${!!clip.kept}" data-action="keep">${clip.kept?'★ Kept':'☆ Keep'}</button></div>
    <div class="preview${clip.status==='ready' ? ' with-thumbnail' : ''}">${clip.status==='ready' ? `<video controls playsinline ${clip.director_result?.edits?.loop?.status==='applied'?'loop':''} preload="metadata" poster="${escapeHtml(thumbnailUrl(clip))}"><source src=".${clip.file_url}?v=${clip.rendered_at || 0}-${clip.start}-${clip.end}-${clip.mode}-${clip.captions}" type="video/mp4"></video><figure class="thumbnail-preview" ${clip.thumbnail_url ? '' : 'hidden'}><img src="${escapeHtml(thumbnailUrl(clip))}" alt="Thumbnail for highlight ${number}" loading="lazy"><figcaption>Thumbnail</figcaption></figure>` : `<div class="placeholder"><span style="font-size:30px">${clip.status==='failed'?'!':'✂'}</span><span>${clip.status==='failed'?'This clip needs another try':clip.status==='cancelled'?'Cancelled — re-render to try again':clip.status==='queued'?escapeHtml(queueLabel(clip.queue_position)):`Creating your highlight…<br>Attempt ${clip.attempt || 1} of 3`}</span></div>`}</div>
    <div class="clip-body">${clip.title?`<h3>${escapeHtml(clip.title)}</h3>`:''}${clip.sources?.length?`<p class="moment-source">${escapeHtml(clip.sources.length>1?'AI + Chat':clip.sources[0]==='ai'?'AI Analysis':'Chat Analysis')}</p>`:''}<div class="clip-meta"><span class="score">↗ ${clip.sources?.length?`${Math.round(clip.score)} / 100 Viral Potential`:`${clip.score.toFixed(2)} spike score`}</span><span>${timestamp(clip.end-clip.start)} clip</span></div><p class="hint">${clip.onset!=null?`Onset ${timestamp(clip.onset)} · `:''}Peak ${timestamp(clip.peak)}</p>
    ${PodcastControls.isPodcastClip(clip)?PodcastControls.summary(clip):`<div class="samples">${clip.message_sample.length ? clip.message_sample.map(m=>`<span class="sample">${escapeHtml(m)}</span>`).join('') : 'No sample messages'}</div>`}
    ${clip.error ? `<p class="error">${escapeHtml(clip.error)}${clip.status==='failed'?'<br>Adjust the settings below or try Re-render again.':''}</p>` : ''}
    ${authenticityHtml(clip.authenticity)}
    ${clip.status==='ready' ? `<form class="thumbnail-editor"><label>Thumbnail text<input name="text" maxlength="64" value="${escapeHtml(clip.thumbnail_text)}" placeholder="${escapeHtml(clip.thumbnail_hook || 'Automatic hook')}" autocomplete="off"></label><p class="hint">Leave blank to use the automatic hook.</p><div class="actions"><button type="submit">Regenerate ↻</button><a class="button thumbnail-download">Download JPG ↓</a></div><p class="hint thumbnail-status" role="status" hidden></p></form>` : ''}
    <form class="review"><div class="time-grid"><div>${timeField('start')}</div><div>${timeField('end')}</div></div>
      <div class="render-options"><label>Layout<select data-field="mode"><option value="crop" ${draft.mode==='crop'?'selected':''}>Center crop</option><option value="blur" ${draft.mode==='blur'?'selected':''}>Blurred background</option></select></label><label class="check"><input type="checkbox" data-field="captions" ${draft.captions?'checked':''}> Captions</label></div>
      ${PodcastControls.clipHtml(clip,draft,disabled)}
      ${directorHtml(clip,draft)}
      <div ${PodcastControls.isPodcastClip(clip)?'hidden':''}>${facecamHtml(clip,draft)}
      ${TrackingControls.html(clip,draft)}</div>
      ${CaptionControls.html(draft)}
      ${PodcastControls.isPodcastClip(clip)?'':`<fieldset ${disabled}><legend>Subtitle corrections</legend>${CaptionControls.corrections(draft.caption_edits ?? clip.caption_edits)}</fieldset>`}
      ${clip.caption_language?`<p class="hint">Transcription language: ${escapeHtml(clip.caption_language)}</p>`:''}
      <p class="draft-note hint" hidden><span>Changes not applied — Re-render to update</span> <button type="button" class="quiet" data-action="reset">Reset</button></p><p class="draft-error error" role="status" hidden></p>
      ${locked(clip)?'<p class="hint">This highlight is queued or rendering. Review actions unlock when it finishes.</p>':''}
      <div class="actions"><button type="submit" class="primary" ${disabled}>Re-render ↻</button>${clip.status==='ready'?(PodcastControls.exportable(current)?`<a class="button" href=".${clip.download_url}">Download ↓</a>`:'<span class="hint">Confirm rights above to download</span>'):''}<button class="delete" type="button" data-action="delete" aria-label="Delete highlight ${number}" ${disabled}>Delete</button></div>
    </form></div></article>`;
}
const stageLabels = {
  source_validation: 'Source validation', video_analysis: 'Video analysis',
  audio_analysis: 'Audio analysis', transcript_generation: 'Transcript generation',
  chat_analysis: 'Chat analysis', viral_moment_detection: 'Viral moment detection',
  ai_director: 'AI Director', final_rendering: 'Final rendering',
};
const stageStatusLabels = {
  pending: 'Pending', processing: 'Processing', completed: 'Completed',
  partially_completed: 'Partially completed', skipped: 'Skipped', retrying: 'Retrying', failed: 'Failed',
};
let stageMarkup, stageAction;
function drawStages(job) {
  const idle = !busy(job) && !job.chat_retrying && !job.chat_pending;
  const resume = idle && !job.analysis_complete && ['failed', 'cancelled'].includes(job.status);
  $('#analysis-stages').hidden = !job.stages;
  $('#resume-analysis').hidden = !resume;
  $('#resume-analysis').disabled = !!stageAction;
  const markup = Object.entries(stageLabels).map(([name, label])=>{
    const stage = job.stages?.[name];
    if (!stage) return '';
    const retry = idle && job.status !== 'cancelled' && ['failed', 'partially_completed'].includes(stage.status);
    const color = {processing:'running', retrying:'queued', partially_completed:'completed_with_errors'}[stage.status] || stage.status;
    const percent = Math.round(stage.progress);
    return `<li class="stage-item">
      <div class="stage-info"><strong>${label}</strong><small class="hint">${stage.attempts ? `Attempts: ${stage.attempts}` : ''}</small></div>
      <span class="badge ${escapeHtml(color)}">${stageStatusLabels[stage.status] || 'Pending'}</span>
      <div class="stage-meter"><progress max="100" value="${percent}" aria-label="${label} progress"></progress><span>${percent}%</span></div>
      ${retry ? `<button class="quiet stage-retry" type="button" data-stage="${name}" aria-label="Retry Failed Stage: ${label}" ${stageAction ? 'disabled' : ''}>Retry Failed Stage</button>` : ''}
      ${stage.error ? `<small class="stage-error">${escapeHtml(stage.error)}</small>` : ''}
    </li>`;
  }).join('');
  if (markup !== stageMarkup) {
    $('#stage-list').innerHTML = markup;
    stageMarkup = markup;
  }
}
function drawJob(job) {
  job.clips.forEach(c=>{if(keeping.has(c.id))c.kept=keeping.get(c.id);});
  current = job;
  const image = safeImage(job.metadata?.thumbnail);
  $('#job-heading').innerHTML = `<div class="job-header">${image?`<img src="${image}" alt="Stream thumbnail">`:''}<div>${badge(job.status, job.queue_position)}<h1>${escapeHtml(jobTitle(job))}</h1><p>${date(job)} ${job.metadata?`· ${timestamp(job.metadata.duration)} stream`:''} · ${escapeHtml(job.request.mode)} layout · ${editingModes[editingMode(job.request)][0]}</p></div></div>`;
  $('#stage').textContent = job.stage;
  $('#progress').value = job.progress;
  $('#progress-label').textContent = `${Math.round(job.progress*100)}%`;
  const cancellable = canCancel(job) && 'queue_position' in job;
  $('#cancel-job').hidden = !cancellable;
  $('#cancel-job').textContent = busy(job) ? 'Cancel job' : 'Cancel chat analysis';
  $('#cancel-hint').hidden = !cancellable || !busy(job);
  $('#job-error').hidden = !job.error;
  $('#job-error').textContent = job.error || '';
  $('#analysis-detail').hidden = !busy(job) || !job.analysis_detail;
  $('#analysis-detail').textContent = job.analysis_detail || '';
  drawStages(job);
  const retry = !PodcastControls.isPodcast(job) && job.url && job.metadata &&
    (job.chat_error || job.status==='failed') && ['completed','completed_with_errors','failed'].includes(job.status);
  $('#chat-status').hidden = !(retry || job.chat_retrying || job.chat_pending || job.chat_warning || job.chat_error);
  $('#chat-warning').hidden = !job.chat_warning;
  $('#chat-warning').textContent = job.chat_warning || '';
  $('#chat-detail').hidden = !(job.chat_retrying || job.chat_pending || job.chat_error);
  $('#chat-detail').textContent = job.chat_retrying ? 'Retrying chat analysis…' :
    job.chat_pending ? 'Chat analysis is still running…' : job.chat_error || '';
  $('#retry-chat').hidden = !retry || !!job.chat_retrying || !!job.chat_pending;
  globalThis.MomentReview?.draw(job);
  PodcastControls.drawRights(job);
  $('.activity-panel').hidden = PodcastControls.isPodcast(job) || !!job.upload;
  if (!$('.activity-panel').hidden) chart(job);
  $('#clips-title').textContent = `Your highlights${job.clips.length ? ' · '+job.clips.length : ''}`;
  const kept = job.clips.filter(c=>c.kept && c.status==='ready').length;
  $('#download-kept').hidden = !kept || !PodcastControls.exportable(job);
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
    node.querySelectorAll('.review [type=submit],[data-action=delete]').forEach(button=>button.disabled=locked(clip));
    if (clip.status==='ready') updateThumbnail(node, clip);
    draftFeedback(node);
  });
  if (!job.clips.length) container.innerHTML = `<p class="empty">${busy(job)?'Your clips will appear here as they finish.':job.status==='cancelled'?'This job was cancelled. Return to All streams to start another.':job.error?'The stream could not be processed. Return to All streams to try another URL.':job.moments?.length?'Preview and select moments above, then render your favorites.':job.request.review?'No viral moments found. Try another replay.':'No significant chat spikes found. Try a livelier replay.'}</p>`;
}
function updateThumbnail(node, clip) {
  const url = thumbnailUrl(clip);
  const figure = $('.thumbnail-preview',node), image = $('img',figure);
  const download = $('.thumbnail-download',node), input = $('.thumbnail-editor input',node);
  if (image.getAttribute('src') !== url) {
    image.setAttribute('src',url);
    $('video',node).setAttribute('poster',url);
  }
  figure.hidden = download.hidden = !url;
  if (url) download.href = url + '&download=true';
  input.placeholder = clip.thumbnail_hook || 'Automatic hook';
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
  globalThis.MomentReview?.reset();
  $('#analysis-detail').hidden=true;
  $('#stage').textContent='Loading job…'; $('#progress-label').textContent='';
  $('#cancel-job').hidden=true; $('#cancel-hint').hidden=true;
  $('#progress').removeAttribute('value'); $('#job-error').hidden=true;
  $('#chat-status').hidden=true; $('#retry-chat-error').hidden=true;
  $('#analysis-stages').hidden=true; $('#stage-action-error').hidden=true;
  chartKey=undefined; $('#chart').innerHTML='';
  $('#download-kept').hidden=true; $('#delete-unkept').hidden=true; $('#keep-hint').hidden=true;
  globalThis.scrollTo(0,0);
  clearInterval(timer); refresh(); timer=setInterval(refresh,2000);
}
$('#lead').addEventListener('input',event=>$('#lead-value').textContent=`${event.target.value} seconds`);
$('#refresh-jobs').addEventListener('click',()=>refresh());
async function runStageAction(path) {
  const job = current, errorNode = $('#stage-action-error');
  if (!job || stageAction) return;
  const action = stageAction = {};
  drawStages(job);
  errorNode.hidden = true;
  try {
    await api(`./api/jobs/${job.id}/${path}`, 'POST');
    if (current?.id === job.id) await refresh();
  } catch (error) {
    if (current?.id === job.id) {
      errorNode.textContent = error.message;
      errorNode.hidden = false;
    }
  } finally {
    if (stageAction === action) stageAction = null;
    if (current) drawStages(current);
  }
}
$('#resume-analysis').addEventListener('click',()=>runStageAction('resume'));
$('#stage-list').addEventListener('click',event=>{
  const button = event.target.closest('[data-stage]');
  if (button) runStageAction(`stages/${button.dataset.stage}/retry`);
});
$('#retry-chat').addEventListener('click',async event=>{
  const job=current, button=event.currentTarget, errorNode=$('#retry-chat-error');
  button.disabled=true; errorNode.hidden=true;
  try {
    await api(`./api/jobs/${job.id}/chat/retry`,'POST');
    if(current?.id===job.id) await refresh();
  } catch(error) {
    if(current?.id===job.id) {
      errorNode.textContent=error.message; errorNode.hidden=false;
    }
  } finally { button.disabled=false; }
});
$('#start-form').addEventListener('submit',async event=>{
  event.preventDefault(); notice(''); const button=$('#start-button'); button.disabled=true;
  try {
    const source=await PodcastControls.source();
    const job=await api('./api/jobs','POST',{...source,...PodcastControls.request(),review:$('#review-moments').checked,clips:Number($('#clip-count').value),lead:Number($('#lead').value),mode:$('#mode').value,captions:$('#captions').checked,facecam:{enabled:$('#facecam').checked},editing_mode:jobMode.value,director:readJobDirector(),...CaptionControls.read($('#start-form'))});
    location.hash=`jobs/${job.id}`;
  } catch(error) {
    notice(error instanceof TypeError ? 'Couldn’t confirm the submission. Check Recent streams before trying again.' : error.message);
  } finally { button.disabled=false; }
});
function saveDraft(node, profiles=drafts.get(node.dataset.id)?.profiles) {
  const clip=current.clips.find(c=>c.id===node.dataset.id);
  const previous=facecamDefaults(drafts.get(clip.id)?.facecam || clip.facecam);
  const facecam={...previous,enabled:$('[data-facecam=enabled]',node).checked,position:$('[data-facecam=position]',node).value,size:Number($('[data-facecam=size]',node).value)};
  drafts.set(node.dataset.id,{start:$('[data-field=start].time',node).value,end:$('[data-field=end].time',node).value,mode:$('[data-field=mode]',node).value,captions:$('[data-field=captions]',node).checked,facecam,tracking:TrackingControls.read(node,drafts.get(clip.id)?.tracking || clip.tracking),editing_mode:$('[data-field=editing_mode]',node).value,director:readDirector(node),profiles,podcast:PodcastControls.read(node,clip),caption_edits:CaptionControls.readCorrections(node),...CaptionControls.read(node)});
  draftFeedback(node);
}
function draftFeedback(node) {
  const clip=current.clips.find(c=>c.id===node.dataset.id), draft=drafts.get(clip.id);
  let error='',changed=false,length=clip.end-clip.start;
  if(draft) {
    changed=draft.start!==timestamp(clip.start)||draft.end!==timestamp(clip.end)||draft.mode!==clip.mode||draft.captions!==clip.captions||draft.editing_mode!==editingMode(clip)||JSON.stringify(draft.facecam)!==JSON.stringify(facecamDefaults(clip.facecam))||JSON.stringify(draft.director)!==JSON.stringify(directorDefaults(clip.director,editingMode(clip)))||JSON.stringify(draft.tracking)!==JSON.stringify(TrackingControls.defaults(clip.tracking))||JSON.stringify(draft.podcast)!==JSON.stringify(PodcastControls.defaults(clip));
    changed=changed||JSON.stringify({language:draft.language,caption_style:draft.caption_style})!==JSON.stringify(CaptionControls.options(clip));
    changed=changed||JSON.stringify(draft.caption_edits)!==JSON.stringify(clip.caption_edits || []);
    try {
      const start=parseTime(draft.start),end=parseTime(draft.end); length=end-start;
      const f=draft.facecam,b=f?.box;
      if(!f || !Number.isFinite(f.size) || f.size<.25 || f.size>.6 || !['auto','top-left','top-right','bottom-left','bottom-right'].includes(f.position))throw Error('Choose valid facecam settings.');
      if(b && (![b.x,b.y,b.w,b.h].every(Number.isFinite) || b.x<0 || b.y<0 || b.w<=0 || b.h<=0 || b.x+b.w>1 || b.y+b.h>1))throw Error('Camera area must fit inside the source frame.');
      if(end<=start || length>60 || end>current.metadata.duration)throw Error('Choose a window within the stream, with end after start and a maximum of 60 seconds.');
    } catch(e) {error=e.message;length=null;}
  }
  $('.clip-meta span:last-child',node).textContent=length===null?'Invalid clip length':`${timestamp(length)} clip`;
  $('.draft-note',node).hidden=!changed;
  $('.draft-error',node).hidden=!error;
  $('.draft-error',node).textContent=error;
  $('.review [type=submit]',node).disabled=locked(clip)||!!error;
}
$('#cancel-job').addEventListener('click',async event=>{
  if(!current || !canCancel(current))return;
  const jid=current.id, scope=busy(current) ? 'job' : 'chat';
  if(!confirm(scope==='job' ? 'Cancel this job? Ready clips will be kept.' :
    'Cancel chat analysis? Existing moments and clips will be kept.')) return;
  event.target.disabled=true;
  try {notice('');await api(`./api/jobs/${jid}/cancel?scope=${scope}`,'POST');await refresh();}
  catch(error){notice(error.message);}finally{event.target.disabled=false;}
});
$('#delete-unkept').addEventListener('click',async event=>{
  const clips=current.clips.filter(c=>!c.kept && c.status!=='rendering');
  if(!confirm(`Delete ${clips.length} unkept ${clips.length===1?'highlight and its files':'highlights and their files'}?`))return;
  event.target.disabled=true;
  try {for(const clip of clips){await api(`./api/jobs/${current.id}/clips/${clip.id}`,'DELETE');drafts.delete(clip.id);}}
  catch(error){notice(error.message);}finally{event.target.disabled=false;await refresh();}
});
$('#clips').addEventListener('input',event=>{
  if(event.target.matches('[data-field=editing_mode]')) {
    const node=event.target.closest('.clip'), clip=current.clips.find(c=>c.id===node.dataset.id);
    const previous=drafts.get(clip.id), lastMode=previous?.editing_mode || editingMode(clip), mode=event.target.value;
    const profiles={...previous?.profiles,[lastMode]:readDirector(node)};
    const profile=profiles[mode] || directorDefaults(null,mode);
    for(const key of directorKeys) {
      const el=$(`[data-director=${key}]`,node);
      if(el.type==='checkbox') el.checked=profile[key];
      else el.value=profile[key] ?? '';
      if(['sfx','music','loop'].includes(key)) el.disabled=mode==='natural';
    }
    $('[data-automatic-hook]',node).textContent=mode==='full'?'Storytelling hook':'Quote the original speech';
    $('[data-editing-hint]',node).textContent=editingModes[mode][1];
    const input=$('[data-director=hook_text]',node);
    $('.story-preview-text',node).textContent=input.value || input.placeholder;
    syncPodcastMode(node,mode,'[data-podcast=version]');
    CaptionControls.applyProfile(node,mode,lastMode);
    saveDraft(node,profiles);
    return;
  }
  if(event.target.matches('[data-director=hook_text]')) {
    const preview=$('.story-preview-text',event.target.closest('.story-hook'));
    preview.textContent=event.target.value.trim() || event.target.placeholder;
  }
  if(event.target.closest('.review'))saveDraft(event.target.closest('.clip'));
});
$('#clips').addEventListener('click',event=>{
  const option=event.target.closest('[data-hook-option]');
  if(!option) return;
  const input=$('[data-director=hook_text]',option.closest('.story-hook'));
  input.value=option.dataset.hookOption;
  input.dispatchEvent(new Event('input',{bubbles:true}));
});
$('#clips').addEventListener('change',event=>{if(event.target.closest('.review'))saveDraft(event.target.closest('.clip'));});
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
        const offset = renderedOffset(clip, parseTime(input.value) - clip.start - (button.dataset.field==='end' ? 2 : 0));
        video.pause();
        video.currentTime = Math.min(video.duration,Math.max(0,offset));
      }
    } else if(button.dataset.action==='keep') {
      const clip=current.clips.find(c=>c.id===cid), previous=!!clip.kept, kept=!previous;
      keeping.set(cid,kept); clip.kept=kept; button.disabled=true; drawJob(current);
      try {await api(base,'PATCH',{kept});}catch(error){const latest=current.clips.find(c=>c.id===cid);if(latest)latest.kept=previous;throw error;}
      finally {keeping.delete(cid);drawJob(current);}
      await refresh();
    } else if(button.dataset.action==='voice-preview') {
      const clip=current.clips.find(c=>c.id===cid), hook=clip?.director_result?.edits?.hook, field=button.closest('fieldset');
      const d=readDirector(node), p=PodcastControls.read(node,clip), mode=$('[data-field=editing_mode]',node).value;
      const old=directorDefaults(clip.director,editingMode(clip)), oldPodcast=PodcastControls.defaults(clip);
      const language=speechLanguage(node,{language:hook?.language || clip.caption_language});
      const commentary=p?.version==='commentary' && mode!=='natural' ? p.commentary || clip.podcast_analysis?.commentary : null;
      const text=p?.hook_text || d.hook_text || (['story','source'].includes(hook?.style) && !clip.director_result?.edits?.commentary ? hook.text : null);
      const same=mode===editingMode(clip) && ['enabled','hook','story','voiceover','voice','delivery','hook_text'].every(k=>d[k]===old[k])
        && ['version','hook_text','commentary'].every(k=>p?.[k]===oldPodcast?.[k])
        && $('[data-caption=language]',node).value===(clip.language || 'auto');
      await previewVoice({voice:d.voice,delivery:d.delivery,text:text?.slice(0,64) || null,commentary,language,seed:cid,category:hook?.category||null},
        button,$('.voice-status',field),same?clip.narration_url:null);
    } else if(button.dataset.action==='facecam-reset') {
      saveDraft(node); drafts.get(cid).facecam.box=null;
      const box=$('.camera-box',node); if(box)box.hidden=true;
      draftFeedback(node);
    } else if(button.dataset.action==='reset') {
      drafts.delete(cid); node.outerHTML=clipHtml(current.clips.find(c=>c.id===cid),current.clips.findIndex(c=>c.id===cid),current);
    } else if(button.dataset.action==='delete' && confirm('Delete this clip and its files?')) {
      button.disabled=true; await api(base,'DELETE'); drafts.delete(cid); await refresh();
    }
  } catch(error) { notice(error.message); } finally { if(button.dataset.action==='keep')button.disabled=false; }
});
$('#clips').addEventListener('submit',async event=>{
  event.preventDefault(); if(!current)return;
  const form=event.target, node=form.closest('.clip'), cid=node.dataset.id;
  const jid=current.id, button=$('[type=submit]',form);
  if (form.matches('.thumbnail-editor')) {
    const input=$('input',form), status=$('.thumbnail-status',form);
    status.hidden=false; status.textContent='Generating thumbnail…';
    button.disabled=input.disabled=true;
    try {
      const result=await api(`./api/jobs/${jid}/clips/${cid}/thumbnail`,'POST',{text:input.value});
      const clip=current?.id===jid && current.clips.find(c=>c.id===cid);
      if (clip?.status==='ready' && node.isConnected) {
        Object.assign(clip,result);
        input.value=result.thumbnail_text || '';
        updateThumbnail(node,clip);
        if(result.authenticity) {
          $('.authenticity',node).outerHTML=authenticityHtml(result.authenticity);
          node.dataset.version=clipVersion(clip);
        }
      }
      status.textContent='Thumbnail updated.';
    } catch(error) { status.textContent=error.message; }
    finally { button.disabled=input.disabled=false; }
    return;
  }
  saveDraft(node);
  const draft=drafts.get(cid);
  try {
    notice(''); button.disabled=true;
    const start=parseTime(draft.start),end=parseTime(draft.end);
    if(end<=start || end-start>60 || end>current.metadata.duration)throw Error('Choose a window within the stream, with end after start and a maximum of 60 seconds.');
    await api(`./api/jobs/${jid}/clips/${cid}/render`,'POST',{start,end,mode:draft.mode,captions:draft.captions,facecam:draft.facecam,tracking:retimeTarget(draft.tracking,current.clips.find(c=>c.id===cid)?.start,start,end),editing_mode:draft.editing_mode,director:draft.director,podcast:draft.podcast,language:draft.language,caption_style:draft.caption_style,caption_edits:draft.caption_edits}); drafts.delete(cid); await refresh();
  } catch(error) { notice(error.message); button.disabled=false; }
});
// A picked target's time is relative to the rendered section; keep it on the same frame when Start moves.
function retimeTarget(tracking,oldStart,start,end){
  if(!tracking?.target)return tracking;
  const t=tracking.target.t+(Number(oldStart)||0)-start;
  return {...tracking,target:t>=0&&t<=end-start?{...tracking.target,t:Math.round(t*100)/100}:null};
}
let cameraDrag=null;
$('#clips').addEventListener('pointerdown',event=>{
  const editor=event.target.closest('.camera-editor'), node=editor?.closest('.clip');
  if(!editor || locked(current.clips.find(c=>c.id===node.dataset.id)) || cameraDrag)return;
  event.preventDefault(); saveDraft(node);
  const bounds=editor.getBoundingClientRect(), x=Math.max(0,Math.min(1,(event.clientX-bounds.left)/bounds.width)), y=Math.max(0,Math.min(1,(event.clientY-bounds.top)/bounds.height));
  const draft=drafts.get(node.dataset.id), used=current.clips.find(c=>c.id===node.dataset.id).facecam_result?.box;
  const inside=event.target.closest('.camera-box'), box=draft.facecam.box || used;
  cameraDrag={editor,node,id:event.pointerId,x,y,box:inside&&box?{...box}:null,corner:event.target.dataset.corner};
  editor.setPointerCapture(event.pointerId);
});
$('#clips').addEventListener('pointermove',event=>{
  const d=cameraDrag; if(!d || d.id!==event.pointerId)return;
  const bounds=d.editor.getBoundingClientRect(), clamp=v=>Math.max(0,Math.min(1,v));
  const x=clamp((event.clientX-bounds.left)/bounds.width), y=clamp((event.clientY-bounds.top)/bounds.height);
  let box;
  if(d.box && !d.corner) {
    box={...d.box,x:Math.max(0,Math.min(1-d.box.w,d.box.x+x-d.x)),y:Math.max(0,Math.min(1-d.box.h,d.box.y+y-d.y))};
  } else {
    const anchorX=d.box?(d.corner.includes('w')?d.box.x+d.box.w:d.box.x):d.x;
    const anchorY=d.box?(d.corner.includes('n')?d.box.y+d.box.h:d.box.y):d.y;
    const left=Math.min(x,anchorX), top=Math.min(y,anchorY);
    box={x:left,y:top,w:Math.abs(x-anchorX),h:Math.abs(y-anchorY)};
    if(box.w<.02 || box.h<.02)return;
  }
  drafts.get(d.node.dataset.id).facecam.box=box;
  const rect=$('.camera-box',d.editor); rect.hidden=false;
  Object.assign(rect.style,{left:`${box.x*100}%`,top:`${box.y*100}%`,width:`${box.w*100}%`,height:`${box.h*100}%`});
  draftFeedback(d.node);
});
for(const type of ['pointerup','pointercancel','lostpointercapture'])$('#clips').addEventListener(type,event=>{
  if(cameraDrag?.id===event.pointerId)cameraDrag=null;
});
function jump(event) {const mark=event.target.closest('[data-clip]');if(mark)document.getElementById(`clip-${mark.dataset.clip}`)?.scrollIntoView({behavior:'smooth',block:'start'});}
$('#chart').addEventListener('click',jump);
$('#chart').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();jump(event);}});
new ResizeObserver(()=>{if(current && !$('#job-view').hidden)chart(current);}).observe($('#chart'));
globalThis.addEventListener('hashchange',route);
route();
