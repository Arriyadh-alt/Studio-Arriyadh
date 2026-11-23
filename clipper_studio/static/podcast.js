globalThis.PodcastControls = (() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const LAYOUTS = {auto:'Auto (best for the conversation)',speaker:'Single speaker focus',split:'Two-person split screen',switch:'Dynamic speaker switching',wide:'Full conversation view'};
  const VERSIONS = {original:'A · Original podcast highlight',commentary:'B · AI-enhanced commentary'};
  const KINDS = {qa:'Question & answer',story:'Story',debate:'Debate',insight:'Insight',discussion:'Discussion'};
  const BASIS = {owned:'I own this content',licensed:'Licensed for reuse',permission:'Used with the creator’s permission',remix:'Permitted remix / commentary use'};
  const METERS = {hook:'Hook strength',context:'Context completeness',retention:'Retention potential',entertaining:'Entertaining',educational:'Educational',surprising:'Surprising',emotional:'Emotional',controversial:'Controversial'};
  const form = () => document.querySelector('#start-form');
  const type = () => form().querySelector('[name=content-type]:checked')?.value || 'gaming';
  const isPodcast = job => job?.request?.content_type === 'podcast';
  const options = (map, value) => Object.entries(map).map(([k,l])=>`<option value="${k}" ${value===k?'selected':''}>${esc(l)}</option>`).join('');
  const jobId = () => location.hash.match(/^#jobs\/([0-9a-f]{32})/)?.[1];

  function sync() {
    const podcast = type() === 'podcast';
    form().querySelectorAll('[data-gaming-only]').forEach(node => { node.hidden = podcast; });
    document.querySelector('#podcast-options').hidden = !podcast;
    document.querySelector('#url-label').textContent = podcast ? 'Paste YouTube podcast or interview URL' : 'Paste YouTube livestream URL';
  }
  form().addEventListener('change', event => {
    if (event.target.name === 'content-type') sync();
    if (event.target.id === 'upload' && event.target.files.length) document.querySelector('#url').value = '';
  });
  sync();

  async function source() {
    const file = document.querySelector('#upload').files[0], url = document.querySelector('#url').value.trim();
    if (!file) {
      if (!url) throw Error('Paste a YouTube link or upload a video you’re authorized to use.');
      return {url};
    }
    const status = document.querySelector('#upload-status');
    status.hidden = false; status.textContent = `Uploading ${file.name}…`;
    try {
      const response = await fetch('./api/uploads', {method:'POST', body:file, headers:{'X-Filename':encodeURIComponent(file.name)}});
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw Error(typeof body.detail === 'string' ? body.detail : 'Upload failed.');
      status.textContent = `Uploaded ${file.name}`;
      return {upload: body.id};
    } catch (error) { status.textContent = error.message; throw error; }
  }
  function request() {
    if (type() !== 'podcast') return {content_type:'gaming'};
    const speakers = document.querySelector('#podcast-speakers').value;
    return {content_type:'podcast', podcast:{layout:document.querySelector('#podcast-layout').value, version:document.querySelector('#podcast-version').value, speakers:speakers==='auto'?null:Number(speakers)}};
  }

  const exportable = job => !isPodcast(job) || !!job.rights?.confirmed;
  function drawRights(job) {
    const panel = document.querySelector('#rights-panel');
    panel.hidden = !isPodcast(job);
    if (panel.hidden) { panel.innerHTML = ''; panel.dataset.key = ''; return; }
    const key = JSON.stringify([job.id, job.rights]);
    if (panel.dataset.key === key) return;
    panel.dataset.key = key;
    const r = job.rights;
    panel.innerHTML = r?.confirmed
      ? `<p class="rights-done"><strong>✓ Rights confirmed</strong> ${esc(BASIS[r.basis])}${r.attribution?' · source credited':''}${r.include_link&&job.url?' · original link in descriptions':''}</p>`
      : `<form class="rights-form"><div><span class="eyebrow">Before you export</span><h2>Confirm your rights to this source</h2><p class="hint">Being public on YouTube does not grant permission to republish or monetize. Preview and review stay available; downloads unlock after you confirm.</p></div>
        <label>Basis<select name="basis">${options(BASIS, 'licensed')}</select></label>
        <label class="check"><input type="checkbox" name="attribution" checked> Credit the original source in descriptions</label>
        <label class="check"><input type="checkbox" name="include_link" checked ${job.url?'':'disabled'}> Include the original podcast link</label>
        <label class="check"><input type="checkbox" name="confirmed" required> I confirm I’m allowed to republish these clips on that basis</label>
        <button class="primary" type="submit">Confirm rights</button><p class="error rights-error" role="status" hidden></p></form>`;
  }
  document.addEventListener('submit', async event => {
    if (!event.target.matches('.rights-form')) return;
    event.preventDefault();
    const f = event.target, data = new FormData(f), error = f.querySelector('.rights-error');
    try {
      const response = await fetch(`./api/jobs/${jobId()}/rights`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({basis:data.get('basis'), confirmed:data.has('confirmed'), attribution:data.has('attribution'), include_link:data.has('include_link')})});
      const body = await response.json();
      if (!response.ok) throw Error(typeof body.detail === 'string' ? body.detail : body.detail?.map(d=>d.msg).join(' ') || 'Could not save.');
      refresh();
    } catch (e) { error.textContent = e.message; error.hidden = false; }
  });

  function meters(p) {
    const values = {hook:p.hook?.strength, context:p.context?.score, ...p.dimensions};
    return `<ul class="podcast-meters">${Object.entries(METERS).filter(([k])=>values[k]!=null).map(([k,l])=>`<li><span>${l}</span><meter min="0" max="100" value="${values[k]}"></meter><b>${Math.round(values[k])}</b></li>`).join('')}</ul>`;
  }
  function momentHtml(m, rendered, job) {
    const p = m.podcast;
    if (!p) return '';
    const e = {...(job.request.podcast || {}), ...(m.edits || {})};
    return `<div class="podcast-facts"><span class="moment-source">${esc(KINDS[p.kind] || 'Discussion')}</span>${p.topic?`<span>Topic: ${esc(p.topic)}</span>`:''}<span>${esc((p.speakers||[]).map(s=>s.label).join(' · '))}</span></div>
      ${p.question?`<p class="podcast-question"><strong>Q</strong> ${esc(p.question)}</p>`:''}
      ${p.hook?.text?`<blockquote class="podcast-hook">“${esc(p.hook.text)}”</blockquote>`:''}
      ${meters(p)}${p.context?.notes?.length?`<p class="hint">${esc(p.context.notes.join(' · '))}</p>`:''}
      ${rendered?'':`<form class="moment-podcast"><label>Opening hook text<input name="hook_text" maxlength="64" value="${esc(e.hook_text)}" placeholder="${esc(m.title)}"></label>
        <label>Speaker framing<select name="layout">${options(LAYOUTS, e.layout || 'auto')}</select></label>
        <label>Editing version<select name="version">${options(VERSIONS, e.version || 'original')}</select></label>
        <label>AI commentary <span class="hint">shown labelled “AI commentary”, never as a speaker</span><textarea name="commentary" maxlength="160" placeholder="${esc(p.commentary)}">${esc(e.commentary)}</textarea></label>
        <button type="submit">Save podcast edits</button></form>`}`;
  }
  function readMoment(f) {
    const data = new FormData(f);
    return {layout:data.get('layout'), version:data.get('version'), hook_text:data.get('hook_text').trim() || null, commentary:data.get('commentary').trim() || null};
  }

  const isPodcastClip = clip => !!clip.podcast;
  const edits = list => (list || []).map(e => `${e.find} → ${e.replace}`).join('\n');
  function defaults(clip) {
    if (!clip.podcast) return null;
    const p = clip.podcast;
    return {layout:p.layout, version:p.version, speakers:p.speakers ?? null, hook_text:p.hook_text ?? null, commentary:p.commentary ?? null, caption_edits:p.caption_edits || []};
  }
  function read(node, clip) {
    if (!clip.podcast) return null;
    const value = name => node.querySelector(`[data-podcast=${name}]`).value.trim();
    const caption_edits = value('caption_edits').split('\n').map(line => line.split(/\s*(?:→|->)\s*/)).filter(([find]) => find?.trim()).map(([find, replace='']) => ({find:find.trim(), replace:replace.trim()}));
    return {layout:value('layout'), version:value('version'), speakers:clip.podcast.speakers ?? null, hook_text:value('hook_text') || null, commentary:value('commentary') || null, caption_edits};
  }
  function clipHtml(clip, draft, disabled) {
    if (!clip.podcast) return '';
    const p = draft.podcast || defaults(clip), a = clip.podcast_analysis || {}, t = clip.tracking_result || {};
    const framing = t.mode === 'podcast' ? `${LAYOUTS[t.layout] || t.layout}${t.layout!==t.auto_layout&&t.auto_layout?` (auto picks ${LAYOUTS[t.auto_layout] || t.auto_layout})`:''} · ${t.switches} camera switches · ${t.cutaways} listener reactions` : t.status==='unavailable' ? 'Speaker framing unavailable; rendered centred' : '';
    return `<fieldset class="podcast-clip" ${disabled}><legend>Podcast</legend>
      <label>Speaker framing<select data-podcast="layout">${options(LAYOUTS, p.layout)}</select></label>
      <label>Editing version<select data-podcast="version">${options(VERSIONS, p.version)}</select></label>
      <label>Opening hook text<input data-podcast="hook_text" maxlength="64" value="${esc(p.hook_text)}" placeholder="${esc(clip.title)}"></label>
      <label>AI commentary <span class="hint">Version B only · labelled “AI commentary”</span><textarea data-podcast="commentary" maxlength="160" placeholder="${esc(a.commentary)}">${esc(p.commentary)}</textarea></label>
      <label>Subtitle fixes <span class="hint">one per line: wrong → right</span><textarea data-podcast="caption_edits" placeholder="clok → clock">${esc(edits(p.caption_edits))}</textarea></label>
      ${framing?`<p class="hint">${esc(framing)}</p>`:''}${t.needs_review?'<p class="hint">Faces didn’t clearly match the voices — check the framing.</p>':''}
      ${clip.status==='ready'?'<div class="actions"><button type="button" data-action="description">Copy description</button><span class="hint description-status" role="status"></span></div>':''}</fieldset>`;
  }
  function summary(clip) {
    const a = clip.podcast_analysis || {};
    return `<div class="samples">${a.hook?.text?`<span class="sample">“${esc(a.hook.text)}”</span>`:''}${(a.speakers||[]).map(s=>`<span class="sample">${esc(s.label)} · ${Math.round(s.seconds)}s</span>`).join('')}</div>`;
  }
  document.addEventListener('click', async event => {
    const button = event.target.closest('[data-action=description]');
    if (!button) return;
    const status = button.parentElement.querySelector('.description-status');
    try {
      const response = await fetch(`./api/jobs/${jobId()}/clips/${button.closest('.clip').dataset.id}/description`, {cache:'no-store'});
      if (!response.ok) throw Error('Could not load the description.');
      const {text} = await response.json();
      await navigator.clipboard.writeText(text);
      status.textContent = 'Copied with source attribution.';
    } catch (error) { status.textContent = error.message; }
  });

  const dimension = (m, filter) => filter === 'retention' ? m.podcast?.dimensions?.retention : m.podcast?.dimensions?.[filter];
  return {type, source, request, isPodcast, exportable, drawRights, momentHtml, readMoment, isPodcastClip, defaults, read, clipHtml, summary, dimension,
          filters:{all:'Ranked (all)',entertaining:'Most entertaining',educational:'Most educational',surprising:'Most surprising',emotional:'Most emotional',controversial:'Most controversial',retention:'Highest retention'}};
})();
