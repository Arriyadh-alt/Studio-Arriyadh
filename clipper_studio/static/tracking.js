// Reaction tracking controls: preview the AI target and crop, pick another target, switch layouts.
globalThis.TrackingControls = (() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const LAYOUTS = {auto:'Auto', gameplay:'Gameplay focus', reaction:'Reaction focus', split:'Split screen'};
  const defaults = value => ({enabled:true, layout:'auto', follow:'auto', x:null, target:null, ...value});
  const time = s => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
  const pathAt = (path, t) => {
    if(!path?.length) return .5;
    if(t <= path[0][0]) return path[0][1];
    for(let i=1;i<path.length;i++) if(t <= path[i][0]) {
      const [t0,x0]=path[i-1],[t1,x1]=path[i];
      return x0 + (x1-x0) * (t-t0) / Math.max(t1-t0, 1e-6);
    }
    return path[path.length-1][1];
  };
  function status(result) {
    if(!result || result.status==='disabled') return 'Reaction tracking off';
    if(result.status==='unavailable') return 'Tracking unavailable for this render — center crop used';
    if(result.status==='not_cropped') return 'Blurred background shows the whole frame; tracking applies to Center crop';
    const t=result.trigger, parts=[];
    if(t) parts.push(`Trigger ${time(t.trigger)} → noticed ${time(t.notice)} → reaction ${time(t.reaction)}`);
    parts.push(`${LAYOUTS[result.layout]||result.layout}${result.auto_layout?' (auto)':''}`);
    if(result.attempts>1) parts.push(`reframed ${result.attempts}×`);
    return parts.join(' · ');
  }
  function checks(result) {
    const c=result?.validation?.checks; if(!c) return '';
    const labels={trigger_visible:'Trigger visible',facecam_clear:'Facecam clear',subject_in_frame:'Subject in frame',camera_follows:'Camera follows action',reaction_understandable:'Reaction readable',story_complete:'Setup → outcome kept'};
    return `<ul class="tracking-checks">${Object.entries(labels).filter(([k])=>k in c).map(([k,l])=>`<li class="${c[k]?'pass':'warn'}">${c[k]?'✓':'!'} ${l}</li>`).join('')}</ul>`;
  }
  function html(clip, draft) {
    const t=defaults(draft.tracking), result=clip.tracking_result;
    const review=result?.needs_review ? '<p class="error">Needs manual review: the trigger could not be kept in frame automatically. Click what the streamer reacts to, or change the layout.</p>' : '';
    const still=clip.tracking_still_url ? `<div class="tracking-editor" data-still-time="${result?.still_time ?? 0}"><img draggable="false" alt="Source frame at the reaction trigger" src=".${esc(clip.tracking_still_url)}?v=${clip.rendered_at||0}"><div class="crop-window"></div><span class="target-dot ai" title="AI target"></span><span class="target-dot manual" hidden title="Your target"></span></div>
      <p class="hint">Blue box = 9:16 crop · dot = tracked target. Click the object or character to follow instead.</p>
      <div class="tracking-preview-wrap"><canvas class="tracking-preview" width="216" height="384" aria-label="Final layout preview"></canvas><p class="hint">Final layout preview at the trigger</p></div>` : '<p class="hint">Tracking preview available after rendering.</p>';
    return `<fieldset class="tracking-options" data-target='${esc(JSON.stringify(t.target))}' ${['queued','rendering'].includes(clip.status)?'disabled':''}><legend>Reaction tracking</legend>
      <label class="check"><input type="checkbox" data-tracking="enabled" ${t.enabled?'checked':''}> Show what the streamer reacts to</label>
      <label>Layout<select data-tracking="layout">${Object.entries(LAYOUTS).map(([k,l])=>`<option value="${k}" ${t.layout===k?'selected':''}>${l}</option>`).join('')}</select></label>
      <label>Camera<select data-tracking="follow"><option value="auto" ${t.follow==='auto'?'selected':''}>Follow the action</option><option value="fixed" ${t.follow==='fixed'?'selected':''}>Fixed crop position</option></select></label>
      <label>Crop position<input type="range" data-tracking="x" min="0" max="1" step="0.01" value="${t.x ?? result?.target?.[0] ?? .5}"></label>
      <p class="hint tracking-status">${esc(status(result))}</p>${checks(result)}${review}
      <details ${result?.needs_review?'open':''}><summary>Preview tracking</summary>${still}
      <button type="button" data-action="tracking-reset">Back to automatic</button></details></fieldset>`;
  }
  function read(node, previous) {
    const field=$('.tracking-options',node); if(!field) return defaults(previous);
    const follow=$('[data-tracking=follow]',field).value;
    return {enabled:$('[data-tracking=enabled]',field).checked, layout:$('[data-tracking=layout]',field).value, follow,
      x:follow==='fixed'?Number($('[data-tracking=x]',field).value):null, target:JSON.parse(field.dataset.target||'null')};
  }
  function draw(field, clip) {
    const editor=$('.tracking-editor',field), img=editor && $('img',editor);
    if(!img?.naturalWidth) return;
    const result=clip.tracking_result||{}, t=read(field.closest('.clip'), clip.tracking);
    const aspect=img.naturalWidth/img.naturalHeight, still=Number(editor.dataset.stillTime)||0;
    const cam=clip.facecam_result?.status==='used' ? clip.facecam_result.box : null;
    const layout=t.layout==='auto' ? (result.layout||'gameplay') : (cam ? t.layout : 'gameplay');
    const band=layout==='split' ? 700 : 0;
    const width=Math.min(1,(1080/(1920-band))/aspect);
    let x = t.follow==='fixed' ? t.x : t.target ? t.target.x : pathAt(result.path, still);
    x=Math.max(width/2,Math.min(1-width/2,x));
    Object.assign($('.crop-window',editor).style,{left:`${(x-width/2)*100}%`,width:`${width*100}%`});
    const ai=$('.target-dot.ai',editor), target=result.target;
    ai.hidden=!target; if(target) Object.assign(ai.style,{left:`${target[0]*100}%`,top:`${target[1]*100}%`});
    const manual=$('.target-dot.manual',editor); manual.hidden=!t.target;
    if(t.target) Object.assign(manual.style,{left:`${t.target.x*100}%`,top:`${t.target.y*100}%`});
    const canvas=$('.tracking-preview',field), ctx=canvas.getContext('2d'), W=canvas.width, H=canvas.height;
    const iw=img.naturalWidth, ih=img.naturalHeight, scale=W/1080;
    ctx.fillStyle='#000'; ctx.fillRect(0,0,W,H);
    const bandH=band*scale;
    ctx.drawImage(img,(x-width/2)*iw,0,width*iw,ih,0,bandH,W,H-bandH);
    if(cam) {
      const sx=cam.x*iw, sy=cam.y*ih, sw=cam.w*iw, sh=cam.h*ih;
      if(layout==='split') {
        const k=W/(width*iw), mx=(sx-(x-width/2)*iw)*k, my=bandH+sy*(H-bandH)/ih;
        ctx.save(); ctx.beginPath(); ctx.rect(mx,my,sw*k,sh*(H-bandH)/ih); ctx.clip(); ctx.filter='blur(14px)';
        ctx.drawImage(img,(x-width/2)*iw,0,width*iw,ih,0,bandH,W,H-bandH); ctx.restore();
        const r=W/bandH, cw=Math.min(sw,sh*r), ch=cw/r;
        ctx.drawImage(img,sx+(sw-cw)/2,sy+(sh-ch)/2,cw,ch,0,0,W,bandH);
      } else {
        const size=(layout==='reaction'?.6:(clip.facecam_result.size||.42))*W, h=size*sh/sw;
        const right=(clip.facecam_result.position||'').endsWith('right'), bottom=(clip.facecam_result.position||'').startsWith('bottom');
        const ox=right?W-size-24*scale:24*scale, oy=bottom?(1600*scale-h):24*scale;
        ctx.drawImage(img,sx,sy,sw,sh,ox,oy,size,h);
        ctx.strokeStyle='rgba(255,255,255,.9)'; ctx.lineWidth=2; ctx.strokeRect(ox,oy,size,h);
      }
    }
  }
  const clipOf = node => (typeof current!=='undefined' && current) ? current.clips.find(c=>c.id===node.dataset.id) : null;
  const refresh = field => { const node=field.closest('.clip'), clip=node&&clipOf(node); if(clip) draw(field, clip); };
  document.addEventListener('load', event => {
    if(event.target.matches?.('.tracking-editor img')) refresh(event.target.closest('.tracking-options'));
  }, true);
  for(const type of ['input','change']) document.addEventListener(type, event => {
    const field=event.target.closest?.('.tracking-options'); if(field) refresh(field);
  });
  document.addEventListener('click', event => {
    const field=event.target.closest('.tracking-options'); if(!field || field.disabled) return;
    if(event.target.closest('[data-action=tracking-reset]')) {
      field.dataset.target='null';
      $('[data-tracking=layout]',field).value='auto'; $('[data-tracking=follow]',field).value='auto'; $('[data-tracking=enabled]',field).checked=true;
    } else if(event.target.closest('.tracking-editor')) {
      const editor=event.target.closest('.tracking-editor'), b=editor.getBoundingClientRect(), clamp=v=>Math.round(Math.max(0,Math.min(1,v))*1e4)/1e4;
      field.dataset.target=JSON.stringify({t:Number(editor.dataset.stillTime)||0, x:clamp((event.clientX-b.left)/b.width), y:clamp((event.clientY-b.top)/b.height)});
      $('[data-tracking=follow]',field).value='auto';
    } else return;
    $('[data-tracking=layout]',field).dispatchEvent(new Event('change',{bubbles:true}));
  });
  return {defaults, html, read, draw};
})();
