globalThis.CaptionControls = (() => {
  const defaults = {preset:'pop',font:'Montserrat',size:80,position:'bottom',text_color:'#FFFFFF',highlight_color:'#FFE34D',emphasis_color:'#67F5B5',uppercase:false,emphasis:true};
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function options(clip = {}) {
    const style = {...defaults,...clip.caption_style};
    if(!clip.caption_style?.font && style.preset!=='pop')style.font='DejaVu Sans';
    return {language:clip.language || 'auto',caption_style:style};
  }
  function html(clip = {}) {
    const {language,caption_style:s} = options(clip);
    const languages = [['auto','Auto-detect'],['en','English'],['id','Indonesian']];
    if(!languages.some(([code])=>code===language))languages.push([language,language]);
    const select = (key, label, values, selected) => `<label>${label}<select data-caption="${key}">${values.map(([value,name])=>`<option value="${esc(value)}" ${selected===value?'selected':''}>${esc(name)}</option>`).join('')}</select></label>`;
    const color = (key, label) => `<label>${label}<input type="color" data-caption="${key}" value="${esc(s[key])}"></label>`;
    const check = (key,label) => `<label class="check"><input type="checkbox" data-caption="${key}" ${s[key]?'checked':''}>${label}</label>`;
    return `<details class="caption-settings"><summary>Caption style</summary><div class="caption-grid">
      ${select('language','Speech language',languages,language)}
      ${select('preset','Preset',[['pop','Pop · active word'],['classic','Classic · static'],['minimal','Minimal · fade']],s.preset)}
      ${select('font','Font',['Montserrat','Anton','Bebas Neue','DejaVu Sans','Liberation Sans','JetBrains Mono'].map(f=>[f,f]),s.font)}
      <label>Size <output data-caption-size>${Number(s.size)} px</output><input type="range" data-caption="size" min="40" max="130" value="${Number(s.size)}"></label>
      ${select('position','Position',[['bottom','Bottom · safe area'],['middle','Middle'],['top','Top · safe area']],s.position)}
      ${color('text_color','Text color')}${color('highlight_color','Active word')}${color('emphasis_color','Important words')}
      ${check('uppercase','Uppercase')}${check('emphasis','Emphasize keywords')}
      </div><div class="caption-sample" aria-label="Caption style preview"><p>That <span class="caption-active">was</span> <span class="caption-emphasis">AMAZING!</span></p></div><p class="hint">Preview is approximate. Re-render applies changes to the video.</p></details>`;
  }
  function read(root) {
    const result = options();
    root.querySelectorAll('[data-caption]').forEach(input => {
      const key = input.dataset.caption;
      const value = input.type==='checkbox' ? input.checked : key==='size' ? Number(input.value) : input.value;
      if(key==='language')result.language=value; else result.caption_style[key]=value;
    });
    return result;
  }
  function preview(root) {
    const s = read(root).caption_style, sample = root.querySelector('.caption-sample');
    if(!sample)return;
    sample.style.fontFamily=s.font;
    sample.style.fontSize=`${Math.max(10,Math.round(s.size*.14))}px`;
    sample.style.color=s.text_color;
    sample.style.textTransform=s.uppercase?'uppercase':'none';
    sample.style.fontWeight=s.font==='Montserrat'?'900':s.preset==='minimal'?'400':'800';
    sample.style.alignItems={top:'start',middle:'center',bottom:'end'}[s.position] || 'end';
    sample.querySelector('.caption-active').style.color=s.preset==='pop'?s.highlight_color:s.text_color;
    sample.querySelector('.caption-emphasis').style.color=s.emphasis?s.emphasis_color:s.text_color;
    root.querySelector('[data-caption-size]').textContent=`${s.size} px`;
  }
  document.addEventListener('input',event=>{
    if(!event.target.matches('[data-caption]'))return;
    const root=event.target.closest('.caption-settings');
    if(event.target.dataset.caption==='preset')root.querySelector('[data-caption=font]').value=event.target.value==='pop'?'Montserrat':'DejaVu Sans';
    preview(root);
  },true);
  return {options,html,read,preview};
})();
