Deno.test('untouched caption controls follow the mode without overriding custom colors', async () => {
  globalThis.document = new EventTarget();
  try {
    await import('../clipper_studio/static/captions.js');
    const controls = globalThis.CaptionControls;
    const fields = Object.fromEntries(Object.entries(controls.options().caption_style).map(([key, value]) => {
      const color = typeof value === 'string' && value.startsWith('#');
      return [key, {
        dataset: {caption:key},
        type: typeof value === 'boolean' ? 'checkbox' : color ? 'color' : 'text',
        checked: value,
        value: color ? value.toLowerCase() : String(value),
      }];
    }));
    const root = {
      querySelectorAll: () => Object.values(fields),
      querySelector: selector => fields[selector.match(/^\[data-caption=(\w+)\]$/)?.[1]],
    };
    controls.applyProfile(root, 'natural', 'full');
    const natural = controls.read(root).caption_style;
    if(natural.preset !== 'classic' || natural.emphasis || natural.font !== 'DejaVu Sans') {
      throw Error(`Expected Natural caption defaults; got ${JSON.stringify(natural)}`);
    }
    fields.text_color.value = '#c0ffee';
    controls.applyProfile(root, 'full', 'natural');
    if(controls.read(root).caption_style.preset !== 'classic' || fields.text_color.value !== '#c0ffee') {
      throw Error('Mode switch replaced customized captions');
    }
  } finally {
    delete globalThis.document;
    delete globalThis.CaptionControls;
  }
});
