(() => {
  const form = document.querySelector('#branding-settings');
  const text = form.elements.namedItem('text');
  const opacity = form.elements.namedItem('opacity');
  const size = form.elements.namedItem('size');
  const position = form.elements.namedItem('position');
  const file = form.elements.namedItem('logo');
  const remove = form.elements.namedItem('remove-logo');
  const sample = document.querySelector('.branding-sample');
  const mark = sample.querySelector('.branding-mark');
  const textPreview = mark.querySelector('span');
  const logo = mark.querySelector('img');
  const status = document.querySelector('#branding-status');
  const canvas = document.createElement('canvas').getContext('2d');

  async function api(path = '', options) {
    const response = await fetch(`./api/branding${path}`, options);
    const result = await response.json();
    if (!response.ok) {
      throw new Error(typeof result.detail === 'string' ? result.detail : 'Check the branding settings and try again.');
    }
    return result;
  }

  function updatePreview() {
    form.elements.namedItem('opacity-value').value = `${opacity.value}%`;
    form.elements.namedItem('size-value').value = size.value;
    textPreview.textContent = text.value.trim().replace(/\s+/g, ' ');
    canvas.font = '100px "DejaVu Sans", sans-serif';
    const width = canvas.measureText(textPreview.textContent).width;
    mark.style.setProperty('--text-fit', width ? `${5000 / width}cqw` : '100cqw');
    mark.style.setProperty('--branding-size', Number(size.value) / 10.8);
    mark.style.setProperty('--branding-opacity', Number(opacity.value) / 100);
    sample.dataset.position = position.value === 'auto' ? 'top-left' : position.value;
  }

  function showLogo(hasLogo) {
    if (hasLogo) logo.src = `./api/branding/logo?v=${Date.now()}`;
    else logo.removeAttribute('src');
    logo.hidden = !hasLogo;
    textPreview.hidden = hasLogo;
    remove.hidden = !hasLogo;
  }

  async function edit(action) {
    const buttons = [...form.querySelectorAll('button')];
    for (const field of [...buttons, file]) field.disabled = true;
    try {
      status.textContent = await action();
    } catch (error) {
      status.textContent = error.message;
    } finally {
      for (const field of [...buttons, file]) field.disabled = false;
    }
  }

  form.addEventListener('input', updatePreview);
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    edit(async () => {
      await api('', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
        text: text.value, opacity: Number(opacity.value), size: Number(size.value), position: position.value,
      })});
      return 'Branding saved for new renders.';
    });
  });
  file.addEventListener('change', () => {
    const selected = file.files[0];
    file.value = '';
    if (!selected) return;
    edit(async () => {
      const state = await api('/logo', {method: 'PUT', body: selected});
      showLogo(state.has_logo);
      return 'PNG logo saved for new renders.';
    });
  });
  remove.addEventListener('click', () => edit(async () => {
    const state = await api('/logo', {method: 'DELETE'});
    showLogo(state.has_logo);
    return 'Logo removed. New renders will use the watermark text.';
  }));
  edit(async () => {
    const state = await api();
    text.value = state.text;
    opacity.value = state.opacity;
    size.value = state.size;
    position.value = state.position;
    showLogo(state.has_logo);
    updatePreview();
    return '';
  });
})();
