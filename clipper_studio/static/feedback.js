const page = document.querySelector('#performance');
const form = page.querySelector('#publication-form');
const importForm = page.querySelector('#performance-import');
const chooser = form.elements.clip;
const metricSelect = page.querySelector('#performance-compare');
const status = page.querySelector('#performance-status');
const endpoint = './api/feedback';
const metricLabels = {
  views: 'Views', engaged_views: 'Engaged views', viewed_percent: 'Viewed (%)',
  swiped_away_percent: 'Swiped away (%)', average_view_duration: 'Average view duration (s)',
  average_percentage_viewed: 'Average percentage viewed (%)', likes: 'Likes', comments: 'Comments',
  shares: 'Shares', subscribers_gained: 'Subscribers gained',
};
const textFields = ['publish_url', 'publish_date', 'title', 'hook', 'hook_style'];
const ratios = ['viewed_percent', 'swiped_away_percent'];
const decimal = [...ratios, 'average_view_duration', 'average_percentage_viewed'];
const fmt = new Intl.NumberFormat(undefined, {maximumFractionDigits: 3});
const number = value => value == null ? 'Not recorded' : fmt.format(value);
let clips = [], records = [], editing = null, revision = 0, comparisonRevision = 0;

function node(tag, text, cls) {
  const element = document.createElement(tag);
  if (text != null) element.textContent = text;
  if (cls) element.className = cls;
  return element;
}

function facts(entries) {
  const list = node('dl');
  for (const [label, value] of entries) {
    const item = node('div');
    item.append(node('dt', label), node('dd', value));
    list.append(item);
  }
  return list;
}

function videoLink(url, label) {
  const link = node('a', label);
  if (!/^https:\/\/(?:www\.|m\.)?(?:youtube\.com|youtu\.be)(?:\/|$)/.test(url || '')) return null;
  link.href = url;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  return link;
}

for (const [name, label] of Object.entries(metricLabels)) {
  const field = node('label', label);
  const input = node('input');
  input.type = 'number';
  input.name = name;
  input.min = 0;
  input.step = decimal.includes(name) ? 'any' : '1';
  if (ratios.includes(name)) input.max = 100;
  field.append(input);
  page.querySelector('#performance-metrics').append(field);
  const option = node('option', label);
  option.value = name;
  metricSelect.append(option);
}
const retentionLabel = node('label', 'Audience retention graph (JSON points)', 'retention-field');
const retention = node('textarea');
retention.name = 'retention_graph';
retention.placeholder = '[{"position_percent":0,"retained_percent":100},{"position_percent":50,"retained_percent":72}]';
retentionLabel.append(retention, node('small', 'Use position within the video (0–100%) and the measured retention percentage.', 'hint'));
page.querySelector('#performance-metrics').append(retentionLabel);
metricSelect.value = 'average_percentage_viewed';

async function request(path = '', method = 'GET', data) {
  const response = await fetch(endpoint + path, {
    method, headers: data ? {'Content-Type': 'application/json'} : {},
    body: data ? JSON.stringify(data) : undefined, cache: 'no-store',
  });
  const payload = await response.json();
  if (!response.ok) {
    const detail = payload.detail;
    throw new Error(Array.isArray(detail) ? detail.map(e => e.loc.slice(1).join('.') + ': ' + e.msg).join('; ')
      : detail || 'Unable to load performance data.');
  }
  return payload;
}

function selection() {
  return clips.find(clip => clipKey(clip) === chooser.value);
}

function clipKey(clip) {
  return `${clip.job_id}/${clip.clip_id}`;
}

function integrity(clip) {
  const wrap = node('details');
  wrap.open = true;
  wrap.append(node('summary', 'Source integrity & disclosure'));
  const {source} = clip;
  const context = node('p', [source.title || 'Original recording', source.channel, source.start != null
    ? `${number(source.start)}–${number(source.end)}s in source` : null].filter(Boolean).join(' · '), 'hint');
  const link = videoLink(source.url, 'Original video ↗');
  if (link) context.append(' · ', link);
  wrap.append(context, node('p', 'Advisory checks only — not copyright clearance or platform approval.', 'hint'));
  const list = node('ul');
  for (const flag of clip.integrity) {
    const item = node('li');
    item.append(node('strong', flag.check + ': '), flag.detail);
    list.append(item);
  }
  wrap.append(list);
  return wrap;
}

function showSelection() {
  const clip = editing || selection();
  const target = page.querySelector('#publication-integrity');
  target.replaceChildren();
  if (clip) {
    target.append(node('p', `Job ID: ${clip.job_id} · Clip ID: ${clip.clip_id}`, 'hint'), integrity(clip));
  }
  if (!editing) {
    form.elements.title.value = clip?.title || '';
    form.elements.hook.value = clip?.hook || '';
    form.elements.hook_style.value = clip?.hook_style || '';
  }
}

function recordForm(record = null, selected = '') {
  editing = record;
  form.reset();
  const key = record ? clipKey(record) : selected;
  chooser.replaceChildren(new Option(clips.length ? 'Select a rendered highlight' : 'No rendered highlights', ''));
  for (const clip of clips) chooser.add(new Option(`${clip.title} · ${clip.source.title || clip.job_id} · ${number(clip.duration)}s`, clipKey(clip)));
  if (record && !clips.some(clip => clipKey(clip) === key)) chooser.add(new Option(record.title || 'Removed highlight', key));
  chooser.value = key;
  chooser.disabled = !!record;
  const confirm = page.querySelector('#confirm-publication-label');
  confirm.hidden = form.elements.confirmed_published.disabled = !!record;
  form.querySelector('button[type="submit"]').disabled = !clips.length && !record;
  form.querySelector('button[type="submit"]').textContent = record ? 'Save changes' : 'Mark as published';
  page.querySelector('#cancel-publication').hidden = !record;
  page.querySelector('#publication-heading').textContent = record ? 'Edit published highlight' : 'Record a published highlight';
  page.querySelector('#publication-metrics').open = !!record;
  if (record) {
    for (const key of textFields) form.elements[key].value = record[key] ?? '';
    for (const name of Object.keys(metricLabels)) form.elements[name].value = record.metrics[name] ?? '';
    retention.value = record.metrics.retention_graph ? JSON.stringify(record.metrics.retention_graph) : '';
  }
  showSelection();
}

function drawCalibration(summary) {
  const target = page.querySelector('#performance-calibration');
  target.replaceChildren(node('p', `${summary.sample_size} of ${summary.record_count} records include average percentage viewed. `
    + `${summary.component_sample_size} also include editorial components. At least ${summary.minimum_sample} comparable records are needed per component or group.`));
  for (const [key, label] of Object.entries({durations:'Duration', hook_styles:'Hook style', editing_modes:'Editing mode'})) {
    target.append(node('h3', label));
    if (summary[key].length) {
      target.append(facts(summary[key].map(item => [item.label, `${number(item.median_percentage_viewed)}% median · n=${item.sample_size}`])));
    } else {
      target.append(node('p', 'Not enough measured data in any group.', 'hint'));
    }
  }
}

function retentionChart(points) {
  const frame = node('figure', null, 'performance-retention');
  const upper = Math.max(100, ...points.map(point => point.retained_percent));
  frame.append(node('figcaption', `Audience retention · horizontal: video 0–100% · vertical: retained 0–${number(upper)}%`, 'hint'));
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 320 90');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Audience retention over the video timeline');
  const path = document.createElementNS(svg.namespaceURI, 'polyline');
  const coordinate = point => [5 + point.position_percent * 3.1, 85 - point.retained_percent / upper * 80];
  path.setAttribute('points', points.map(point => coordinate(point).join(',')).join(' '));
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', 'currentColor');
  path.setAttribute('stroke-width', '2');
  svg.append(path);
  for (const point of points) {
    const dot = document.createElementNS(svg.namespaceURI, 'circle');
    const [x, y] = coordinate(point);
    dot.setAttribute('cx', x);
    dot.setAttribute('cy', y);
    dot.setAttribute('r', points.length > 100 ? '1' : '2');
    dot.setAttribute('fill', 'currentColor');
    svg.append(dot);
  }
  frame.append(svg);
  return frame;
}

function comparisonTable(result) {
  const wrap = node('div', null, 'performance-table');
  const table = node('table');
  table.append(node('caption', 'Published highlights and measured results'));
  const head = table.createTHead().insertRow();
  const showDirector = records.some(record => record.director_score?.overall != null);
  const labels = ['Highlight', 'AI editorial estimate'];
  if (showDirector) labels.push('AI Director render estimate');
  labels.push(metricLabels[result.metric], 'Measured vs. editorial percentile');
  for (const label of labels) {
    const cell = node('th', label);
    cell.scope = 'col';
    head.append(cell);
  }
  const body = table.createTBody();
  for (const record of records) {
    const row = body.insertRow();
    const score = record.editorial_score == null ? 'Unavailable' : `${number(record.editorial_score)} / 100`;
    const gap = result.percentile_gaps[record.id];
    const rank = gap == null ? 'Unavailable' : gap === 0 ? 'Same rank' : `${number(Math.abs(gap))} points ${gap > 0 ? 'above' : 'below'}`;
    const header = node('th', record.title || 'Published highlight');
    header.scope = 'row';
    const values = [score];
    if (showDirector) values.push(record.director_score?.overall == null ? 'Unavailable' : `${number(record.director_score.overall)} / 100`);
    values.push(number(record.metrics[result.metric]), rank);
    row.append(header, ...values.map(value => node('td', value)));
  }
  wrap.tabIndex = 0;
  wrap.setAttribute('role', 'region');
  wrap.setAttribute('aria-label', 'Editorial estimates and measured performance comparison');
  wrap.append(table);
  return wrap;
}

function renderRecord(record) {
  const card = node('article', null, 'performance-record');
  card.append(node('h3', record.title || 'Published highlight'));
  const link = videoLink(record.publish_url, 'Published video ↗');
  if (link) card.append(link);
  card.append(node('p', [record.publish_date, record.detection_sources.join(' + '), record.editing_mode,
    record.duration != null ? `${number(record.duration)}s${record.duration_basis === 'source timeline' ? ' source timeline' : ''}` : null].filter(Boolean).join(' · '), 'hint'));
  if (record.hook) card.append(node('p', `Hook: ${record.hook}${record.hook_style ? ` · ${record.hook_style}` : ''}`));
  const details = node('details');
  details.append(node('summary', 'Metrics, editorial snapshot & integrity'));
  const present = Object.keys(metricLabels).filter(key => record.metrics[key] != null);
  details.append(present.length ? facts(present.map(key => [metricLabels[key], number(record.metrics[key])]))
    : node('p', 'No metrics recorded.', 'hint'));
  if (record.metrics.retention_graph?.length) details.append(retentionChart(record.metrics.retention_graph));
  if (Object.keys(record.score_breakdown).length) details.append(facts(Object.entries(record.score_breakdown).map(([k,v]) => [k, number(v)])));
  if (Object.keys(record.director_score || {}).length) {
    details.append(node('h4', 'AI Director render estimate'),
      facts(Object.entries(record.director_score).map(([key, value]) => [key, number(value)])));
  }
  if (record.score_explanation) details.append(node('p', record.score_explanation));
  details.append(integrity(record));
  card.append(details);
  const actions = node('div', null, 'performance-actions');
  const edit = node('button', 'Edit record');
  edit.addEventListener('click', () => {
    recordForm(record);
    form.elements.publish_url.focus();
  });
  const remove = node('button', 'Delete record', 'quiet');
  remove.addEventListener('click', async () => {
    if (!globalThis.confirm('Delete this local performance record? The published video will not change.')) return;
    await action(remove, async () => {
      await request('/' + record.id, 'DELETE');
      if (editing?.id === record.id) recordForm();
      await reloadRecords();
      status.textContent = 'Local record deleted.';
    });
  });
  actions.append(edit, remove);
  card.append(actions);
  return card;
}

async function drawComparison() {
  const version = ++comparisonRevision;
  const result = await request('/comparison?metric=' + encodeURIComponent(metricSelect.value));
  if (version !== comparisonRevision || page.hidden) return;
  const target = page.querySelector('#performance-comparison');
  target.replaceChildren(node('p', result.correlation == null
    ? `Matched clips: ${result.sample_size}. Correlation needs at least five matched clips and variation in both values.`
    : `Pearson correlation: ${number(result.correlation)} (n=${result.sample_size}). Relative ranks describe this sample only.`, 'hint'));
  const available = Object.entries(result.components).filter(([, value]) => value != null);
  if (available.length) target.append(facts(available.map(([key,value]) => [key + ' correlation', number(value)])));
  if (result.director.sample_size) {
    target.append(node('h3', 'AI Director render comparison'), node('p', result.director.correlation == null
      ? `Matched rendered clips: ${result.director.sample_size}. Correlation needs at least five matches and variation in both values.`
      : `Pearson correlation: ${number(result.director.correlation)} (n=${result.director.sample_size}).`, 'hint'));
    const components = Object.entries(result.director.components).filter(([, value]) => value != null);
    if (components.length) target.append(facts(components.map(([key, value]) => [key + ' correlation', number(value)])));
  }
  if (records.length) target.append(comparisonTable(result));
  page.querySelector('#publication-records').replaceChildren(...(records.length ? records.map(renderRecord)
    : [node('p', 'No performance records yet. Mark an already-published highlight or import metrics.', 'hint')]));
}

async function reloadRecords() {
  const version = revision;
  const [items, summary] = await Promise.all([request(), request('/calibration')]);
  if (version !== revision || page.hidden) return;
  records = items;
  drawCalibration(summary);
  await drawComparison();
}

async function action(button, work) {
  if (button) button.disabled = true;
  status.textContent = '';
  try { await work(); }
  catch (error) { status.textContent = error.message; }
  finally { if (button) button.disabled = false; }
}

async function route() {
  const version = ++revision;
  const match = location.hash.match(/^#performance(?:\/([a-zA-Z0-9]+)\/([a-zA-Z0-9]+))?$/);
  const link = document.querySelector('#performance-link');
  if (!match) { link.removeAttribute('aria-current'); return; }
  link.setAttribute('aria-current', 'page');
  await action(null, async () => {
    const items = await request('/clips');
    if (version !== revision) return;
    clips = items;
    recordForm(null, match[1] ? `${match[1]}/${match[2]}` : '');
    await reloadRecords();
  });
}

chooser.addEventListener('change', showSelection);
metricSelect.addEventListener('change', () => action(null, drawComparison));
page.querySelector('#cancel-publication').addEventListener('click', () => recordForm());
form.addEventListener('submit', event => {
  event.preventDefault();
  action(event.submitter, async () => {
    const clip = selection();
    if (!editing && !clip) throw new Error('Select a rendered highlight.');
    const data = {};
    for (const key of textFields) {
      const value = form.elements[key].value.trim();
      if (value || editing) data[key] = value || null;
    }
    data.metrics = {};
    for (const key of Object.keys(metricLabels)) {
      const input = form.elements[key];
      if (input.value || editing) data.metrics[key] = input.value ? input.valueAsNumber : null;
    }
    try { if (retention.value.trim()) data.metrics.retention_graph = JSON.parse(retention.value); }
    catch { throw new Error('Retention graph must be valid JSON; see the example in the field.'); }
    if (editing && !retention.value.trim()) data.metrics.retention_graph = null;
    if (editing) await request('/' + editing.id, 'PATCH', data);
    else await request('', 'POST', {...data, job_id:clip.job_id, clip_id:clip.clip_id, confirmed_published:form.elements.confirmed_published.checked});
    recordForm();
    await reloadRecords();
    status.textContent = 'Performance record saved. No video was uploaded.';
  });
});
importForm.addEventListener('submit', event => {
  event.preventDefault();
  action(event.submitter, async () => {
    const file = importForm.elements.file.files[0];
    if (!file || file.size > 1_000_000) throw new Error('Select a CSV or JSON file up to 1 MB.');
    const format = file.name.split('.').at(-1).toLowerCase();
    if (!['json', 'csv'].includes(format)) throw new Error('Use a CSV or JSON file.');
    const clip = selection();
    const imported = await request('/import', 'POST', {format, content:await file.text(),
      job_id:clip?.job_id, clip_id:clip?.clip_id, confirmed_published:importForm.elements.confirmed_published.checked});
    importForm.reset();
    await reloadRecords();
    status.textContent = `Imported ${imported.length} performance record(s). No videos were uploaded.`;
  });
});
globalThis.addEventListener('hashchange', route);
route();
