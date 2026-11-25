import { isDifferentPublishedVideo, setPublicationMetricsAvailable } from '../clipper_studio/static/feedback_url.js';

Deno.test('editing only another YouTube video ID hides the existing metrics', () => {
  const saved = 'https://www.youtube.com/watch?v=AbCdEf12345';
  for (const url of ['https://youtu.be/AbCdEf12345?si=test', 'https://www.youtube.com/shorts/AbCdEf12345']) {
    if (isDifferentPublishedVideo(saved, url)) throw new Error('Same video should keep the existing metrics');
  }
  for (const url of ['https://m.youtube.com/watch?v=AbCdEf12346', 'https://www.youtube.com.example/shorts/AbCdEf12345', '']) {
    if (!isDifferentPublishedVideo(saved, url)) throw new Error('Existing metrics should not be offered for a different link');
  }
  if (isDifferentPublishedVideo(null, saved)) throw new Error('A new record should accept metric entry');
});

Deno.test('hidden metrics are disabled until the user returns to the original video', () => {
  const panel = {hidden: false};
  const fieldset = {disabled: false};
  setPublicationMetricsAvailable(false, panel, fieldset);
  if (!panel.hidden || !fieldset.disabled) throw new Error('Hidden fields must not block form validation');
  setPublicationMetricsAvailable(true, panel, fieldset);
  if (panel.hidden || fieldset.disabled) throw new Error('Returning to the original video must re-enable the fields');
});
