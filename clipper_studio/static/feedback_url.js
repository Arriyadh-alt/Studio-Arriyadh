export function publishedVideoId(value) {
  try {
    const url = new URL(value.trim());
    if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password || url.port) return null;
    const parts = url.pathname.replace(/^\/+|\/+$/g, '').split('/');
    let id;
    if (url.hostname === 'youtu.be' && parts.length === 1) id = parts[0];
    if (['youtube.com', 'www.youtube.com', 'm.youtube.com'].includes(url.hostname)) {
      if (parts.length === 1 && parts[0] === 'watch') id = url.searchParams.get('v');
      if (parts.length === 2 && ['shorts', 'live'].includes(parts[0])) id = parts[1];
    }
    return /^[\w-]{11}$/.test(id || '') ? id : null;
  } catch {
    return null;
  }
}

export function isDifferentPublishedVideo(savedUrl, currentUrl) {
  return !!savedUrl && publishedVideoId(savedUrl) !== publishedVideoId(currentUrl);
}

export function setPublicationMetricsAvailable(available, panel, fieldset) {
  panel.hidden = fieldset.disabled = !available;
}
