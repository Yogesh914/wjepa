/* Host-page helper: resize only the explicitly marked W-JEPA figure frames. */
(() => {
  'use strict';
  window.addEventListener('message', event => {
    const data = event.data;
    if (!data || data.type !== 'wjepa:resize' || !Number.isFinite(data.height)) return;
    const frames = document.querySelectorAll('iframe[data-wjepa-figure]');
    for (const frame of frames) {
      if (event.source !== frame.contentWindow) continue;
      const expectedOrigin = new URL(frame.src, document.baseURI).origin;
      if (event.origin !== expectedOrigin) continue;
      const height = Math.ceil(data.height);
      if (height < 200 || height > 10000) return;
      frame.style.height = height + 'px';
    }
  });
})();
