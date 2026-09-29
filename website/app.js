/* The demo is self-contained: no remote dependencies, calls, or inference. */
(() => {
  'use strict';
  const D = window.WJEPA_DATA;
  const root = document.getElementById('wjepa-figure');
  if (!root) return;
  const $ = id => root.querySelector('#' + id);
  if (!D || D.schemaVersion !== 1) {
    root.textContent = 'The figure data could not be loaded.';
    root.setAttribute('role', 'alert');
    return;
  }
  const hasPca = Boolean($('scatter'));
  const hasWavelets = Boolean($('raw-signal'));
  const state = { split: 'window', progress: 0, playing: false, participant: 'all', trails: false, selected: null, hovered: null, band: 2 };
  const scatter = $('scatter');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const colors = { 14: '#007c83', 15: '#d46f32' };
  const signalColors = { 14: '#007c72', 15: '#c3632b' };
  const ranges = ['0–3.13', '3.13–6.25', '6.25–12.5', '12.5–25', '25–50'];
  const reference = $('scatter-reference');
  const plotViews = new Map();
  let frameId = null;
  let animationStart = null;
  let animationFrom = 0;
  let rendered = [];
  const animationDuration = 6500;
  const endpointHold = 1200;

  const current = () => D.splits[state.split];
  const points = () => current().points;
  const lerp = (a, b, t) => a + (b - a) * t;
  const coordinate = (point, t) => [lerp(point.from[0], point.to[0], t), lerp(point.from[1], point.to[1], t)];
  const className = label => D.classes[String(label)] || 'Another activity';
  const announce = text => { $('announcer').textContent = text; };

  function canvasSize(canvas) {
    const box = canvas.getBoundingClientRect();
    const width = Math.max(1, box.width);
    const height = Math.max(1, box.height);
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const context = canvas.getContext('2d');
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    return { context, width, height };
  }

  function ticks(min, max, count = 5) {
    const raw = (max - min) / Math.max(1, count);
    const power = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    const scaled = raw / power;
    const step = (scaled > 5 ? 10 : scaled > 2 ? 5 : scaled > 1 ? 2 : 1) * power;
    const values = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) values.push(Math.abs(v) < step * 1e-6 ? 0 : v);
    return values;
  }

  function number(value) {
    if (value === 0) return '0';
    const abs = Math.abs(value);
    if (abs >= 100) return value.toFixed(0);
    if (abs >= 1) return value.toFixed(1).replace(/\.0$/, '');
    return value.toFixed(abs < .01 ? 3 : 2).replace(/0+$/, '').replace(/\.$/, '');
  }

  function plotGeometry(width, height) {
    const all = points().flatMap(p => [p.from, p.to]);
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const [x, y] of all) { xMin = Math.min(xMin, x); xMax = Math.max(xMax, x); yMin = Math.min(yMin, y); yMax = Math.max(yMax, y); }
    const padX = (xMax - xMin) * .105, padY = (yMax - yMin) * .14;
    xMin -= padX; xMax += padX; yMin -= padY; yMax += padY;
    const left = width < 440 ? 45 : 57, right = 24, top = 25, bottom = 46;
    const innerWidth = width - left - right, innerHeight = height - top - bottom;
    const scale = Math.min(innerWidth / (xMax - xMin), innerHeight / (yMax - yMin));
    const mx = (xMin + xMax) / 2, my = (yMin + yMax) / 2;
    const cx = left + innerWidth / 2, cy = top + innerHeight / 2;
    return { left, top, right: width - right, bottom: height - bottom, width, height,
      x: v => cx + (v - mx) * scale, y: v => cy - (v - my) * scale,
      xMin: mx - innerWidth / scale / 2, xMax: mx + innerWidth / scale / 2,
      yMin: my - innerHeight / scale / 2, yMax: my + innerHeight / scale / 2 };
  }

  function mark(context, x, y, activity, radius, color, alpha = 1, outline = false) {
    context.globalAlpha = alpha;
    context.beginPath();
    if (activity === 14) context.arc(x, y, radius, 0, Math.PI * 2);
    else { context.moveTo(x, y - radius * 1.28); context.lineTo(x + radius * 1.1, y + radius * .83); context.lineTo(x - radius * 1.1, y + radius * .83); context.closePath(); }
    context.fillStyle = color;
    if (outline) { context.strokeStyle = color; context.lineWidth = 1.4; context.stroke(); }
    else context.fill();
    context.globalAlpha = 1;
  }

  function drawPlot(canvas, progress, moving) {
    const { width, height, context: ctx } = canvasSize(canvas);
    const g = plotGeometry(width, height);
    const visiblePoints = [];
    ctx.font = '12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    ctx.lineWidth = 1;
    ctx.textBaseline = 'middle';
    for (const v of ticks(g.xMin, g.xMax, width < 440 ? 3 : 5)) {
      const x = g.x(v);
      if (x < g.left + 4 || x > g.right - 4) continue;
      ctx.strokeStyle = v === 0 ? '#c9d6de' : '#e9eef2';
      ctx.setLineDash(v === 0 ? [] : [2, 5]);
      ctx.beginPath(); ctx.moveTo(x, g.top); ctx.lineTo(x, g.bottom); ctx.stroke();
      ctx.fillStyle = '#64748b'; ctx.textAlign = 'center'; ctx.fillText(number(v), x, g.bottom + 17);
    }
    for (const v of ticks(g.yMin, g.yMax, 4)) {
      const y = g.y(v);
      if (y < g.top + 4 || y > g.bottom - 4) continue;
      ctx.strokeStyle = v === 0 ? '#c9d6de' : '#e9eef2';
      ctx.setLineDash(v === 0 ? [] : [2, 5]);
      ctx.beginPath(); ctx.moveTo(g.left, y); ctx.lineTo(g.right, y); ctx.stroke();
      ctx.fillStyle = '#64748b'; ctx.textAlign = 'right'; ctx.fillText(number(v), g.left - 10, y);
    }
    ctx.setLineDash([]);
    ctx.strokeStyle = '#dce5ec'; ctx.strokeRect(g.left, g.top, g.right - g.left, g.bottom - g.top);
    ctx.fillStyle = '#475569'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
    const middle = moving && progress > 0 && progress < 1;
    ctx.fillText(middle ? 'Interpolated X' : 'PC1', g.right, g.bottom + 29);
    ctx.textAlign = 'left'; ctx.fillText(middle ? 'Interpolated Y' : 'PC2', g.left, 5);
    ctx.save(); ctx.beginPath(); ctx.rect(g.left, g.top, g.right - g.left, g.bottom - g.top); ctx.clip();

    const highlighted = state.participant !== 'all';
    points().forEach((p, i) => {
      const active = !highlighted || p.participant === Number(state.participant);
      const pos = coordinate(p, progress);
      const x = g.x(pos[0]), y = g.y(pos[1]);
      const radius = active ? (width < 440 ? 2.9 : 3.5) : 2.7;
      if (moving && state.trails && active) {
        ctx.strokeStyle = colors[p.activity]; ctx.globalAlpha = .14; ctx.lineWidth = .75;
        ctx.beginPath(); ctx.moveTo(g.x(p.from[0]), g.y(p.from[1])); ctx.lineTo(g.x(p.to[0]), g.y(p.to[1])); ctx.stroke();
        ctx.globalAlpha = 1;
      }
      mark(ctx, x, y, p.activity, radius, colors[p.activity], active ? .74 : .09);
      visiblePoints.push({ i, x, y, active, canvas });
    });
    const selected = state.hovered !== null ? state.hovered : state.selected;
    if (selected !== null && points()[selected]) {
      const p = points()[selected];
      const pos = coordinate(p, progress);
      const x = g.x(pos[0]), y = g.y(pos[1]);
      if (moving) {
        ctx.globalAlpha = .6; ctx.strokeStyle = colors[p.activity]; ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]);
        ctx.beginPath(); ctx.moveTo(g.x(p.from[0]), g.y(p.from[1])); ctx.lineTo(g.x(p.to[0]), g.y(p.to[1])); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
        mark(ctx, g.x(p.from[0]), g.y(p.from[1]), p.activity, 5.5, colors[p.activity], .75, true);
        mark(ctx, g.x(p.to[0]), g.y(p.to[1]), p.activity, 5.5, colors[p.activity], .75, true);
      }
      ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1.4; ctx.stroke();
      mark(ctx, x, y, p.activity, 4.5, colors[p.activity]);
    }
    ctx.restore();
    plotViews.set(canvas, { geometry: g, rendered: visiblePoints });
  }

  function drawScatter() {
    drawPlot(reference, 0, false);
    drawPlot(scatter, state.progress, true);
    rendered = plotViews.get(scatter).rendered;
  }

  function updateProgress(value) {
    state.progress = Math.max(0, Math.min(1, Number(value)));
    const percent = Math.round(state.progress * 100);
    $('progress').value = Math.round(state.progress * 1000);
    $('progress').style.setProperty('--progress', `${percent}%`);
    $('progress-value').value = `${percent}%`;
    $('progress-value').textContent = `${percent}%`;
    if (!state.playing) $('play-label').textContent = state.progress === 1 ? 'Replay transition' : 'Play transition';
    $('before').classList.toggle('active', state.progress === 0);
    $('after').classList.toggle('active', state.progress === 1);
    const start = state.progress === 0, end = state.progress === 1;
    $('stage-title').textContent = start ? 'JEPA' : end ? 'W-JEPA' : 'Transition to W-JEPA';
    $('stage-subtitle').textContent = start ? 'Raw temporal tokens' : end ? 'Multiscale wavelet tokens' : 'Illustrative interpolation';
    $('stage-tag').textContent = end ? 'Wavelet endpoint' : start ? 'Starting point' : `${percent}%`;
    $('baseline-variance').textContent = `${(current().metrics.no_wavelet.pcaVariance.reduce((a,b) => a+b,0) * 100).toFixed(1)}% variance explained`;
    $('variance').textContent = start || end
      ? `${(current().metrics[start ? 'no_wavelet' : 'wavelet'].pcaVariance.reduce((a,b) => a+b,0) * 100).toFixed(1)}% variance explained`
      : 'Endpoints only';
    $('progress').setAttribute('aria-valuetext', start ? 'JEPA, 0 percent' : end ? 'W-JEPA, 100 percent' : `Illustrative transition, ${percent} percent`);
    drawScatter();
  }

  function stop() {
    state.playing = false;
    if (frameId !== null) cancelAnimationFrame(frameId);
    frameId = null;
    animationStart = null;
    $('play').setAttribute('aria-pressed', 'false');
    $('play-symbol').textContent = '▶';
    $('play-label').textContent = state.progress === 1 ? 'Replay transition' : 'Play transition';
  }

  function tick(time) {
    if (!state.playing) return;
    if (animationStart === null) animationStart = time;
    const elapsed = time - animationStart;
    const progress = Math.min(1, animationFrom + elapsed / animationDuration);
    if (progress !== state.progress) updateProgress(progress);
    if (elapsed >= (1 - animationFrom) * animationDuration + endpointHold) {
      animationStart = time;
      animationFrom = 0;
      updateProgress(0);
    }
    frameId = requestAnimationFrame(tick);
  }

  function play() {
    if (state.playing) { stop(); announce('Transition paused.'); return; }
    hideTooltip();
    if (reducedMotion.matches) {
      updateProgress(state.progress === 1 ? 0 : 1);
      stop(); announce('Moved directly to the endpoint to respect reduced motion.');
      return;
    }
    if (state.progress >= 1) updateProgress(0);
    animationFrom = state.progress;
    animationStart = null;
    state.playing = true;
    $('play-symbol').textContent = 'Ⅱ';
    $('play-label').textContent = 'Pause transition';
    $('play').setAttribute('aria-pressed', 'true');
    frameId = requestAnimationFrame(tick);
  }

  function populateParticipants() {
    $('participant').replaceChildren(new Option('All participants', 'all'));
    current().participants.forEach(p => $('participant').add(new Option(`Participant ${p}`, String(p))));
    $('participant').value = state.participant;
  }

  function updateInspector() {
    const p = state.selected === null ? null : points()[state.selected];
    $('clear-selection').hidden = !p;
    $('inspector').hidden = !p;
    $('inspector-text').replaceChildren();
    if (!p) return;
    const strong = document.createElement('strong');
    strong.textContent = `${className(p.activity)} · P${p.participant} · window ${p.id}`;
    $('inspector-text').append(strong, document.createTextNode(` — JEPA: ${className(p.predictionFrom)}. W-JEPA: ${className(p.predictionTo)}.`));
  }

  function selectPoint(index) {
    state.selected = index;
    updateInspector();
    drawScatter();
  }

  function hideTooltip() { $('tooltip').hidden = true; state.hovered = null; }

  function nearest(event) {
    const canvas = event.currentTarget;
    const view = plotViews.get(canvas);
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    let found = null, best = event.pointerType === 'touch' ? 26 : 19;
    for (const p of view.rendered) {
      const distance = Math.hypot(p.x - x, p.y - y);
      if (distance < best) { found = p; best = distance; }
    }
    return found;
  }

  function showTooltip(found) {
    const p = points()[found.i];
    const tooltip = $('tooltip');
    found.canvas.parentElement.append(tooltip);
    const geometry = plotViews.get(found.canvas).geometry;
    tooltip.replaceChildren();
    const title = document.createElement('strong'); title.textContent = className(p.activity);
    const label = document.createElement('small'); label.textContent = `Participant ${p.participant} · window ${p.id}`;
    const before = document.createElement('div'); before.textContent = `JEPA: ${className(p.predictionFrom)}`;
    const after = document.createElement('div'); after.textContent = `W-JEPA: ${className(p.predictionTo)}`;
    tooltip.append(title, label, before, after);
    tooltip.hidden = false;
    const box = tooltip.getBoundingClientRect();
    tooltip.style.left = `${Math.max(7, Math.min(found.x + 16, geometry.width - box.width - 7))}px`;
    tooltip.style.top = `${Math.max(7, Math.min(found.y - box.height - 13, geometry.height - box.height - 7))}px`;
  }

  function drawSignal(canvas, series, yLabel) {
    const { context: c, width, height } = canvasSize(canvas);
    const left = 57, right = 16, top = 32, bottom = 37;
    const w = width - left - right, h = height - top - bottom;
    const all = series.flat();
    const low = Math.min(...all), high = Math.max(...all), gap = Math.max((high - low) * .13, .00001);
    const yMin = low - gap, yMax = high + gap;
    const x = i => left + (i / (series[0].length - 1)) * w;
    const y = v => top + (yMax - v) / (yMax - yMin) * h;
    c.font = '12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
    c.textBaseline = 'middle'; c.textAlign = 'right';
    for (const v of ticks(yMin, yMax, 3)) {
      if (y(v) < top + 2 || y(v) > top + h - 2) continue;
      c.strokeStyle = v === 0 ? '#cbd8df' : '#edf1f4'; c.lineWidth = 1;
      c.beginPath(); c.moveTo(left, y(v)); c.lineTo(left + w, y(v)); c.stroke();
      c.fillStyle = '#667c8a'; c.fillText(number(v), left - 9, y(v));
    }
    c.textAlign = 'center'; c.fillStyle = '#667c8a';
    for (const sec of (width < 320 ? [0, 2] : width < 400 ? [0, 2, 4] : [0, 1, 2, 3, 4])) c.fillText(String(sec), x(sec * 100), top + h + 18);
    c.textAlign = 'right'; c.fillText('Time (s)', left + w, top + h + 18);
    c.textAlign = 'left'; c.fillStyle = '#667c8a'; c.fillText(yLabel, left, 14);
    c.save(); c.beginPath(); c.rect(left, top, w, h); c.clip();
    series.forEach((values, k) => {
      c.strokeStyle = signalColors[14 + k]; c.lineWidth = 1.4; c.globalAlpha = .89;
      c.setLineDash(k === 1 ? [4, 2] : []);
      c.beginPath(); values.forEach((v, i) => i === 0 ? c.moveTo(x(i), y(v)) : c.lineTo(x(i), y(v))); c.stroke();
    });
    c.setLineDash([]); c.globalAlpha = 1;
    c.restore();
  }

  function drawSignals() {
    drawSignal($('raw-signal'), D.example.raw, 'Angular velocity (native units)');
    const band = D.example.bands[state.band];
    drawSignal($('detail-signal'), band.signals, `${band.name} amplitude (native units)`);
  }

  function setBand(index) {
    state.band = index;
    const band = D.example.bands[index];
    $('band-heading').textContent = index === 0 ? 'The A4 approximation' : `The ${band.name} wavelet detail`;
    $('detail-fold').textContent = `${band.ratio.toFixed(2)}× H/V RMS`;
    $('detail-caption').textContent = index === 2
      ? 'Isolating a scale reveals differences already in the signal.'
      : `${band.name} · nominal ${ranges[index]} Hz · a separate amplitude scale from the raw panel.`;
    $('detail-signal').setAttribute('aria-label', `Reconstructed db4 ${band.name} component. Horizontal to vertical RMS ratio ${band.ratio.toFixed(3)} in the selected pair. Frequency range is nominal ${ranges[index]} hertz.`);
    root.querySelectorAll('[data-band]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.band) === index)));
    $('band-note').hidden = index <= 2;
    $('band-note').textContent = index > 2
      ? `${band.name} is above DSADS’s native 12.5 Hz Nyquist limit. These coefficients arise from resampling and filtering; they are not newly measured high-frequency movement. The band choice does not change the PCA or model results.`
      : '';
    drawSignals();
  }

  if (hasPca) {
    $('play').addEventListener('click', play);
    $('progress').addEventListener('input', event => { stop(); hideTooltip(); updateProgress(Number(event.target.value) / 1000); });
    $('progress').addEventListener('change', () => announce($('progress').getAttribute('aria-valuetext')));
    $('before').addEventListener('click', () => { stop(); hideTooltip(); updateProgress(0); announce('JEPA endpoint.'); });
    $('after').addEventListener('click', () => { stop(); hideTooltip(); updateProgress(1); announce('W-JEPA endpoint.'); });
    $('reset').addEventListener('click', () => {
      stop(); hideTooltip(); state.selected = null; state.participant = 'all'; state.trails = false;
      $('participant').value = 'all'; $('trails').checked = false;
      updateInspector(); updateProgress(0); announce('PCA view reset.');
    });
    $('split').addEventListener('change', event => {
      stop(); hideTooltip(); state.split = event.target.value; state.selected = null; state.participant = 'all';
      populateParticipants(); updateInspector(); updateProgress(state.progress);
      announce(`Showing ${state.split === 'participant' ? 'unseen participants' : 'random windows'}.`);
    });
    $('participant').addEventListener('change', event => { state.participant = event.target.value; hideTooltip(); drawScatter(); });
    $('trails').addEventListener('change', event => { state.trails = event.target.checked; drawScatter(); });
    $('clear-selection').addEventListener('click', () => { hideTooltip(); selectPoint(null); });
    for (const canvas of [reference, scatter]) {
    canvas.addEventListener('pointermove', event => {
      if (event.pointerType === 'touch' || state.playing) return;
      const found = nearest(event);
      if (!found) { hideTooltip(); drawScatter(); return; }
      state.hovered = found.i; drawScatter(); showTooltip(found);
    });
    canvas.addEventListener('pointerleave', () => { hideTooltip(); drawScatter(); });
    canvas.addEventListener('click', event => {
      const found = nearest(event); stop(); hideTooltip(); selectPoint(found ? found.i : null);
    });
    canvas.addEventListener('keydown', event => {
      if (event.key === 'Escape') { hideTooltip(); selectPoint(null); return; }
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault(); stop(); hideTooltip();
      const visible = rendered.filter(p => p.active).map(p => p.i);
      if (!visible.length) return;
      const index = visible.indexOf(state.selected);
      let next = event.key === 'Home' ? 0 : event.key === 'End' ? visible.length - 1
        : index === -1 ? 0 : (index + (event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1) + visible.length) % visible.length;
      selectPoint(visible[next]);
    });
    }
    populateParticipants();
    updateProgress(1);
  }
  if (hasWavelets) {
    root.querySelectorAll('[data-band]').forEach(button => button.addEventListener('click', () => { setBand(Number(button.dataset.band)); announce(`${D.example.bands[state.band].name} waveform shown.`); }));
    setBand(2);
  }
  document.addEventListener('visibilitychange', () => { if (hasPca && document.hidden) stop(); });
  reducedMotion.addEventListener('change', () => { if (hasPca && reducedMotion.matches) stop(); });
  const redraw = () => {
    if (hasPca) { hideTooltip(); drawScatter(); }
    if (hasWavelets) drawSignals();
  };
  window.addEventListener('beforeprint', () => { if (hasPca) stop(); redraw(); });
  const resize = new ResizeObserver(redraw);
  [reference, scatter, $('raw-signal'), $('detail-signal')].filter(Boolean).forEach(canvas => resize.observe(canvas));
  if (window.parent !== window) {
    let lastHeight = 0;
    const sendHeight = () => {
      const height = Math.ceil(root.getBoundingClientRect().height + 2);
      if (height !== lastHeight) { window.parent.postMessage({ type: 'wjepa:resize', height }, '*'); lastHeight = height; }
    };
    new ResizeObserver(sendHeight).observe(root);
    window.addEventListener('load', sendHeight);
  }
})();
