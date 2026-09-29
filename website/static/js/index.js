/* Academic Project Page Template interactions, adapted for the W-JEPA content. */
(() => {
  'use strict';
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const topButton = document.getElementById('scroll-to-top');
  const updateTopButton = () => topButton.classList.toggle('visible', window.scrollY > 300);
  window.addEventListener('scroll', updateTopButton, { passive: true });
  updateTopButton();
  topButton.addEventListener('click', () => window.scrollTo({ top: 0, behavior: reducedMotion.matches ? 'instant' : 'smooth' }));

  document.querySelectorAll('[data-placeholder]').forEach(link => {
    link.addEventListener('click', event => {
      event.preventDefault();
      document.getElementById('resource-status').textContent = `${link.dataset.placeholder} link coming soon.`;
    });
  });

  const copyButton = document.getElementById('copy-bibtex');
  copyButton.addEventListener('click', async () => {
    const label = copyButton.querySelector('.copy-text');
    const citation = document.getElementById('bibtex-code');
    try {
      await navigator.clipboard.writeText(citation.textContent);
      label.textContent = 'Copied!';
      copyButton.classList.add('copied');
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(citation);
      selection.removeAllRanges();
      selection.addRange(range);
      label.textContent = 'Select & copy';
    }
    setTimeout(() => { label.textContent = 'Copy'; copyButton.classList.remove('copied'); }, 2500);
  });

})();
