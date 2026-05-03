/**
 * CSV Lab UI: registers the scoped service worker (HTTPS or localhost only).
 * First install may require one reload so the page is controlled — see sessionStorage guard.
 */
(function () {
  const swUrl = new URL('sw.js', window.location.href);
  const scopeUrl = new URL('./', window.location.href).href;

  const els = {
    file: document.getElementById('csvFile'),
    tableWrap: document.getElementById('tableWrap'),
    meta: document.getElementById('meta'),
    prompt: document.getElementById('sqlPrompt'),
    error: document.getElementById('sqlError'),
    swStatus: document.getElementById('swStatus'),
  };

  els.file.disabled = true;

  function setError(msg) {
    els.error.textContent = msg || '';
    els.error.hidden = !msg;
  }

  function setSwStatus(text) {
    els.swStatus.textContent = text;
  }

  function renderTable(data) {
    const { columns, rows, rowCount, truncated } = data;
    const total = rowCount != null ? rowCount : rows.length;
    const truncNote = truncated ? ' (showing first rows only; full set is in the worker)' : '';
    els.meta.textContent = `${total} row(s), ${columns.length} column(s)${truncNote}`;

    if (!columns.length) {
      els.tableWrap.innerHTML = '<p class="muted">No columns in CSV.</p>';
      return;
    }

    const thead = `<thead><tr>${columns.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead>`;
    const tbody = `<tbody>${rows
      .map(
        (row) =>
          `<tr>${columns.map((c) => `<td>${escapeHtml(row[c] ?? '')}</td>`).join('')}</tr>`
      )
      .join('')}</tbody>`;
    els.tableWrap.innerHTML = `<table class="dataframe">${thead}${tbody}</table>`;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function sendMessage(type, payload) {
    return new Promise((resolve, reject) => {
      const controller = navigator.serviceWorker.controller;
      if (!controller) {
        reject(new Error('Service worker is not controlling this page yet. Reload once and try again.'));
        return;
      }
      const id = crypto.randomUUID();
      const onMessage = (ev) => {
        const d = ev.data;
        if (!d || d.id !== id) {
          return;
        }
        navigator.serviceWorker.removeEventListener('message', onMessage);
        if (d.ok) {
          resolve(d.data);
        } else {
          reject(new Error(d.error || 'Unknown error'));
        }
      };
      navigator.serviceWorker.addEventListener('message', onMessage);
      controller.postMessage({ id, type, payload });
    });
  }

  async function loadCsv(text) {
    setError('');
    const data = await sendMessage('LOAD_CSV', { csvText: text });
    renderTable(data);
  }

  async function runQuery(sql) {
    setError('');
    const data = await sendMessage('QUERY', { sql });
    renderTable(data);
  }

  async function resetView() {
    setError('');
    const data = await sendMessage('RESET', {});
    renderTable(data);
  }

  async function initSw() {
    if (!('serviceWorker' in navigator)) {
      setSwStatus('Service workers are not available in this browser.');
      return false;
    }
    if (window.location.protocol === 'file:') {
      setSwStatus('Open this page over http://localhost (not file://) so the service worker can register.');
      return false;
    }

    try {
      await navigator.serviceWorker.register(swUrl.href, { scope: scopeUrl });
    } catch (e) {
      setSwStatus(`Registration failed: ${e.message || e}`);
      return false;
    }

    if (!navigator.serviceWorker.controller) {
      if (!sessionStorage.getItem('csv-lab-sw-activated')) {
        sessionStorage.setItem('csv-lab-sw-activated', '1');
        setSwStatus('Activating worker… reloading once.');
        window.location.reload();
        return false;
      }
      setSwStatus('No controlling service worker. Try a hard refresh.');
      return false;
    }

    setSwStatus('Service worker active — CSV parsing and queries run in the worker.');
    return true;
  }

  els.file.addEventListener('change', async () => {
    const f = els.file.files && els.file.files[0];
    if (!f) {
      return;
    }
    if (!navigator.serviceWorker.controller) {
      setError('Service worker not ready.');
      return;
    }
    const text = await f.text();
    try {
      await loadCsv(text);
    } catch (e) {
      setError(e.message || String(e));
    }
  });

  els.prompt.addEventListener('keydown', async (ev) => {
    if (ev.key !== 'Enter') {
      return;
    }
    ev.preventDefault();
    if (!navigator.serviceWorker.controller) {
      setError('Service worker not ready.');
      return;
    }
    const raw = els.prompt.value.trim();
    try {
      if (!raw || /^reset$/i.test(raw)) {
        await resetView();
        if (/^reset$/i.test(raw)) {
          els.prompt.value = '';
        }
        return;
      }
      await runQuery(raw);
    } catch (e) {
      setError(e.message || String(e));
    }
  });

  initSw().then((ok) => {
    els.file.disabled = !ok;
    if (ok) {
      els.tableWrap.innerHTML =
        '<p class="muted">Choose a .csv file. Then use the prompt, e.g. <code>SELECT DISTINCT(JOBS) WHERE location = \'US\'</code></p>';
    }
  });
})();
