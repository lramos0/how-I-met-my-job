/* global Papa */
/* importScripts loads Papa and micro-sql into global scope */
importScripts('./vendor/papaparse.min.js', './micro-sql.js');

const MAX_PREVIEW = 500;
const MAX_QUERY_ROWS = 5000;

/** @type {{ columns: string[], rows: Record<string,string>[] } | null} */
let table = null;

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

function normalizeTable(parsed) {
  const columns = parsed.meta.fields || [];
  const rows = (parsed.data || []).map((row) => {
    const o = {};
    for (const c of columns) {
      o[c] = row[c] == null ? '' : String(row[c]);
    }
    return o;
  });
  return { columns, rows };
}

function reply(source, id, ok, data, error) {
  if (source && typeof source.postMessage === 'function') {
    source.postMessage({ id, ok, data, error });
  }
}

self.addEventListener('message', (event) => {
  const { id, type, payload } = event.data || {};
  if (!id || !type) {
    return;
  }
  const src = event.source;

  try {
    if (type === 'LOAD_CSV') {
      const csvText = payload && payload.csvText;
      if (typeof csvText !== 'string') {
        throw new Error('Missing csvText');
      }
      const parsed = Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
        dynamicTyping: false,
      });
      const quoteErr = (parsed.errors || []).find((e) => e && e.type === 'Quotes');
      if (quoteErr) {
        throw new Error(quoteErr.message || 'CSV parse error');
      }
      table = normalizeTable(parsed);
      const preview = table.rows.slice(0, MAX_PREVIEW);
      reply(src, id, true, {
        columns: table.columns,
        rowCount: table.rows.length,
        rows: preview,
        truncated: table.rows.length > MAX_PREVIEW,
      });
      return;
    }

    if (type === 'RESET') {
      if (!table) {
        throw new Error('No CSV loaded');
      }
      const preview = table.rows.slice(0, MAX_PREVIEW);
      reply(src, id, true, {
        columns: table.columns,
        rowCount: table.rows.length,
        rows: preview,
        truncated: table.rows.length > MAX_PREVIEW,
      });
      return;
    }

    if (type === 'QUERY') {
      if (!table) {
        throw new Error('No CSV loaded');
      }
      const sql = (payload && payload.sql) || '';
      const ast = parseMicroSql(sql);
      const result = executeMicroSql(ast, table);
      const truncated = result.rows.length > MAX_QUERY_ROWS;
      reply(src, id, true, {
        columns: result.columns,
        rows: result.rows.slice(0, MAX_QUERY_ROWS),
        rowCount: result.rows.length,
        truncated,
      });
      return;
    }

    reply(src, id, false, null, `Unknown message type: ${type}`);
  } catch (e) {
    const msg = e && e.message ? e.message : String(e);
    reply(src, id, false, null, msg);
  }
});
