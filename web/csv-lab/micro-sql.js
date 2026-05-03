/* global self */
// Micro-SQL parser and executor (service worker global scope via importScripts).

function tokenizeMicroSql(input) {
  const tokens = [];
  let i = 0;
  const keywords = new Set(['SELECT', 'DISTINCT', 'WHERE', 'AND', 'FROM']);

  while (i < input.length) {
    const c = input[i];
    if (/\s/.test(c)) {
      i += 1;
      continue;
    }
    if (c === '*') {
      tokens.push({ type: '*' });
      i += 1;
      continue;
    }
    if (c === ',') {
      tokens.push({ type: ',' });
      i += 1;
      continue;
    }
    if (c === '(') {
      tokens.push({ type: '(' });
      i += 1;
      continue;
    }
    if (c === ')') {
      tokens.push({ type: ')' });
      i += 1;
      continue;
    }
    if (c === '=') {
      tokens.push({ type: '=' });
      i += 1;
      continue;
    }
    if (c === "'") {
      let j = i + 1;
      let str = '';
      while (j < input.length) {
        if (input[j] === "'" && input[j + 1] === "'") {
          str += "'";
          j += 2;
          continue;
        }
        if (input[j] === "'") {
          j += 1;
          break;
        }
        str += input[j];
        j += 1;
      }
      if (j > input.length || input[j - 1] !== "'") {
        throw new Error('Unterminated string literal');
      }
      tokens.push({ type: 'STRING', value: str });
      i = j;
      continue;
    }
    if (/[a-zA-Z_]/.test(c)) {
      let j = i;
      while (j < input.length && /[a-zA-Z0-9_]/.test(input[j])) {
        j += 1;
      }
      const word = input.slice(i, j);
      const upper = word.toUpperCase();
      if (keywords.has(upper)) {
        tokens.push({ type: 'KW', value: upper });
      } else {
        tokens.push({ type: 'IDENT', value: word });
      }
      i = j;
      continue;
    }
    throw new Error(`Unexpected character "${c}" at position ${i}`);
  }
  return tokens;
}

function parseMicroSql(sql) {
  const trimmed = sql.trim().replace(/;+\s*$/, '');
  if (!trimmed) {
    throw new Error('Empty query');
  }
  const tokens = tokenizeMicroSql(trimmed);
  let pos = 0;

  function peek() {
    return tokens[pos];
  }

  function take() {
    if (pos >= tokens.length) {
      throw new Error('Unexpected end of input');
    }
    return tokens[pos++];
  }

  function expectKw(w) {
    const t = take();
    if (!t || t.type !== 'KW' || t.value !== w) {
      throw new Error(`Expected keyword ${w}`);
    }
  }

  expectKw('SELECT');

  let distinct = false;
  if (peek() && peek().type === 'KW' && peek().value === 'DISTINCT') {
    take();
    distinct = true;
  }

  let selectCols;
  if (peek() && peek().type === '*') {
    take();
    selectCols = null;
  } else {
    selectCols = [];
    while (true) {
      if (peek() && peek().type === '(') {
        take();
        const id = take();
        if (!id || id.type !== 'IDENT') {
          throw new Error('Expected column name inside parentheses');
        }
        selectCols.push(id.value);
        const close = take();
        if (!close || close.type !== ')') {
          throw new Error('Expected ) after column name');
        }
      } else {
        const id = take();
        if (!id || id.type !== 'IDENT') {
          throw new Error('Expected column name or *');
        }
        selectCols.push(id.value);
      }
      if (!peek() || peek().type !== ',') {
        break;
      }
      take();
    }
  }

  if (peek() && peek().type === 'KW' && peek().value === 'FROM') {
    throw new Error('FROM is not supported; the CSV is the only table');
  }

  const conditions = [];
  if (peek() && peek().type === 'KW' && peek().value === 'WHERE') {
    take();
    while (true) {
      const colTok = take();
      if (!colTok || colTok.type !== 'IDENT') {
        throw new Error('Expected column name in WHERE');
      }
      const eq = take();
      if (!eq || eq.type !== '=') {
        throw new Error('Expected = in WHERE clause');
      }
      const lit = take();
      if (!lit || lit.type !== 'STRING') {
        throw new Error("Expected quoted string after = (e.g. 'US')");
      }
      conditions.push({ col: colTok.value, value: lit.value });
      if (!peek() || peek().type !== 'KW' || peek().value !== 'AND') {
        break;
      }
      take();
    }
  }

  if (peek()) {
    throw new Error(`Unexpected token after query: ${JSON.stringify(peek())}`);
  }

  return { distinct, selectCols, conditions };
}

function executeMicroSql(ast, table) {
  const { distinct, selectCols, conditions } = ast;
  const colSet = new Set(table.columns);

  for (const w of conditions) {
    if (!colSet.has(w.col)) {
      throw new Error(`Unknown column in WHERE: ${w.col}`);
    }
  }

  const filtered = table.rows.filter((row) =>
    conditions.every((w) => (row[w.col] ?? '') === w.value)
  );

  const cols =
    selectCols === null ? table.columns.slice() : selectCols.slice();

  for (const c of cols) {
    if (!colSet.has(c)) {
      throw new Error(`Unknown column in SELECT: ${c}`);
    }
  }

  let out = filtered.map((row) => {
    const o = {};
    for (const c of cols) {
      o[c] = row[c];
    }
    return o;
  });

  if (distinct) {
    const seen = new Set();
    out = out.filter((o) => {
      const key = cols.map((c) => o[c]).join('\0');
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  return { columns: cols, rows: out };
}
