/**
 * Fortune500 — functional registry + search + deterministic “brand mark” icons for every company.
 * Data: Fortune 500 ranks from public historical CSV (cmusam/fortune500 2019); domains curated + heuristics.
 *
 * @see https://github.com/cmusam/fortune500
 */
(function (global) {
  "use strict";

  var LIST_URL = global.FORTUNE500_CSV_URL || "data/fortune500.csv";

  /** @type {{ rank: number, name: string, revenue: number, profit: number, domain: string, slug: string }[] | null} */
  var _companies = null;
  var _loadPromise = null;
  var _domainMap = Object.create(null);

  function normalizeKey(name) {
    return String(name || "")
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9]/g, "");
  }

  function slugify(name) {
    return String(name || "")
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
      .slice(0, 80) || "company";
  }

  /** FNV-1a 32-bit — stable hash for hues */
  function hash32(str) {
    var h = 2166136261;
    var s = String(str);
    for (var i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function stripLegalSuffix(name) {
    return String(name || "").replace(
      /\s*,?\s*(Inc\.?|LLC|L\.L\.C\.|LP|L\.P\.|PLC|Corp\.?|Corporation|Company|Co\.|Cos\.|Group|Holdings|International)\.?$/i,
      "",
    ).trim();
  }

  function guessDomainFromName(name) {
    var core = stripLegalSuffix(name);
    var slug = normalizeKey(core).replace(/^(\d+)/, "x$1");
    if (!slug) slug = "company";
    if (slug.length > 40) slug = slug.slice(0, 40);
    return slug + ".com";
  }

  function buildDomainMap() {
    var tuples = global.FORTUNE_DOMAIN_OVERRIDES || [];
    _domainMap = Object.create(null);
    for (var i = 0; i < tuples.length; i++) {
      var pair = tuples[i];
      var label = pair[0];
      var domain = pair[1];
      var keys = [normalizeKey(label), normalizeKey(stripLegalSuffix(label))];
      for (var k = 0; k < keys.length; k++) {
        if (keys[k]) _domainMap[keys[k]] = domain;
      }
    }
  }

  function resolveDomain(csvName) {
    buildDomainMap();
    var candidates = [
      normalizeKey(csvName),
      normalizeKey(stripLegalSuffix(csvName)),
      normalizeKey(csvName.replace(/\s+Wholesale\s*/i, " ").trim()),
    ];
    for (var i = 0; i < candidates.length; i++) {
      if (candidates[i] && _domainMap[candidates[i]]) return _domainMap[candidates[i]];
    }
    var nk = normalizeKey(csvName);
    if (global.FORTUNE_DOMAIN_OVERRIDES) {
      for (var j = 0; j < global.FORTUNE_DOMAIN_OVERRIDES.length; j++) {
        var t = global.FORTUNE_DOMAIN_OVERRIDES[j];
        var ok = normalizeKey(t[0]);
        if (ok.length >= 6 && (nk.indexOf(ok.slice(0, 8)) === 0 || ok.indexOf(nk.slice(0, 8)) === 0)) {
          return t[1];
        }
      }
    }
    return guessDomainFromName(csvName);
  }

  function parseCsvLine(line) {
    var parts = [];
    var cur = "";
    var q = false;
    for (var i = 0; i < line.length; i++) {
      var c = line[i];
      if (c === '"') {
        q = !q;
      } else if ((c === "," && !q) || c === "\r") {
        parts.push(cur);
        cur = "";
      } else if (c !== "\n") {
        cur += c;
      }
    }
    parts.push(cur);
    return parts;
  }

  function parseFortune500Csv(text) {
    var lines = text.split(/\r?\n/).filter(function (l) {
      return l.trim();
    });
    if (!lines.length) return [];
    var header = parseCsvLine(lines[0]).map(function (h) {
      return h.trim().toLowerCase();
    });
    var rankIdx = header.indexOf("rank");
    var companyIdx = header.indexOf("company");
    var revIdx = header.indexOf("revenue ($ millions)");
    var profitIdx = header.indexOf("profit ($ millions)");
    var out = [];
    for (var r = 1; r < lines.length; r++) {
      var cols = parseCsvLine(lines[r]);
      if (cols.length < 2) continue;
      var rank = parseInt(cols[rankIdx >= 0 ? rankIdx : 0], 10);
      var name = (cols[companyIdx >= 0 ? companyIdx : 1] || "").trim();
      if (!name || !Number.isFinite(rank)) continue;
      var revenue = parseFloat((cols[revIdx] || "0").replace(/,/g, "")) || 0;
      var profit = parseFloat((cols[profitIdx] || "0").replace(/,/g, "")) || 0;
      var domain = resolveDomain(name);
      out.push({
        rank: rank,
        name: name,
        revenue: revenue,
        profit: profit,
        domain: domain,
        slug: slugify(name),
      });
    }
    return out.sort(function (a, b) {
      return a.rank - b.rank;
    });
  }

  function scoreMatch(query, company) {
    var q = normalizeKey(query);
    if (!q) return 0;
    var n = normalizeKey(company.name);
    if (n === q) return 1e9;
    if (n.indexOf(q) === 0) return 1e8 - company.rank;
    if (n.indexOf(q) >= 0) return 1e7 - company.rank;
    var words = company.name.toLowerCase().split(/[^a-z0-9]+/);
    var hit = 0;
    for (var i = 0; i < words.length; i++) {
      if (words[i] && words[i].indexOf(query.toLowerCase()) === 0) hit++;
    }
    if (hit) return 5e6 + hit * 100 - company.rank;
    return 0;
  }

  /**
   * Search loaded companies by name substring / prefix.
   * @returns {typeof _companies}
   */
  function searchCompanies(query, limit) {
    if (!_companies || !query || !String(query).trim()) return [];
    var lim = limit || 25;
    var scored = [];
    var q = String(query).trim();
    for (var i = 0; i < _companies.length; i++) {
      var c = _companies[i];
      var s = scoreMatch(q, c);
      if (s > 0) scored.push({ c: c, s: s });
    }
    scored.sort(function (a, b) {
      return b.s - a.s;
    });
    return scored.slice(0, lim).map(function (x) {
      return x.c;
    });
  }

  function getByRank(rank) {
    if (!_companies) return null;
    var r = parseInt(rank, 10);
    for (var i = 0; i < _companies.length; i++) {
      if (_companies[i].rank === r) return _companies[i];
    }
    return null;
  }

  function getBySlug(slug) {
    if (!_companies) return null;
    for (var i = 0; i < _companies.length; i++) {
      if (_companies[i].slug === slug) return _companies[i];
    }
    return null;
  }

  /** Two-letter (or one) initials for icon */
  function initials(name) {
    var parts = String(name || "")
      .replace(/&/g, " and ")
      .split(/[^a-zA-Z0-9]+/)
      .filter(Boolean);
    if (!parts.length) return "?";
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return (parts[0][0] + parts[1][0]).toUpperCase();
  }

  /**
   * Escape for SVG text (minimal).
   */
  function escXml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /**
   * Deterministic gradient “icon” for any company (SVG string). Unique gradient id per slug.
   * @param {{ name: string, domain?: string, slug?: string, rank?: number }} company
   * @param {number} size viewBox size (default 40)
   */
  function iconMarkSvg(company, size) {
    var sz = size || 40;
    var seed = (company && company.domain) || (company && company.name) || "x";
    var h = hash32(seed) % 360;
    var h2 = (h + 48 + (company && company.rank ? company.rank % 20 : 0)) % 360;
    var slug = (company && company.slug) || slugify(company && company.name);
    var gid = "fg-" + slug.replace(/[^a-z0-9-]/gi, "") + "-" + (company && company.rank ? company.rank : "x");
    var ini = initials(company && company.name);
    var fs = ini.length > 2 ? 11 : 14;
    return (
      '<svg xmlns="http://www.w3.org/2000/svg" width="' +
      sz +
      '" height="' +
      sz +
      '" viewBox="0 0 40 40" role="img" aria-hidden="true">' +
      "<defs>" +
      '<linearGradient id="' +
      gid +
      '" x1="0" y1="0" x2="40" y2="40">' +
      '<stop offset="0%" stop-color="hsl(' +
      h +
      ',62%,46%)"/>' +
      '<stop offset="100%" stop-color="hsl(' +
      h2 +
      ',52%,34%)"/>' +
      "</linearGradient>" +
      "</defs>" +
      '<rect width="40" height="40" rx="11" fill="url(#' +
      gid +
      ')"/>' +
      '<text x="20" y="25" text-anchor="middle" fill="white" font-family="system-ui,Segoe UI,sans-serif" font-size="' +
      fs +
      '" font-weight="700">' +
      escXml(ini) +
      "</text>" +
      "</svg>"
    );
  }

  /** Same colors as iconMarkSvg; returns { background: css linear-gradient, initials } for CSS-only chips */
  function iconMarkStyle(company) {
    var seed = (company && company.domain) || (company && company.name) || "x";
    var h = hash32(seed) % 360;
    var h2 = (h + 48 + (company && company.rank ? company.rank % 20 : 0)) % 360;
    return {
      initials: initials(company && company.name),
      background: "linear-gradient(145deg, hsl(" + h + ",62%,46%), hsl(" + h2 + ",52%,34%))",
    };
  }

  /**
   * Safe data URL for SVG (base64 avoids `#` truncating data:image/svg+xml,... URLs).
   */
  function svgToDataUrl(svg) {
    try {
      var utf8 = unescape(encodeURIComponent(svg));
      return "data:image/svg+xml;base64," + btoa(utf8);
    } catch (e) {
      return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
    }
  }

  function iconMarkDataUrl(company, size) {
    return svgToDataUrl(iconMarkSvg(company, size));
  }

  function loadFromText(text) {
    _companies = parseFortune500Csv(text);
    return _companies;
  }

  function load(url) {
    if (_loadPromise) return _loadPromise;
    var u = url || LIST_URL;
    _loadPromise = fetch(u, { cache: "force-cache" })
      .then(function (res) {
        if (!res.ok) throw new Error("Fortune500: could not load " + u + " (" + res.status + ")");
        return res.text();
      })
      .then(function (text) {
        loadFromText(text);
        return _companies;
      });
    return _loadPromise;
  }

  function ready() {
    return _loadPromise || load();
  }

  function list() {
    return _companies ? _companies.slice() : [];
  }

  buildDomainMap();

  var Fortune500 = {
    normalizeKey: normalizeKey,
    slugify: slugify,
    parseFortune500Csv: parseFortune500Csv,
    load: load,
    loadFromText: loadFromText,
    ready: ready,
    searchCompanies: searchCompanies,
    getByRank: getByRank,
    getBySlug: getBySlug,
    list: list,
    iconMarkSvg: iconMarkSvg,
    iconMarkDataUrl: iconMarkDataUrl,
    svgToDataUrl: svgToDataUrl,
    iconMarkStyle: iconMarkStyle,
    initials: initials,
    resolveDomain: resolveDomain,
    get CSV_URL() {
      return LIST_URL;
    },
    setCsvUrl: function (url) {
      LIST_URL = url;
      _loadPromise = null;
      _companies = null;
    },
  };

  global.Fortune500 = Fortune500;
})(typeof window !== "undefined" ? window : globalThis);
