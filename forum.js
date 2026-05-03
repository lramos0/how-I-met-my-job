/**
 * Company forums — Fortune 500 hub + per-user posts, comments, upvotes/downvotes.
 * Routes: #forums and #company/:slug. Keeps old #f routes as redirects.
 */
(function () {
  var STORAGE_KEY = "hc_company_forums_v2";
  var LEGACY_KEY = "himmj_forum_v1";
  var LOCAL_USER_KEY = "hc_account_v1";
  var GUEST_KEY = "hc_forum_guest_uid";

  function $(id) { return document.getElementById(id); }
  function esc(s) { return String(s || "").replace(/[&<>\"]/g, function (c) { return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]; }); }
  function attr(s) { return esc(s).replace(/'/g, "&#39;"); }
  function uid(prefix) { return prefix + "-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8); }
  function debounce(fn, ms) { var t; return function () { var args = arguments; clearTimeout(t); t = setTimeout(function () { fn.apply(null, args); }, ms); }; }

  function readJson(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key) || "") || fallback; }
    catch (e) { return fallback; }
  }

  function defaultStore() { return { companies: {}, votes: {}, createdAt: Date.now(), updatedAt: Date.now() }; }

  function migrateLegacy(store) {
    var legacy = readJson(LEGACY_KEY, null);
    if (!legacy || store.__legacyMigrated) return store;
    Object.keys(legacy).forEach(function (slug) {
      if (!legacy[slug] || !Array.isArray(legacy[slug].posts)) return;
      if (!store.companies[slug]) store.companies[slug] = { posts: [] };
      legacy[slug].posts.forEach(function (p) {
        if (store.companies[slug].posts.some(function (x) { return x.id === p.id; })) return;
        store.companies[slug].posts.push(normalizePost(p));
      });
    });
    store.__legacyMigrated = true;
    saveStore(store);
    return store;
  }

  function loadStore() {
    var store = readJson(STORAGE_KEY, defaultStore());
    store = Object.assign(defaultStore(), store);
    store.companies = store.companies || {};
    store.votes = store.votes || {};
    return migrateLegacy(store);
  }

  function saveStore(store) {
    store.updatedAt = Date.now();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
    mirrorToAppState(store);
  }

  function auth() { return window.HiringCafeAuth || {}; }
  function getUser() {
    var api = auth();
    var u = api.getUser && api.getUser();
    if (u) return u;
    return readJson(LOCAL_USER_KEY, null);
  }
  function getProfile() {
    var api = auth();
    return (api.getProfile && api.getProfile()) || null;
  }
  function isLoggedIn() { return !!getUser(); }
  function getGuestId() {
    var id = localStorage.getItem(GUEST_KEY);
    if (!id) { id = "guest-" + Math.random().toString(36).slice(2, 10); localStorage.setItem(GUEST_KEY, id); }
    return id;
  }
  function currentUid() {
    var u = getUser();
    return u && u.uid ? u.uid : getGuestId();
  }
  function displayName() {
    var u = getUser();
    var p = getProfile();
    return (p && p.displayName) || (u && (u.name || u.email)) || "Community Member";
  }
  function promptLogin(action) {
    var msg = "Create an account to " + action + " and keep your posts/votes tied to your profile.";
    if (auth().promptLogin) auth().promptLogin(msg);
    else {
      var panel = $("accountPanel");
      var status = $("accountStatus");
      if (panel) panel.classList.remove("hidden");
      if (status) status.textContent = msg;
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }
  function requireLogin(action) {
    if (isLoggedIn()) return true;
    promptLogin(action);
    return false;
  }

  function normalizeComment(c) {
    c = c || {};
    return {
      id: c.id || uid("c"),
      body: c.body || "",
      author: c.author || "Community Member",
      authorUid: c.authorUid || null,
      createdAt: Number(c.createdAt || Date.now()),
      score: Number(c.score || 0),
      parentId: c.parentId || null,
      replies: (c.replies || []).map(normalizeComment)
    };
  }
  function normalizePost(p) {
    p = p || {};
    return {
      id: p.id || uid("p"),
      title: p.title || "Untitled thread",
      body: p.body || "",
      author: p.author || "Community Member",
      authorUid: p.authorUid || null,
      createdAt: Number(p.createdAt || Date.now()),
      score: Number(p.score || 0),
      comments: (p.comments || []).map(normalizeComment)
    };
  }
  function getCompanyBucket(store, slug) {
    if (!store.companies[slug]) store.companies[slug] = { posts: [] };
    store.companies[slug].posts = (store.companies[slug].posts || []).map(normalizePost);
    return store.companies[slug];
  }
  function findComment(list, id) {
    for (var i = 0; i < (list || []).length; i++) {
      if (list[i].id === id) return list[i];
      var nested = findComment(list[i].replies || [], id);
      if (nested) return nested;
    }
    return null;
  }
  function countComments(list) {
    return (list || []).reduce(function (n, c) { return n + 1 + countComments(c.replies || []); }, 0);
  }

  function targetKey(type, slug, postId, commentId) {
    return [type, slug, postId || "", commentId || ""].join(":");
  }
  function getVote(store, key) {
    var uid = currentUid();
    return store.votes && store.votes[uid] ? Number(store.votes[uid][key] || 0) : 0;
  }
  function setVote(store, key, value) {
    var uid = currentUid();
    if (!store.votes[uid]) store.votes[uid] = {};
    if (value) store.votes[uid][key] = value;
    else delete store.votes[uid][key];
  }
  function applyVote(store, item, key, delta) {
    var prev = getVote(store, key);
    var next = prev === delta ? 0 : delta;
    item.score = Number(item.score || 0) + (next - prev);
    setVote(store, key, next);
  }
  function votePost(slug, postId, delta) {
    if (!requireLogin(delta > 0 ? "upvote posts" : "downvote posts")) return;
    var store = loadStore();
    var post = getCompanyBucket(store, slug).posts.find(function (p) { return p.id === postId; });
    if (!post) return;
    applyVote(store, post, targetKey("post", slug, postId), delta);
    saveStore(store);
    route();
  }
  function voteComment(slug, postId, commentId, delta) {
    if (!requireLogin(delta > 0 ? "upvote comments" : "downvote comments")) return;
    var store = loadStore();
    var post = getCompanyBucket(store, slug).posts.find(function (p) { return p.id === postId; });
    var comment = post && findComment(post.comments, commentId);
    if (!comment) return;
    applyVote(store, comment, targetKey("comment", slug, postId, commentId), delta);
    saveStore(store);
    route();
  }

  function addPost(slug, title, body) {
    if (!requireLogin("start a thread")) return;
    title = String(title || "").trim(); body = String(body || "").trim();
    if (!title) return;
    var store = loadStore();
    var bucket = getCompanyBucket(store, slug);
    var id = uid("p");
    var post = { id: id, title: title, body: body, author: displayName(), authorUid: currentUid(), createdAt: Date.now(), score: 1, comments: [] };
    bucket.posts.unshift(post);
    setVote(store, targetKey("post", slug, id), 1);
    saveStore(store);
    route();
  }
  function addComment(slug, postId, body, parentId) {
    if (!requireLogin(parentId ? "reply" : "comment")) return;
    body = String(body || "").trim();
    if (!body) return;
    var store = loadStore();
    var post = getCompanyBucket(store, slug).posts.find(function (p) { return p.id === postId; });
    if (!post) return;
    var id = uid("c");
    var comment = { id: id, body: body, author: displayName(), authorUid: currentUid(), createdAt: Date.now(), score: 1, parentId: parentId || null, replies: [] };
    if (parentId) {
      var parent = findComment(post.comments, parentId);
      if (parent) parent.replies.push(comment);
      else post.comments.push(comment);
    } else post.comments.push(comment);
    setVote(store, targetKey("comment", slug, postId, id), 1);
    saveStore(store);
    route();
  }

  function mirrorToAppState(store) {
    if (!window.HiringCafeAuth || !window.HiringCafeAuth.saveForumState) return;
    var flatPosts = {}, postVotes = {}, flatVotes = {};
    Object.keys(store.companies || {}).forEach(function (slug) {
      (store.companies[slug].posts || []).forEach(function (p) {
        flatPosts[p.id] = { id: p.id, companySlug: slug, title: p.title, body: p.body, authorUid: p.authorUid, authorName: p.author, createdAt: p.createdAt, upvotes: Math.max(0, p.score || 0) };
        postVotes[p.id] = Math.max(0, p.score || 0);
      });
    });
    Object.keys(store.votes || {}).forEach(function (uid) {
      Object.keys(store.votes[uid] || {}).forEach(function (key) {
        flatVotes[uid + ":" + key] = store.votes[uid][key];
      });
    });
    window.HiringCafeAuth.saveForumState({ posts: flatPosts, upvotes: flatVotes, postVotes: postVotes });
  }

  function parseRoute() {
    var raw = (location.hash || "").replace(/^#/, "").replace(/^!/, "");
    if (raw === "forums" || raw === "forums/") return { type: "hub" };
    if (raw.indexOf("company/") === 0) return { type: "company", slug: decodeURIComponent(raw.slice(8).split("/")[0] || "") };
    if (raw === "f" || raw === "f/") { location.replace("#forums"); return { type: "hub" }; }
    if (raw.indexOf("f/") === 0) { location.replace("#company/" + encodeURIComponent(decodeURIComponent(raw.slice(2).split("/")[0] || ""))); return { type: "company", slug: decodeURIComponent(raw.slice(2).split("/")[0] || "") }; }
    return { type: "jobs" };
  }
  function setShell(route) {
    var isForum = route.type === "hub" || route.type === "company";
    var jm = $("jobsMain"), fv = $("forumView"), jobs = $("navJobs"), forums = $("navForums");
    if (jm) jm.classList.toggle("hidden", isForum);
    if (fv) fv.classList.toggle("hidden", !isForum);
    if (jobs) jobs.classList.toggle("is-active", !isForum);
    if (forums) forums.classList.toggle("is-active", isForum);
    [".filters", ".hire-banner", ".toolbar"].forEach(function (sel) {
      var el = document.querySelector(sel);
      if (el) el.classList.toggle("hidden", isForum);
    });
  }

  function waitForFortune() {
    if (window.Fortune500 && window.Fortune500.ready) return window.Fortune500.ready();
    return Promise.resolve();
  }
  function fortuneList() { return window.Fortune500 && window.Fortune500.list ? window.Fortune500.list() : []; }
  function companyBySlug(slug) { return window.Fortune500 && window.Fortune500.getBySlug ? window.Fortune500.getBySlug(slug) : null; }
  function searchCompanies(q, n) { return window.Fortune500 && window.Fortune500.searchCompanies ? window.Fortune500.searchCompanies(q, n) : []; }
  function logoHtml(company, size) {
    if (window.Fortune500 && window.Fortune500.iconMarkSvg) return window.Fortune500.iconMarkSvg(company, size || 40);
    var label = esc((company && company.name || "?").slice(0, 2).toUpperCase());
    return '<span class="fortune-logo-img" style="display:grid;place-items:center;width:' + (size || 40) + 'px;height:' + (size || 40) + 'px">' + label + '</span>';
  }

  function timeAgo(ts) {
    var diff = Math.max(0, Date.now() - Number(ts || Date.now()));
    var min = Math.floor(diff / 60000);
    if (min < 1) return "just now";
    if (min < 60) return min + " min ago";
    var h = Math.floor(min / 60);
    if (h < 48) return h + " hr ago";
    return Math.floor(h / 24) + " days ago";
  }
  function sortPosts(posts, mode) {
    return posts.slice().sort(function (a, b) {
      if (mode === "new") return b.createdAt - a.createdAt;
      if (mode === "top") return b.score - a.score;
      return (b.score + countComments(b.comments) * 2 + b.createdAt / 1e13) - (a.score + countComments(a.comments) * 2 + a.createdAt / 1e13);
    });
  }
  function fmtNum(n) { return n == null || n === "" ? "—" : Number(n).toLocaleString(); }
  function hashNum(str) { var h = 2166136261; str = String(str || ""); for (var i=0;i<str.length;i++){ h ^= str.charCodeAt(i); h += (h<<1)+(h<<4)+(h<<7)+(h<<8)+(h<<24); } return Math.abs(h>>>0); }
  function demoOutcomes(company) {
    var roles = ["Software Engineer", "Product Manager", "Data Analyst", "Business Analyst", "Operations", "Sales", "Finance", "UX Designer"];
    var locs = ["New York, NY", "San Francisco, CA", "Austin, TX", "Chicago, IL", "Seattle, WA", "Remote", "Boston, MA", "Atlanta, GA"];
    var skills = ["SQL · Excel · Tableau", "Python · APIs · Cloud", "React · TypeScript", "Stakeholder mgmt", "Java · Spring · AWS", "Power BI", "Kubernetes", "Figma · Research"];
    var seed = hashNum(company.slug || company.name);
    return Array.from({ length: 8 }, function (_, i) { return { sourceIndustry: company.industry || "Fortune 500", role: roles[(seed+i)%roles.length], location: locs[(seed+i*3)%locs.length], skills: skills[(seed+i*5)%skills.length], yoe: 1 + ((seed+i)%10), outcome: i % 4 === 0 ? "Offer" : i % 3 === 0 ? "Final round" : "Interview", timing: (7 + ((seed+i*11)%35)) + " days" }; });
  }
  function renderOutcomePanel(company) {
    var rows = demoOutcomes(company);
    var body = rows.map(function (r) { return '<tr><td>' + esc(r.sourceIndustry) + '</td><td>' + esc(r.role) + '</td><td>' + esc(r.location) + '</td><td>' + esc(r.skills) + '</td><td>' + esc(r.yoe) + '</td><td><span class="reddit-outcome-pill">' + esc(r.outcome) + '</span></td><td>' + esc(r.timing) + '</td></tr>'; }).join("");
    return '<section class="reddit-outcomes"><div class="reddit-outcomes-head"><div><h3>Hiring outcomes & metadata</h3><p>GradCafe-style rows for this company. Replace demo rows with real submissions when backend persistence is added.</p></div><button type="button" class="reddit-btn-primary" data-open-outcome-survey>Share outcome</button></div><div class="reddit-outcome-table-wrap"><table class="reddit-outcome-table"><thead><tr><th>Hired from</th><th>Role track</th><th>Location</th><th>Skills</th><th>YOE</th><th>Outcome</th><th>Timing</th></tr></thead><tbody>' + body + '</tbody></table></div></section>';
  }

  function renderHub() {
    var root = $("forumRoot"); if (!root) return;
    var list = fortuneList();
    if (!list.length) {
      root.innerHTML = '<div class="reddit-empty">Loading Fortune 500 companies…</div>';
      waitForFortune().then(renderHub);
      return;
    }
    var cards = list.map(function (c) {
      return '<a class="reddit-hub-card" href="#company/' + encodeURIComponent(c.slug) + '"><span class="forum-icon-slot fc-mini">' + logoHtml(c, 36) + '</span><span><strong>' + esc(c.name) + '</strong><small>' + esc(c.industry || "Fortune 500") + '</small></span><span class="fc-rank">#' + esc(c.rank) + '</span></a>';
    }).join("");
    root.innerHTML = '<div class="reddit-top"><a href="#" class="reddit-back">← Job listings</a><div class="reddit-hub-title"><h1>Company forums</h1><p>Pick a Fortune 500 company. Each one has a logo, metadata, voting, posts, and comments.</p></div><div class="forum-co-search-wrap"><span class="forum-co-search-icon">⌕</span><input type="search" class="forum-co-search" id="forumHubSearch" placeholder="Search companies…" autocomplete="off"><ul class="forum-co-results" id="forumHubResults"></ul></div></div><div class="reddit-hub-grid">' + cards + '</div>';
    var input = $("forumHubSearch"), results = $("forumHubResults");
    if (input && results) input.addEventListener("input", debounce(function () {
      var q = input.value.trim(); results.innerHTML = ""; if (!q) return;
      searchCompanies(q, 12).forEach(function (c) {
        var li = document.createElement("li");
        li.innerHTML = '<button type="button"><span class="forum-icon-slot">' + logoHtml(c, 32) + '</span><span>' + esc(c.name) + '</span><span class="fc-rank">#' + esc(c.rank) + '</span></button>';
        li.querySelector("button").addEventListener("click", function () { location.hash = "company/" + encodeURIComponent(c.slug); });
        results.appendChild(li);
      });
    }, 150));
  }

  function renderCommentTree(comments, slug, postId, store, depth) {
    return (comments || []).map(function (c) {
      var key = targetKey("comment", slug, postId, c.id);
      var v = getVote(store, key);
      return '<div class="reddit-comment" data-depth="' + (depth || 0) + '"><div class="reddit-comment-bar" aria-hidden="true"></div><div class="reddit-comment-body"><div class="reddit-comment-meta"><strong>' + esc(c.author) + '</strong> · ' + timeAgo(c.createdAt) + ' · ' + esc(c.score) + ' pts</div><div class="reddit-comment-text">' + esc(c.body) + '</div><div class="reddit-post-actions"><button type="button" class="rd-c-up' + (v === 1 ? ' is-on' : '') + '" data-slug="' + attr(slug) + '" data-post="' + attr(postId) + '" data-cid="' + attr(c.id) + '">▲</button><button type="button" class="rd-c-down' + (v === -1 ? ' is-on' : '') + '" data-slug="' + attr(slug) + '" data-post="' + attr(postId) + '" data-cid="' + attr(c.id) + '">▼</button><button type="button" class="rd-reply" data-parent="' + attr(c.id) + '">Reply</button></div><div class="rd-reply-box hidden" data-reply-for="' + attr(c.id) + '"><textarea placeholder="Reply…"></textarea><button type="button" class="reddit-btn-primary rd-send-reply" data-parent="' + attr(c.id) + '">Reply</button></div>' + (c.replies && c.replies.length ? '<div class="reddit-nested">' + renderCommentTree(c.replies, slug, postId, store, (depth || 0) + 1) + '</div>' : '') + '</div></div>';
    }).join("");
  }

  function renderCompanyPage(slug) {
    var root = $("forumRoot"); if (!root) return;
    var company = companyBySlug(slug) || { slug: slug, name: slug.replace(/-/g, " "), industry: "Company" };
    var store = loadStore();
    var bucket = getCompanyBucket(store, company.slug || slug);
    var sortMode = window.__forumSort || "hot";
    var loggedIn = isLoggedIn();
    var create = loggedIn ? '<div class="reddit-create"><h3>Start a thread</h3><input type="text" id="forumNewTitle" placeholder="Title" maxlength="300"><textarea id="forumNewBody" placeholder="Text — what happened, what helped, what surprised you?"></textarea><div class="reddit-create-actions"><button type="button" class="reddit-btn-primary" id="forumSubmitPost">Post</button></div></div>' : '<div class="reddit-create reddit-login-gate"><h3>Sign in to start a thread</h3><p>Reading is open. Posting, commenting, and votes require an account so history attaches to your profile.</p><button type="button" class="reddit-btn-primary rd-login-required" data-action="post">Sign in to post</button></div>';
    var sortBar = '<div class="reddit-sort"><button type="button" data-sort="hot" class="' + (sortMode === "hot" ? "is-active" : "") + '">🔥 Hot</button><button type="button" data-sort="new" class="' + (sortMode === "new" ? "is-active" : "") + '">✨ New</button><button type="button" data-sort="top" class="' + (sortMode === "top" ? "is-active" : "") + '">⬆️ Top</button></div>';
    var postsHtml = sortPosts(bucket.posts || [], sortMode).map(function (p) {
      var key = targetKey("post", company.slug || slug, p.id);
      var v = getVote(store, key);
      return '<article class="reddit-post" data-post-id="' + attr(p.id) + '"><div class="reddit-votes"><button type="button" class="reddit-vote up rd-p-up' + (v === 1 ? ' is-on' : '') + '" data-slug="' + attr(company.slug || slug) + '" data-post="' + attr(p.id) + '">▲</button><span class="reddit-score">' + esc(p.score) + '</span><button type="button" class="reddit-vote down rd-p-down' + (v === -1 ? ' is-on' : '') + '" data-slug="' + attr(company.slug || slug) + '" data-post="' + attr(p.id) + '">▼</button></div><div class="reddit-post-main"><div class="reddit-post-meta">Posted by <a href="#">' + esc(p.author) + '</a> · ' + timeAgo(p.createdAt) + '</div><h3 class="reddit-post-title">' + esc(p.title) + '</h3>' + (p.body ? '<div class="reddit-post-body">' + esc(p.body) + '</div>' : '') + '<div class="reddit-post-actions"><span style="font-weight:700;color:#878787">' + countComments(p.comments) + ' comments</span></div><div class="reddit-comments"><div class="reddit-comment-form">' + (loggedIn ? '<textarea class="rd-top-comment" placeholder="What are your thoughts?"></textarea><div class="reddit-create-actions"><button type="button" class="reddit-btn-primary rd-add-top-comment" data-post="' + attr(p.id) + '">Comment</button></div>' : '<div class="reddit-login-inline">Sign in to comment. <button type="button" class="reddit-btn-ghost rd-login-required" data-action="comment">Sign in</button></div>') + '</div><div class="reddit-comment-thread">' + renderCommentTree(p.comments || [], company.slug || slug, p.id, store, 0) + '</div></div></div></article>';
    }).join("") || '<p class="reddit-empty">No threads yet — be the first to post.</p>';
    root.innerHTML = '<div class="reddit-top"><a href="#forums" class="reddit-back">← All forums</a><a href="#" class="reddit-back">Job listings</a></div><div class="reddit-sub-header"><div class="reddit-sub-banner"></div><div class="reddit-sub-bar"><div class="forum-icon-slot reddit-sub-avatar">' + logoHtml(company, 64) + '</div><div class="reddit-sub-info"><h2><a href="#company/' + attr(company.slug || slug) + '">' + esc(company.name) + '</a></h2><p class="reddit-sub-meta"><strong>Company forum</strong> · Fortune ' + (company.rank ? '#' + esc(company.rank) : '?') + ' · ' + esc(company.industry || 'Company') + ' · ' + esc(company.headquarters || 'community hiring data') + '</p></div></div></div><div class="reddit-layout"><div>' + renderOutcomePanel(company) + create + sortBar + '<div class="reddit-feed">' + postsHtml + '</div></div><aside class="reddit-sidebar"><h4>About this forum</h4><p>Posts, comments, upvotes, and downvotes are persisted locally and tied to the signed-in uid when available.</p><p><strong>Company metadata</strong><br>Industry: ' + esc(company.industry || '—') + '<br>Revenue: $' + fmtNum(company.revenueMillions) + 'M<br>Employees: ' + fmtNum(company.employees) + '<br>HQ: ' + esc(company.headquarters || '—') + '</p><p>' + (loggedIn ? 'Signed in as <strong>' + esc(displayName()) + '</strong>.' : 'Sign in to post, comment, or vote.') + '</p><ul><li>Be useful</li><li>No doxxing</li><li>Share signal, not spam</li></ul></aside></div>';
    bindCompanyEvents(company.slug || slug);
  }

  function bindCompanyEvents(slug) {
    var root = $("forumRoot"); if (!root) return;
    var submit = $("forumSubmitPost");
    if (submit) submit.addEventListener("click", function () { addPost(slug, $("forumNewTitle") && $("forumNewTitle").value, $("forumNewBody") && $("forumNewBody").value); });
    root.querySelectorAll(".rd-login-required").forEach(function (btn) { btn.addEventListener("click", function () { promptLogin(btn.getAttribute("data-action") || "post"); }); });
    root.querySelectorAll("[data-sort]").forEach(function (btn) { btn.addEventListener("click", function () { window.__forumSort = btn.getAttribute("data-sort"); route(); }); });
    root.querySelectorAll(".rd-p-up").forEach(function (btn) { btn.addEventListener("click", function () { votePost(btn.dataset.slug, btn.dataset.post, 1); }); });
    root.querySelectorAll(".rd-p-down").forEach(function (btn) { btn.addEventListener("click", function () { votePost(btn.dataset.slug, btn.dataset.post, -1); }); });
    root.querySelectorAll(".rd-c-up").forEach(function (btn) { btn.addEventListener("click", function () { voteComment(btn.dataset.slug, btn.dataset.post, btn.dataset.cid, 1); }); });
    root.querySelectorAll(".rd-c-down").forEach(function (btn) { btn.addEventListener("click", function () { voteComment(btn.dataset.slug, btn.dataset.post, btn.dataset.cid, -1); }); });
    root.querySelectorAll(".rd-add-top-comment").forEach(function (btn) { btn.addEventListener("click", function () { var ta = btn.closest(".reddit-comments").querySelector(".rd-top-comment"); addComment(slug, btn.dataset.post, ta && ta.value, null); }); });
    root.querySelectorAll(".rd-reply").forEach(function (btn) { btn.addEventListener("click", function () { var box = root.querySelector('.rd-reply-box[data-reply-for="' + CSS.escape(btn.dataset.parent) + '"]'); if (box) box.classList.toggle("hidden"); }); });
    root.querySelectorAll(".rd-send-reply").forEach(function (btn) { btn.addEventListener("click", function () { var article = btn.closest(".reddit-post"); var box = btn.closest(".rd-reply-box"); var ta = box && box.querySelector("textarea"); addComment(slug, article && article.dataset.postId, ta && ta.value, btn.dataset.parent); }); });
  }

  function route() {
    var r = parseRoute();
    setShell(r);
    if (r.type === "jobs") return;
    waitForFortune().then(function () {
      if (r.type === "hub") renderHub();
      else if (r.type === "company" && r.slug) renderCompanyPage(r.slug);
    });
  }

  function bindNav() {
    var forums = $("navForums"), jobs = $("navJobs");
    if (forums && !forums.__hcForumBound) { forums.__hcForumBound = true; forums.addEventListener("click", function (e) { e.preventDefault(); location.hash = "forums"; route(); }); }
    if (jobs && !jobs.__hcForumBound) { jobs.__hcForumBound = true; jobs.addEventListener("click", function (e) { e.preventDefault(); location.hash = ""; route(); }); }
    var input = $("searchInput");
    if (input && !input.__hcForumSearchBound) { input.__hcForumSearchBound = true; input.addEventListener("keydown", function (e) { if (e.key !== "Enter" || !location.hash.includes("forums")) return; var hit = searchCompanies(input.value.trim(), 1)[0]; if (hit) { e.preventDefault(); location.hash = "company/" + encodeURIComponent(hit.slug); } }); }
  }

  function init() {
    bindNav();
    window.addEventListener("hashchange", route);
    window.addEventListener("hiringcafe:authchange", route);
    window.addEventListener("hiringcafe:showforums", route);
    waitForFortune().then(route);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
