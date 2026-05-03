/**
 * Company forums — hash routes #f (hub) and #f/:slug (Reddit-style threads, localStorage).
 */
(function () {
  var STORAGE_KEY = "himmj_forum_v1";
  var USER_KEY = "himmj_forum_display_name";

  function $(id) {
    return document.getElementById(id);
  }

  function debounce(fn, ms) {
    var t;
    return function () {
      var a = arguments;
      clearTimeout(t);
      t = setTimeout(function () {
        fn.apply(null, a);
      }, ms);
    };
  }

  function esc(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function uid(prefix) {
    return prefix + "-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 7);
  }

  function getDisplayName() {
    try {
      var n = localStorage.getItem(USER_KEY);
      if (n) return n;
      var animals = ["RedPanda", "Capybara", "Quokka", "Manatee", "Axolotl", "Dolphin", "Fennec", "Lynx"];
      n = "u/" + animals[Math.floor(Math.random() * animals.length)] + Math.floor(Math.random() * 900 + 100);
      localStorage.setItem(USER_KEY, n);
      return n;
    } catch (e) {
      return "u/guest";
    }
  }

  function loadStore() {
    try {
      var j = localStorage.getItem(STORAGE_KEY);
      return j ? JSON.parse(j) : {};
    } catch (e) {
      return {};
    }
  }

  function saveStore(obj) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(obj));
    } catch (e) {}
  }

  function getCompanyData(slug) {
    var d = loadStore();
    if (!d[slug]) d[slug] = { posts: [] };
    return d[slug];
  }

  function persistCompany(slug, data) {
    var d = loadStore();
    d[slug] = data;
    saveStore(d);
  }

  function parseRoute() {
    var raw = (location.hash || "").replace(/^#/, "").replace(/^!/, "");
    if (raw === "f" || raw === "f/") return { type: "forum-hub" };
    if (raw.indexOf("f/") === 0) {
      var slug = raw.slice(2).split("/")[0];
      if (slug) return { type: "forum-company", slug: decodeURIComponent(slug) };
    }
    return { type: "jobs" };
  }

  function setActiveNav(route) {
    var jobs = $("navJobs");
    var forums = $("navForums");
    if (!jobs || !forums) return;
    var isForum = route.type === "forum-hub" || route.type === "forum-company";
    jobs.classList.toggle("is-active", !isForum);
    forums.classList.toggle("is-active", isForum);
  }

  function toggleShellForRoute(route) {
    var filters = document.querySelector(".filters");
    var hire = document.querySelector(".hire-banner");
    var topSearch = document.querySelector(".topbar .search-box");
    var toolbar = document.querySelector(".toolbar");
    var isForum = route.type === "forum-hub" || route.type === "forum-company";
    if (filters) filters.style.display = isForum ? "none" : "";
    if (hire) hire.style.display = isForum ? "none" : "";
    if (topSearch) topSearch.style.display = isForum ? "none" : "";
    if (toolbar) toolbar.style.display = isForum ? "none" : "";
  }

  function showJobsMain() {
    var jm = $("jobsMain");
    var fv = $("forumView");
    if (jm) jm.classList.remove("hidden");
    if (fv) fv.classList.add("hidden");
  }

  function showForumView() {
    var jm = $("jobsMain");
    var fv = $("forumView");
    if (jm) jm.classList.add("hidden");
    if (fv) fv.classList.remove("hidden");
  }

  function timeAgo(ts) {
    var s = Math.floor((Date.now() - ts) / 1000);
    if (s < 60) return "just now";
    var m = Math.floor(s / 60);
    if (m < 60) return m + " min ago";
    var h = Math.floor(m / 60);
    if (h < 48) return h + " hr ago";
    var d = Math.floor(h / 24);
    return d + " days ago";
  }

  function mountForumIcon(el, company, size) {
    if (!el || !window.Fortune500 || !company) return;
    el.innerHTML = window.Fortune500.iconMarkSvg(company, size);
    var svg = el.querySelector("svg");
    if (svg) {
      svg.setAttribute("width", String(size));
      svg.setAttribute("height", String(size));
    }
  }

  function votePost(slug, postId, delta) {
    var d = loadStore();
    var bucket = d[slug];
    if (!bucket || !bucket.posts) return;
    var post = bucket.posts.find(function (p) {
      return p.id === postId;
    });
    if (!post) return;
    var prev = post.userVote || 0;
    if (delta === prev) {
      post.score -= prev;
      post.userVote = 0;
    } else {
      post.score += delta - prev;
      post.userVote = delta;
    }
    saveStore(d);
    route();
  }

  function voteComment(slug, postId, commentId, delta) {
    var d = loadStore();
    var bucket = d[slug];
    if (!bucket || !bucket.posts) return;
    var post = bucket.posts.find(function (p) {
      return p.id === postId;
    });
    if (!post) return;
    var c = findComment(post.comments, commentId);
    if (!c) return;
    var prev = c.userVote || 0;
    if (delta === prev) {
      c.score -= prev;
      c.userVote = 0;
    } else {
      c.score += delta - prev;
      c.userVote = delta;
    }
    saveStore(d);
    route();
  }

  function findComment(list, id) {
    for (var i = 0; i < list.length; i++) {
      if (list[i].id === id) return list[i];
      var sub = findComment(list[i].replies || [], id);
      if (sub) return sub;
    }
    return null;
  }

  function addComment(slug, postId, body, parentId) {
    body = String(body || "").trim();
    if (!body) return;
    var d = loadStore();
    if (!d[slug]) d[slug] = { posts: [] };
    var post = d[slug].posts.find(function (p) {
      return p.id === postId;
    });
    if (!post) return;
    var c = {
      id: uid("c"),
      body: body,
      author: getDisplayName(),
      createdAt: Date.now(),
      score: 1,
      userVote: 1,
      parentId: parentId || null,
      replies: [],
    };
    if (parentId) {
      var parent = findComment(post.comments, parentId);
      if (parent) {
        parent.replies = parent.replies || [];
        parent.replies.push(c);
      } else {
        post.comments = post.comments || [];
        post.comments.push(c);
      }
    } else {
      post.comments = post.comments || [];
      post.comments.push(c);
    }
    saveStore(d);
    route();
  }

  function addPost(slug, title, body) {
    title = String(title || "").trim();
    body = String(body || "").trim();
    if (!title) return;
    var d = loadStore();
    if (!d[slug]) d[slug] = { posts: [] };
    d[slug].posts.unshift({
      id: uid("p"),
      title: title,
      body: body,
      author: getDisplayName(),
      createdAt: Date.now(),
      score: 1,
      userVote: 1,
      comments: [],
    });
    saveStore(d);
    route();
  }

  function renderCommentTree(comments, slug, postId, depth) {
    if (!comments || !comments.length) return "";
    depth = depth || 0;
    var html = "";
    for (var i = 0; i < comments.length; i++) {
      var c = comments[i];
      var nest = renderCommentTree(c.replies || [], slug, postId, depth + 1);
      var upOn = c.userVote === 1 ? " is-on" : "";
      var downOn = c.userVote === -1 ? " is-on" : "";
      html +=
        '<div class="reddit-comment" data-depth="' +
        depth +
        '">' +
        '<div class="reddit-comment-bar" aria-hidden="true"></div>' +
        '<div class="reddit-comment-body">' +
        '<div class="reddit-comment-meta"><strong>' +
        esc(c.author) +
        "</strong> · " +
        timeAgo(c.createdAt) +
        " · " +
        c.score +
        " pts</div>" +
        '<div class="reddit-comment-text">' +
        esc(c.body) +
        "</div>" +
        '<div class="reddit-post-actions">' +
        '<button type="button" class="rd-c-up' +
        upOn +
        '" data-slug="' +
        esc(slug) +
        '" data-post="' +
        esc(postId) +
        '" data-cid="' +
        esc(c.id) +
        '">▲</button>' +
        '<button type="button" class="rd-c-down' +
        downOn +
        '" data-slug="' +
        esc(slug) +
        '" data-post="' +
        esc(postId) +
        '" data-cid="' +
        esc(c.id) +
        '">▼</button>' +
        '<button type="button" class="rd-reply" data-parent="' +
        esc(c.id) +
        '">Reply</button>' +
        "</div>" +
        '<div class="rd-reply-box hidden" data-reply-for="' +
        esc(c.id) +
        '">' +
        '<textarea placeholder="Wholesome reply…"></textarea>' +
        '<button type="button" class="reddit-btn-primary rd-send-reply" data-parent="' +
        esc(c.id) +
        '">Reply</button>' +
        "</div>" +
        (nest ? '<div class="reddit-nested">' + nest + "</div>" : "") +
        "</div></div>";
    }
    return html;
  }

  function sortPosts(posts, mode) {
    var copy = posts.slice();
    if (mode === "new") copy.sort(function (a, b) {
      return b.createdAt - a.createdAt;
    });
    else if (mode === "top") copy.sort(function (a, b) {
      return b.score - a.score;
    });
    else
      copy.sort(function (a, b) {
        return b.score + b.comments.length * 2 - (a.score + a.comments.length * 2);
      });
    return copy;
  }

  function renderCompanyPage(slug) {
    var root = $("forumRoot");
    if (!root) return;
    var company = window.Fortune500.getBySlug(slug);
    var display = company ? company.name : slug.replace(/-/g, " ");
    var fslug = company ? company.slug : slug;
    var data = getCompanyData(fslug);
    var sortMode = window.__forumSort || "hot";

    var avatarSlot =
      '<div class="forum-icon-slot reddit-sub-avatar" id="forumPageAvatar"></div>';
    var banner =
      '<div class="reddit-sub-header">' +
      '<div class="reddit-sub-banner"></div>' +
      '<div class="reddit-sub-bar">' +
      avatarSlot +
      '<div class="reddit-sub-info">' +
      "<h2><a href=\"#f/" +
      esc(fslug) +
      '">f/' +
      esc(display.replace(/\s+/g, "")) +
      "</a></h2>" +
      '<p class="reddit-sub-meta"><strong>f/' +
      esc(fslug) +
      "</strong> · Fortune " +
      (company ? "#" + company.rank : "?") +
      " · a cozy corner to swap interview stories 🌸</p>" +
      "</div></div></div>";

    var sortBar =
      '<div class="reddit-sort">' +
      '<button type="button" data-sort="hot" class="' +
      (sortMode === "hot" ? "is-active" : "") +
      '">🔥 Hot</button>' +
      '<button type="button" data-sort="new" class="' +
      (sortMode === "new" ? "is-active" : "") +
      '">✨ New</button>' +
      '<button type="button" data-sort="top" class="' +
      (sortMode === "top" ? "is-active" : "") +
      '">⬆️ Top</button>' +
      "</div>";

    var create =
      '<div class="reddit-create">' +
      "<h3>Start a thread</h3>" +
      '<input type="text" id="forumNewTitle" placeholder="Title" maxlength="300" />' +
      '<textarea id="forumNewBody" placeholder="Text (optional) — how did it go?"></textarea>' +
      '<div class="reddit-create-actions">' +
      '<button type="button" class="reddit-btn-primary" id="forumSubmitPost">Post</button>' +
      "</div></div>";

    var posts = sortPosts(data.posts, sortMode);
    var postsHtml = posts
      .map(function (p) {
        var upOn = p.userVote === 1 ? " is-on" : "";
        var downOn = p.userVote === -1 ? " is-on" : "";
        var commentsHtml = renderCommentTree(p.comments || [], fslug, p.id, 0);
        return (
          '<article class="reddit-post" data-post-id="' +
          esc(p.id) +
          '">' +
          '<div class="reddit-votes">' +
          '<button type="button" class="reddit-vote up rd-p-up' +
          upOn +
          '" data-slug="' +
          esc(fslug) +
          '" data-post="' +
          esc(p.id) +
          '">▲</button>' +
          '<span class="reddit-score">' +
          p.score +
          "</span>" +
          '<button type="button" class="reddit-vote down rd-p-down' +
          downOn +
          '" data-slug="' +
          esc(fslug) +
          '" data-post="' +
          esc(p.id) +
          '">▼</button>' +
          "</div>" +
          '<div class="reddit-post-main">' +
          '<div class="reddit-post-meta">Posted by <a href="#">' +
          esc(p.author) +
          "</a> · " +
          timeAgo(p.createdAt) +
          "</div>" +
          '<h3 class="reddit-post-title">' +
          esc(p.title) +
          "</h3>" +
          (p.body
            ? '<div class="reddit-post-body">' + esc(p.body) + "</div>"
            : "") +
          '<div class="reddit-post-actions">' +
          '<span style="font-weight:700;color:#878787">' +
          (p.comments ? countComments(p.comments) : 0) +
          " comments</span>" +
          "</div>" +
          '<div class="reddit-comments">' +
          '<div class="reddit-comment-form">' +
          '<textarea class="rd-top-comment" placeholder="What are your thoughts? Join the conversation 💬"></textarea>' +
          '<div class="reddit-create-actions">' +
          '<button type="button" class="reddit-btn-primary rd-add-top-comment" data-post="' +
          esc(p.id) +
          '">Comment</button>' +
          "</div></div>" +
          '<div class="reddit-comment-thread">' +
          commentsHtml +
          "</div></div></div></article>"
        );
      })
      .join("");

    if (!postsHtml) postsHtml = '<p class="reddit-empty">No threads yet — be the first to post something sweet ✨</p>';

    var sidebar =
      '<aside class="reddit-sidebar">' +
      "<h4>About this forum</h4>" +
      "<p>Posts stay on <strong>your device</strong> in this demo. Same vibe as Reddit threads: vote, reply, nest comments.</p>" +
      "<p>You're posting as <strong>" +
      esc(getDisplayName()) +
      "</strong>.</p>" +
      "<ul><li>Be kind</li><li>No doxxing</li><li>Share signal, not spam</li></ul>" +
      "</aside>";

    root.innerHTML =
      '<div class="reddit-top">' +
      '<a href="#f" class="reddit-back">← All forums</a>' +
      '<a href="#" class="reddit-back">Job listings</a>' +
      "</div>" +
      banner +
      '<div class="reddit-layout">' +
      "<div>" +
      create +
      sortBar +
      '<div class="reddit-feed">' +
      postsHtml +
      "</div></div>" +
      sidebar +
      "</div>";

    var av = $("forumPageAvatar");
    if (av && company) mountForumIcon(av, company, 64);

    $("forumSubmitPost")?.addEventListener("click", function () {
      var t = $("forumNewTitle");
      var b = $("forumNewBody");
      addPost(fslug, t && t.value, b && b.value);
      if (t) t.value = "";
      if (b) b.value = "";
    });

    root.querySelectorAll("[data-sort]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        window.__forumSort = btn.getAttribute("data-sort");
        route();
      });
    });

    root.querySelectorAll(".rd-p-up").forEach(function (btn) {
      btn.addEventListener("click", function () {
        votePost(btn.getAttribute("data-slug"), btn.getAttribute("data-post"), 1);
      });
    });
    root.querySelectorAll(".rd-p-down").forEach(function (btn) {
      btn.addEventListener("click", function () {
        votePost(btn.getAttribute("data-slug"), btn.getAttribute("data-post"), -1);
      });
    });
    root.querySelectorAll(".rd-c-up").forEach(function (btn) {
      btn.addEventListener("click", function () {
        voteComment(btn.getAttribute("data-slug"), btn.getAttribute("data-post"), btn.getAttribute("data-cid"), 1);
      });
    });
    root.querySelectorAll(".rd-c-down").forEach(function (btn) {
      btn.addEventListener("click", function () {
        voteComment(btn.getAttribute("data-slug"), btn.getAttribute("data-post"), btn.getAttribute("data-cid"), -1);
      });
    });
    root.querySelectorAll(".rd-add-top-comment").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var postId = btn.getAttribute("data-post");
        var ta = btn.closest(".reddit-comments").querySelector(".rd-top-comment");
        addComment(fslug, postId, ta && ta.value, null);
        if (ta) ta.value = "";
      });
    });
    root.querySelectorAll(".rd-reply").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var pid = btn.getAttribute("data-parent");
        var box = root.querySelector('.rd-reply-box[data-reply-for="' + pid + '"]');
        if (box) box.classList.toggle("hidden");
      });
    });
    root.querySelectorAll(".rd-send-reply").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var parent = btn.getAttribute("data-parent");
        var box = btn.closest(".rd-reply-box");
        var ta = box && box.querySelector("textarea");
        var article = btn.closest(".reddit-post");
        var postId = article && article.getAttribute("data-post-id");
        addComment(fslug, postId, ta && ta.value, parent);
        if (ta) ta.value = "";
        if (box) box.classList.add("hidden");
      });
    });
  }

  function countComments(arr) {
    var n = arr.length;
    for (var i = 0; i < arr.length; i++) {
      n += countComments(arr[i].replies || []);
    }
    return n;
  }

  function renderHub() {
    var root = $("forumRoot");
    if (!root) return;
    var list = window.Fortune500.list().slice(0, 60);
    var cards = list
      .map(function (c) {
        return (
          '<a class="reddit-hub-card" href="#f/' +
          encodeURIComponent(c.slug) +
          '">' +
          '<span class="forum-icon-slot fc-mini" data-slug="' +
          esc(c.slug) +
          '"></span>' +
          "<span>f/" +
          esc(c.name.replace(/\s+/g, "")) +
          "</span>" +
          '<span class="fc-rank">#' +
          c.rank +
          "</span></a>"
        );
      })
      .join("");

    root.innerHTML =
      '<div class="reddit-top">' +
      '<a href="#" class="reddit-back">← Job listings</a>' +
      '<div class="reddit-hub-title">' +
      "<h1>Company forums</h1>" +
      "<p>Pick a Fortune 500 — each has its own f/ page. Cute Reddit vibes, local only.</p>" +
      "</div>" +
      '<div class="forum-co-search-wrap">' +
      '<span class="forum-co-search-icon">⌕</span>' +
      '<input type="search" class="forum-co-search" id="forumHubSearch" placeholder="Search companies…" autocomplete="off" />' +
      '<ul class="forum-co-results" id="forumHubResults"></ul>' +
      "</div></div>" +
      '<div class="reddit-hub-grid">' +
      cards +
      "</div>";

    list.forEach(function (c) {
      var slot = root.querySelector('.fc-mini[data-slug="' + c.slug + '"]');
      if (slot) mountForumIcon(slot, c, 36);
    });

    var inp = $("forumHubSearch");
    var res = $("forumHubResults");
    var search = debounce(function () {
      var q = (inp && inp.value) || "";
      if (!res) return;
      res.innerHTML = "";
      if (q.length < 1) return;
      var hits = window.Fortune500.searchCompanies(q, 12);
      hits.forEach(function (c) {
        var li = document.createElement("li");
        var b = document.createElement("button");
        b.type = "button";
        var icon = document.createElement("span");
        icon.className = "forum-icon-slot";
        mountForumIcon(icon, c, 32);
        var lab = document.createElement("span");
        lab.textContent = c.name;
        var rk = document.createElement("span");
        rk.className = "fc-rank";
        rk.textContent = "#" + c.rank;
        b.appendChild(icon);
        b.appendChild(lab);
        b.appendChild(rk);
        b.addEventListener("click", function () {
          location.hash = "f/" + encodeURIComponent(c.slug);
        });
        li.appendChild(b);
        res.appendChild(li);
      });
    }, 200);
    if (inp) inp.addEventListener("input", search);
  }

  function route() {
    var r = parseRoute();
    setActiveNav(r);
    toggleShellForRoute(r);
    if (r.type === "jobs") {
      showJobsMain();
      return;
    }
    showForumView();
    if (!window.Fortune500 || !window.Fortune500.list().length) {
      window.Fortune500.ready().then(function () {
        continueForumRoute(r);
      });
      return;
    }
    continueForumRoute(r);
  }

  function continueForumRoute(r) {
    if (r.type === "forum-hub") renderHub();
    else if (r.type === "forum-company") renderCompanyPage(r.slug);
  }

  function init() {
    window.addEventListener("hashchange", route);
    if (window.Fortune500) {
      window.Fortune500.ready().then(route).catch(route);
    } else {
      route();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
