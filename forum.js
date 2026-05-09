/**
 * Reddit-style company forums.
 *
 * Production mode uses Firebase Auth + Firestore collections:
 * - users/{uid}
 * - forumPosts/{postId}
 * - forumComments/{commentId}
 * - forumVotes/{targetType_targetId_uid}
 *
 * Development mode keeps the same UX working from localStorage.
 */
(function () {
  var STORAGE_KEY = "hc_company_forums_v3";
  var LEGACY_KEYS = ["hc_company_forums_v2", "himmj_forum_v1"];
  var LOCAL_USER_KEY = "hc_account_v1";
  var GUEST_KEY = "hc_forum_guest_uid";
  var MAX_POSTS = 75;
  var MAX_COMMENTS = 700;
  var MAX_VOTES = 1000;
  var routeSeq = 0;

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c];
    });
  }
  function attr(s) { return esc(s).replace(/'/g, "&#39;"); }
  function uid(prefix) { return prefix + "-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8); }
  function debounce(fn, ms) {
    var t;
    return function () {
      var args = arguments;
      clearTimeout(t);
      t = setTimeout(function () { fn.apply(null, args); }, ms);
    };
  }
  function cleanText(value, max) {
    return String(value || "").replace(/\s+\n/g, "\n").replace(/[ \t]{2,}/g, " ").trim().slice(0, max || 2000);
  }
  function toMillis(value) {
    if (!value) return Date.now();
    if (typeof value === "number") return value;
    if (typeof value.toMillis === "function") return value.toMillis();
    var parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : Date.now();
  }
  function readJson(key, fallback) {
    try {
      var raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) {
      return fallback;
    }
  }

  function auth() { return window.HiringCafeAuth || {}; }
  function firebaseApi() {
    var api = auth();
    if (api.getFirebase) return api.getFirebase();
    return window.firebase && firebase.apps && firebase.apps.length ? firebase : null;
  }
  function db() {
    var api = auth();
    if (api.getDb) return api.getDb();
    var fb = firebaseApi();
    return fb ? fb.firestore() : null;
  }
  function fieldValue() {
    var fb = firebaseApi();
    return fb && fb.firestore && fb.firestore.FieldValue ? fb.firestore.FieldValue : null;
  }

  function getUser() {
    var api = auth();
    var u = api.getUser && api.getUser();
    if (u) return u;
    return readJson(LOCAL_USER_KEY, null);
  }
  function getProfile() {
    var api = auth();
    var profile = api.getProfile && api.getProfile();
    return profile || null;
  }
  function isLoggedIn() { return !!getUser(); }
  function isLocalUser() {
    var u = getUser();
    return !!(u && u.local && !u.google);
  }
  function useCloudReads() { return !!db() && !isLocalUser(); }
  function useCloudWrites() {
    var u = getUser();
    return !!(db() && u && u.google);
  }
  function getGuestId() {
    var id = localStorage.getItem(GUEST_KEY);
    if (!id) {
      id = "guest-" + Math.random().toString(36).slice(2, 10);
      localStorage.setItem(GUEST_KEY, id);
    }
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
  function authorHeadline() {
    var p = getProfile();
    return p && p.headline ? p.headline : "";
  }
  function promptLogin(action) {
    var msg = "Join or sign in to " + action + " and keep your posts, comments, and votes tied to your profile.";
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

  function defaultStore() {
    return { companies: {}, votes: {}, createdAt: Date.now(), updatedAt: Date.now() };
  }
  function normalizeComment(c) {
    c = c || {};
    var authorName = c.authorName || c.author || "Community Member";
    return {
      id: c.id || uid("c"),
      postId: c.postId || null,
      companySlug: c.companySlug || null,
      body: c.body || "",
      author: authorName,
      authorName: authorName,
      authorUid: c.authorUid || null,
      authorHeadline: c.authorHeadline || "",
      createdAt: toMillis(c.createdAt),
      score: Number(c.score || 0),
      parentId: c.parentId || null,
      status: c.status || "active",
      replies: (c.replies || []).map(normalizeComment)
    };
  }
  function normalizePost(p) {
    p = p || {};
    var authorName = p.authorName || p.author || "Community Member";
    return {
      id: p.id || uid("p"),
      companySlug: p.companySlug || null,
      title: p.title || "Untitled thread",
      body: p.body || "",
      author: authorName,
      authorName: authorName,
      authorUid: p.authorUid || null,
      authorHeadline: p.authorHeadline || "",
      createdAt: toMillis(p.createdAt),
      updatedAt: toMillis(p.updatedAt || p.createdAt),
      score: Number(p.score || 0),
      commentCount: Number(p.commentCount || countComments(p.comments || [])),
      status: p.status || "active",
      comments: (p.comments || []).map(normalizeComment)
    };
  }
  function migrateLegacy(store) {
    LEGACY_KEYS.forEach(function (key) {
      var legacy = readJson(key, null);
      if (!legacy || store.__migratedKeys && store.__migratedKeys[key]) return;
      if (legacy.companies) legacy = legacy.companies;
      Object.keys(legacy || {}).forEach(function (slug) {
        if (!legacy[slug] || !Array.isArray(legacy[slug].posts)) return;
        if (!store.companies[slug]) store.companies[slug] = { posts: [] };
        legacy[slug].posts.forEach(function (p) {
          if (store.companies[slug].posts.some(function (x) { return x.id === p.id; })) return;
          p.companySlug = p.companySlug || slug;
          store.companies[slug].posts.push(normalizePost(p));
        });
      });
      store.__migratedKeys = store.__migratedKeys || {};
      store.__migratedKeys[key] = true;
    });
    return store;
  }
  function loadLocalStore() {
    var store = readJson(STORAGE_KEY, defaultStore());
    store = Object.assign(defaultStore(), store || {});
    store.companies = store.companies || {};
    store.votes = store.votes || {};
    return migrateLegacy(store);
  }
  function saveLocalStore(store) {
    store.updatedAt = Date.now();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
    mirrorToAppState(store);
  }
  function getCompanyBucket(store, slug) {
    if (!store.companies[slug]) store.companies[slug] = { posts: [] };
    store.companies[slug].posts = (store.companies[slug].posts || []).map(function (p) {
      p.companySlug = p.companySlug || slug;
      return normalizePost(p);
    });
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
    return (list || []).reduce(function (n, c) {
      return n + 1 + countComments(c.replies || []);
    }, 0);
  }
  function targetKey(type, slug, postId, commentId) {
    return [type, slug, postId || "", commentId || ""].join(":");
  }
  function voteDocId(type, targetId, userUid) {
    return [type, targetId, userUid].join("_").replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 240);
  }
  function getVote(store, key) {
    var id = currentUid();
    return store.votes && store.votes[id] ? Number(store.votes[id][key] || 0) : 0;
  }
  function setLocalVote(store, key, value) {
    var id = currentUid();
    if (!store.votes[id]) store.votes[id] = {};
    if (value) store.votes[id][key] = value;
    else delete store.votes[id][key];
  }
  function applyLocalVote(store, item, key, delta) {
    var prev = getVote(store, key);
    var next = prev === delta ? 0 : delta;
    item.score = Number(item.score || 0) + (next - prev);
    setLocalVote(store, key, next);
  }

  function normalizeCloudPost(docId, data) {
    data = data || {};
    return normalizePost({
      id: data.id || docId,
      companySlug: data.companySlug,
      title: data.title,
      body: data.body,
      authorName: data.authorName,
      authorUid: data.authorUid,
      authorHeadline: data.authorHeadline,
      createdAt: data.createdAt,
      updatedAt: data.updatedAt,
      score: data.score,
      commentCount: data.commentCount,
      status: data.status || "active",
      comments: []
    });
  }
  function normalizeCloudComment(docId, data) {
    data = data || {};
    return normalizeComment({
      id: data.id || docId,
      postId: data.postId,
      companySlug: data.companySlug,
      body: data.body,
      authorName: data.authorName,
      authorUid: data.authorUid,
      authorHeadline: data.authorHeadline,
      createdAt: data.createdAt,
      score: data.score,
      parentId: data.parentId || null,
      status: data.status || "active",
      replies: []
    });
  }
  async function getWithFallback(primaryQuery, fallbackQuery, label) {
    try {
      return await primaryQuery.get();
    } catch (err) {
      console.warn(label + " query fell back to a simpler Firestore query", err);
      if (!fallbackQuery) throw err;
      return fallbackQuery.get();
    }
  }
  function attachComments(posts, comments) {
    var postsById = {};
    var commentsById = {};
    posts.forEach(function (p) {
      p.comments = [];
      postsById[p.id] = p;
    });
    comments
      .filter(function (c) { return c.status === "active" && postsById[c.postId]; })
      .sort(function (a, b) { return a.createdAt - b.createdAt; })
      .forEach(function (c) {
        c.replies = [];
        commentsById[c.id] = c;
      });
    comments
      .filter(function (c) { return c.status === "active" && postsById[c.postId]; })
      .sort(function (a, b) { return a.createdAt - b.createdAt; })
      .forEach(function (c) {
        var parent = c.parentId ? commentsById[c.parentId] : null;
        if (parent) parent.replies.push(c);
        else postsById[c.postId].comments.push(c);
      });
    posts.forEach(function (p) {
      p.commentCount = Math.max(Number(p.commentCount || 0), countComments(p.comments));
    });
  }
  function chunkArray(items, size) {
    var chunks = [];
    for (var i = 0; i < items.length; i += size) chunks.push(items.slice(i, i + size));
    return chunks;
  }
  async function loadCloudCompany(slug) {
    var database = db();
    if (!database) throw new Error("Firestore is not initialized.");

    var postsSnap = await getWithFallback(
      database.collection("forumPosts").where("companySlug", "==", slug).where("status", "==", "active").orderBy("createdAt", "desc").limit(MAX_POSTS),
      database.collection("forumPosts").where("companySlug", "==", slug).where("status", "==", "active").limit(MAX_POSTS),
      "forumPosts"
    );
    var posts = postsSnap.docs
      .map(function (doc) { return normalizeCloudPost(doc.id, doc.data()); })
      .filter(function (p) { return p.status === "active"; })
      .sort(function (a, b) { return b.createdAt - a.createdAt; });
    var postIds = {};
    posts.forEach(function (p) { postIds[p.id] = true; });

    var comments = [];
    if (posts.length) {
      var postIdChunks = chunkArray(posts.map(function (p) { return p.id; }), 10);
      try {
        for (var i = 0; i < postIdChunks.length; i++) {
          var chunk = postIdChunks[i];
          var commentsSnap = await getWithFallback(
            database.collection("forumComments").where("postId", "in", chunk).where("status", "==", "active").limit(Math.ceil(MAX_COMMENTS / postIdChunks.length)),
            database.collection("forumComments").where("postId", "in", chunk).where("status", "==", "active").limit(Math.ceil(MAX_COMMENTS / postIdChunks.length)),
            "forumComments"
          );
          commentsSnap.docs.forEach(function (doc) {
            comments.push(normalizeCloudComment(doc.id, doc.data()));
          });
        }
      } catch (err) {
        console.warn("Post-scoped comments query failed; falling back to company comment scan", err);
        var fallbackCommentsSnap = await getWithFallback(
          database.collection("forumComments").where("companySlug", "==", slug).where("status", "==", "active").orderBy("createdAt", "asc").limit(MAX_COMMENTS),
          database.collection("forumComments").where("companySlug", "==", slug).where("status", "==", "active").limit(MAX_COMMENTS),
          "forumComments fallback"
        );
        comments = fallbackCommentsSnap.docs.map(function (doc) { return normalizeCloudComment(doc.id, doc.data()); });
      }
      comments = comments.filter(function (c) { return postIds[c.postId] && c.status === "active"; });
    }
    attachComments(posts, comments);

    var voteMap = {};
    var user = getUser();
    if (user && user.uid) {
      try {
        var votesSnap = await getWithFallback(
          database.collection("forumVotes").where("userUid", "==", user.uid).where("companySlug", "==", slug).limit(MAX_VOTES),
          database.collection("forumVotes").where("userUid", "==", user.uid).limit(MAX_VOTES),
          "forumVotes"
        );
        votesSnap.docs.forEach(function (doc) {
          var data = doc.data() || {};
          if (data.companySlug !== slug) return;
          var key = data.voteKey || targetKey(data.targetType, slug, data.postId, data.commentId);
          voteMap[key] = Number(data.value || 0);
        });
      } catch (err) {
        console.warn("Could not load current user's forum votes", err);
      }
    }

    var store = defaultStore();
    store.cloud = true;
    store.companies[slug] = { posts: posts };
    if (user && user.uid) store.votes[user.uid] = voteMap;
    return store;
  }
  async function loadCompanyStore(slug) {
    if (!useCloudReads()) return loadLocalStore();
    try {
      return await loadCloudCompany(slug);
    } catch (err) {
      console.warn("Firestore forum read failed; showing local fallback", err);
      var store = loadLocalStore();
      store.cloudError = err.message || String(err);
      return store;
    }
  }

  function mergeStatPayload(fields, userUid) {
    var fv = fieldValue();
    var stats = {};
    var profileStats = {};
    Object.keys(fields || {}).forEach(function (key) {
      var amount = Number(fields[key] || 0);
      if (!amount) return;
      stats[key] = fv ? fv.increment(amount) : amount;
      profileStats[key] = fv ? fv.increment(amount) : amount;
    });
    return { uid: userUid, stats: stats, profile: { stats: profileStats }, updatedAt: Date.now() };
  }
  function addUserStatsToBatch(batch, database, userUid, fields) {
    if (!userUid || !fields) return;
    batch.set(database.collection("users").doc(userUid), mergeStatPayload(fields, userUid), { merge: true });
  }
  function addUserStatsToTransaction(tx, database, userUid, fields) {
    if (!userUid || !fields) return;
    tx.set(database.collection("users").doc(userUid), mergeStatPayload(fields, userUid), { merge: true });
  }
  async function createCloudPost(slug, title, body) {
    var database = db();
    var user = getUser();
    var profile = getProfile() || {};
    var createdAt = Date.now();
    var postRef = database.collection("forumPosts").doc();
    var voteRef = database.collection("forumVotes").doc(voteDocId("post", postRef.id, user.uid));
    var voteKey = targetKey("post", slug, postRef.id);
    var batch = database.batch();
    batch.set(postRef, {
      id: postRef.id,
      companySlug: slug,
      title: title,
      body: body,
      authorUid: user.uid,
      authorName: displayName(),
      authorHeadline: profile.headline || "",
      createdAt: createdAt,
      updatedAt: createdAt,
      score: 1,
      commentCount: 0,
      status: "active"
    });
    batch.set(voteRef, {
      id: voteRef.id,
      userUid: user.uid,
      targetType: "post",
      targetId: postRef.id,
      postId: postRef.id,
      companySlug: slug,
      voteKey: voteKey,
      value: 1,
      createdAt: createdAt,
      updatedAt: createdAt
    });
    addUserStatsToBatch(batch, database, user.uid, { posts: 1, votesCast: 1, karma: 1 });
    await batch.commit();
  }
  async function createCloudComment(slug, postId, body, parentId) {
    var database = db();
    var user = getUser();
    var profile = getProfile() || {};
    var createdAt = Date.now();
    var commentRef = database.collection("forumComments").doc();
    var postRef = database.collection("forumPosts").doc(postId);
    var voteRef = database.collection("forumVotes").doc(voteDocId("comment", commentRef.id, user.uid));
    var voteKey = targetKey("comment", slug, postId, commentRef.id);
    var fv = fieldValue();
    var batch = database.batch();
    batch.set(commentRef, {
      id: commentRef.id,
      companySlug: slug,
      postId: postId,
      parentId: parentId || null,
      body: body,
      authorUid: user.uid,
      authorName: displayName(),
      authorHeadline: profile.headline || "",
      createdAt: createdAt,
      updatedAt: createdAt,
      score: 1,
      status: "active"
    });
    batch.set(voteRef, {
      id: voteRef.id,
      userUid: user.uid,
      targetType: "comment",
      targetId: commentRef.id,
      postId: postId,
      commentId: commentRef.id,
      companySlug: slug,
      voteKey: voteKey,
      value: 1,
      createdAt: createdAt,
      updatedAt: createdAt
    });
    batch.update(postRef, {
      commentCount: fv ? fv.increment(1) : 1,
      updatedAt: createdAt
    });
    addUserStatsToBatch(batch, database, user.uid, { comments: 1, votesCast: 1, karma: 1 });
    await batch.commit();
  }
  async function voteCloud(type, slug, postId, commentId, delta) {
    var database = db();
    var user = getUser();
    var fv = fieldValue();
    var targetId = type === "post" ? postId : commentId;
    var targetRef = type === "post" ? database.collection("forumPosts").doc(postId) : database.collection("forumComments").doc(commentId);
    var voteRef = database.collection("forumVotes").doc(voteDocId(type, targetId, user.uid));
    var voteKey = targetKey(type, slug, postId, commentId);
    var now = Date.now();
    await database.runTransaction(async function (tx) {
      var voteSnap = await tx.get(voteRef);
      var prev = voteSnap.exists ? Number((voteSnap.data() || {}).value || 0) : 0;
      var next = prev === delta ? 0 : delta;
      var diff = next - prev;
      if (next) {
        tx.set(voteRef, {
          id: voteRef.id,
          userUid: user.uid,
          targetType: type,
          targetId: targetId,
          postId: postId,
          commentId: type === "comment" ? commentId : null,
          companySlug: slug,
          voteKey: voteKey,
          value: next,
          updatedAt: now,
          createdAt: voteSnap.exists ? ((voteSnap.data() || {}).createdAt || now) : now
        }, { merge: true });
      } else {
        tx.delete(voteRef);
      }
      if (diff) {
        tx.update(targetRef, {
          score: fv ? fv.increment(diff) : diff,
          updatedAt: now
        });
      }
      var voteCountDiff = !prev && next ? 1 : (prev && !next ? -1 : 0);
      if (voteCountDiff) addUserStatsToTransaction(tx, database, user.uid, { votesCast: voteCountDiff });
    });
  }

  async function addPost(slug, title, body) {
    if (!requireLogin("start a thread")) return;
    title = cleanText(title, 300);
    body = cleanText(body, 6000);
    if (!title) return;
    try {
      if (useCloudWrites()) await createCloudPost(slug, title, body);
      else {
        var store = loadLocalStore();
        var bucket = getCompanyBucket(store, slug);
        var id = uid("p");
        var post = {
          id: id,
          companySlug: slug,
          title: title,
          body: body,
          author: displayName(),
          authorName: displayName(),
          authorUid: currentUid(),
          authorHeadline: authorHeadline(),
          createdAt: Date.now(),
          updatedAt: Date.now(),
          score: 1,
          commentCount: 0,
          comments: []
        };
        bucket.posts.unshift(post);
        setLocalVote(store, targetKey("post", slug, id), 1);
        saveLocalStore(store);
      }
      route();
    } catch (err) {
      showNotice("Could not publish that thread. Check Firestore rules and try again.", err);
    }
  }
  async function addComment(slug, postId, body, parentId) {
    if (!requireLogin(parentId ? "reply" : "comment")) return;
    body = cleanText(body, 4000);
    if (!body) return;
    try {
      if (useCloudWrites()) await createCloudComment(slug, postId, body, parentId);
      else {
        var store = loadLocalStore();
        var post = getCompanyBucket(store, slug).posts.find(function (p) { return p.id === postId; });
        if (!post) return;
        var id = uid("c");
        var comment = {
          id: id,
          postId: postId,
          companySlug: slug,
          body: body,
          author: displayName(),
          authorName: displayName(),
          authorUid: currentUid(),
          authorHeadline: authorHeadline(),
          createdAt: Date.now(),
          score: 1,
          parentId: parentId || null,
          replies: []
        };
        if (parentId) {
          var parent = findComment(post.comments, parentId);
          if (parent) parent.replies.push(comment);
          else post.comments.push(comment);
        } else {
          post.comments.push(comment);
        }
        post.commentCount = countComments(post.comments);
        setLocalVote(store, targetKey("comment", slug, postId, id), 1);
        saveLocalStore(store);
      }
      route();
    } catch (err) {
      showNotice("Could not publish that comment. Check Firestore rules and try again.", err);
    }
  }
  async function votePost(slug, postId, delta) {
    if (!requireLogin(delta > 0 ? "upvote posts" : "downvote posts")) return;
    try {
      if (useCloudWrites()) await voteCloud("post", slug, postId, null, delta);
      else {
        var store = loadLocalStore();
        var post = getCompanyBucket(store, slug).posts.find(function (p) { return p.id === postId; });
        if (!post) return;
        applyLocalVote(store, post, targetKey("post", slug, postId), delta);
        saveLocalStore(store);
      }
      route();
    } catch (err) {
      showNotice("Could not save that vote. Check Firestore rules and try again.", err);
    }
  }
  async function voteComment(slug, postId, commentId, delta) {
    if (!requireLogin(delta > 0 ? "upvote comments" : "downvote comments")) return;
    try {
      if (useCloudWrites()) await voteCloud("comment", slug, postId, commentId, delta);
      else {
        var store = loadLocalStore();
        var post = getCompanyBucket(store, slug).posts.find(function (p) { return p.id === postId; });
        var comment = post && findComment(post.comments, commentId);
        if (!comment) return;
        applyLocalVote(store, comment, targetKey("comment", slug, postId, commentId), delta);
        saveLocalStore(store);
      }
      route();
    } catch (err) {
      showNotice("Could not save that vote. Check Firestore rules and try again.", err);
    }
  }

  function mirrorToAppState(store) {
    if (!window.HiringCafeAuth || !window.HiringCafeAuth.saveForumState) return;
    var flatPosts = {};
    var flatComments = {};
    var postVotes = {};
    var flatVotes = {};
    function walkComments(slug, postId, comments) {
      (comments || []).forEach(function (c) {
        flatComments[c.id] = {
          id: c.id,
          postId: postId,
          companySlug: slug,
          body: c.body,
          authorUid: c.authorUid,
          authorName: c.authorName || c.author,
          createdAt: c.createdAt,
          upvotes: Math.max(0, c.score || 0)
        };
        walkComments(slug, postId, c.replies || []);
      });
    }
    Object.keys(store.companies || {}).forEach(function (slug) {
      (store.companies[slug].posts || []).forEach(function (p) {
        flatPosts[p.id] = {
          id: p.id,
          companySlug: slug,
          title: p.title,
          body: p.body,
          authorUid: p.authorUid,
          authorName: p.authorName || p.author,
          createdAt: p.createdAt,
          upvotes: Math.max(0, p.score || 0)
        };
        postVotes[p.id] = Math.max(0, p.score || 0);
        walkComments(slug, p.id, p.comments || []);
      });
    });
    Object.keys(store.votes || {}).forEach(function (id) {
      Object.keys(store.votes[id] || {}).forEach(function (key) {
        flatVotes[id + ":" + key] = store.votes[id][key];
      });
    });
    window.HiringCafeAuth.saveForumState({
      posts: flatPosts,
      comments: flatComments,
      upvotes: flatVotes,
      postVotes: postVotes
    });
  }

  function parseRoute() {
    var raw = (location.hash || "").replace(/^#/, "").replace(/^!/, "");
    if (raw === "forums" || raw === "forums/") return { type: "hub" };
    if (raw.indexOf("company/") === 0) return { type: "company", slug: decodeURIComponent(raw.slice(8).split("/")[0] || "") };
    if (raw.indexOf("user/") === 0) return { type: "user", uid: decodeURIComponent(raw.slice(5).split("/")[0] || "") };
    if (raw === "f" || raw === "f/") {
      location.replace("#forums");
      return { type: "hub" };
    }
    if (raw.indexOf("f/") === 0) {
      var slug = decodeURIComponent(raw.slice(2).split("/")[0] || "");
      location.replace("#company/" + encodeURIComponent(slug));
      return { type: "company", slug: slug };
    }
    return { type: "jobs" };
  }
  function setShell(route) {
    var isForum = route.type === "hub" || route.type === "company" || route.type === "user";
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
  function fortuneList() {
    return window.Fortune500 && window.Fortune500.list ? window.Fortune500.list() : [];
  }
  function companyBySlug(slug) {
    return window.Fortune500 && window.Fortune500.getBySlug ? window.Fortune500.getBySlug(slug) : null;
  }
  function searchCompanies(q, n) {
    return window.Fortune500 && window.Fortune500.searchCompanies ? window.Fortune500.searchCompanies(q, n) : [];
  }
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
      if (mode === "top") return b.score - a.score || b.createdAt - a.createdAt;
      return (b.score + countComments(b.comments) * 2 + b.createdAt / 1e13) - (a.score + countComments(a.comments) * 2 + a.createdAt / 1e13);
    });
  }
  function fmtNum(n) { return n == null || n === "" ? "-" : Number(n).toLocaleString(); }
  function hashNum(str) {
    var h = 2166136261;
    str = String(str || "");
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24);
    }
    return Math.abs(h >>> 0);
  }
  function demoOutcomes(company) {
    var roles = ["Software Engineer", "Product Manager", "Data Analyst", "Business Analyst", "Operations", "Sales", "Finance", "UX Designer"];
    var locs = ["New York, NY", "San Francisco, CA", "Austin, TX", "Chicago, IL", "Seattle, WA", "Remote", "Boston, MA", "Atlanta, GA"];
    var skills = ["SQL / Excel / Tableau", "Python / APIs / Cloud", "React / TypeScript", "Stakeholder management", "Java / Spring / AWS", "Power BI", "Kubernetes", "Figma / Research"];
    var seed = hashNum(company.slug || company.name);
    return Array.from({ length: 8 }, function (_, i) {
      return {
        sourceIndustry: company.industry || "Fortune 500",
        role: roles[(seed + i) % roles.length],
        location: locs[(seed + i * 3) % locs.length],
        skills: skills[(seed + i * 5) % skills.length],
        yoe: 1 + ((seed + i) % 10),
        outcome: i % 4 === 0 ? "Offer" : i % 3 === 0 ? "Final round" : "Interview",
        timing: (7 + ((seed + i * 11) % 35)) + " days"
      };
    });
  }
  function renderOutcomePanel(company) {
    var rows = demoOutcomes(company);
    var body = rows.map(function (r) {
      return '<tr><td>' + esc(r.sourceIndustry) + '</td><td>' + esc(r.role) + '</td><td>' + esc(r.location) + '</td><td>' + esc(r.skills) + '</td><td>' + esc(r.yoe) + '</td><td><span class="reddit-outcome-pill">' + esc(r.outcome) + '</span></td><td>' + esc(r.timing) + '</td></tr>';
    }).join("");
    return '<section class="reddit-outcomes"><div class="reddit-outcomes-head"><div><h3>Hiring outcomes & metadata</h3><p>GradCafe-style rows for this company. These can move into the same Firestore model as the forum when outcome sharing is ready.</p></div><button type="button" class="reddit-btn-primary" data-open-outcome-survey>Share outcome</button></div><div class="reddit-outcome-table-wrap"><table class="reddit-outcome-table"><thead><tr><th>Hired from</th><th>Role track</th><th>Location</th><th>Skills</th><th>YOE</th><th>Outcome</th><th>Timing</th></tr></thead><tbody>' + body + '</tbody></table></div></section>';
  }
  function authorLink(authorName, authorUid) {
    var label = "u/" + (authorName || "Community Member");
    if (!authorUid) return '<span>' + esc(label) + '</span>';
    return '<a href="#user/' + encodeURIComponent(authorUid) + '">' + esc(label) + '</a>';
  }

  function renderHub() {
    var root = $("forumRoot");
    if (!root) return;
    var list = fortuneList();
    if (!list.length) {
      root.innerHTML = '<div class="reddit-empty">Loading company forums...</div>';
      waitForFortune().then(renderHub);
      return;
    }
    var cards = list.map(function (c) {
      return '<a class="reddit-hub-card" href="#company/' + encodeURIComponent(c.slug) + '"><span class="forum-icon-slot fc-mini">' + logoHtml(c, 36) + '</span><span><strong>' + esc(c.name) + '</strong><small>' + esc(c.industry || "Fortune 500") + '</small></span><span class="fc-rank">#' + esc(c.rank) + '</span></a>';
    }).join("");
    root.innerHTML = '<div class="reddit-top"><a href="#" class="reddit-back">&larr; Job listings</a><div class="reddit-hub-title"><h1>Company forums</h1><p>Choose a company, start threads, comment, and vote. Firebase turns this into shared public forum data; local mode remains available for development.</p><p>Job Data Pool community: <a href="https://www.reddit.com/r/jobdatapool/" target="_blank" rel="noopener noreferrer">r/jobdatapool</a></p></div><div class="forum-co-search-wrap"><span class="forum-co-search-icon">r/</span><input type="search" class="forum-co-search" id="forumHubSearch" placeholder="Search companies..." autocomplete="off"><ul class="forum-co-results" id="forumHubResults"></ul></div></div><div class="reddit-hub-grid">' + cards + '</div>';
    var input = $("forumHubSearch"), results = $("forumHubResults");
    if (input && results) input.addEventListener("input", debounce(function () {
      var q = input.value.trim();
      results.innerHTML = "";
      if (!q) return;
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
      return '<div class="reddit-comment" data-depth="' + (depth || 0) + '"><div class="reddit-comment-bar" aria-hidden="true"></div><div class="reddit-comment-body"><div class="reddit-comment-meta"><strong>' + authorLink(c.authorName || c.author, c.authorUid) + '</strong> ' + (c.authorHeadline ? '<span class="reddit-author-headline">' + esc(c.authorHeadline) + '</span> ' : '') + '&middot; ' + timeAgo(c.createdAt) + ' &middot; ' + esc(c.score) + ' pts</div><div class="reddit-comment-text">' + esc(c.body) + '</div><div class="reddit-post-actions"><button type="button" class="rd-c-up' + (v === 1 ? ' is-on' : '') + '" data-slug="' + attr(slug) + '" data-post="' + attr(postId) + '" data-cid="' + attr(c.id) + '">Upvote</button><button type="button" class="rd-c-down' + (v === -1 ? ' is-on' : '') + '" data-slug="' + attr(slug) + '" data-post="' + attr(postId) + '" data-cid="' + attr(c.id) + '">Downvote</button><button type="button" class="rd-reply" data-parent="' + attr(c.id) + '">Reply</button></div><div class="rd-reply-box hidden" data-reply-for="' + attr(c.id) + '"><textarea placeholder="Reply..."></textarea><button type="button" class="reddit-btn-primary rd-send-reply" data-parent="' + attr(c.id) + '">Reply</button></div>' + (c.replies && c.replies.length ? '<div class="reddit-nested">' + renderCommentTree(c.replies, slug, postId, store, (depth || 0) + 1) + '</div>' : '') + '</div></div>';
    }).join("");
  }

  async function renderCompanyPage(slug, seq) {
    var root = $("forumRoot");
    if (!root) return;
    var company = companyBySlug(slug) || { slug: slug, name: slug.replace(/-/g, " "), industry: "Company" };
    root.innerHTML = '<div class="reddit-empty">Loading r/' + esc(company.slug || slug) + '...</div>';
    var store = await loadCompanyStore(company.slug || slug);
    if (seq !== routeSeq) return;
    var bucket = getCompanyBucket(store, company.slug || slug);
    var sortMode = window.__forumSort || "hot";
    var loggedIn = isLoggedIn();
    var cloud = !!store.cloud;
    var create = loggedIn ? '<div class="reddit-create"><h3>Start a thread</h3><input type="text" id="forumNewTitle" placeholder="Title" maxlength="300"><textarea id="forumNewBody" placeholder="Text: what happened, what helped, what surprised you?"></textarea><div class="reddit-create-actions"><button type="button" class="reddit-btn-primary" id="forumSubmitPost">Post</button></div></div>' : '<div class="reddit-create reddit-login-gate"><h3>Join to start a thread</h3><p>Reading is open. Posting, commenting, and votes need a profile so your history follows you.</p><button type="button" class="reddit-btn-primary rd-login-required" data-action="post">Join or sign in</button></div>';
    var sortBar = '<div class="reddit-sort"><button type="button" data-sort="hot" class="' + (sortMode === "hot" ? "is-active" : "") + '">Hot</button><button type="button" data-sort="new" class="' + (sortMode === "new" ? "is-active" : "") + '">New</button><button type="button" data-sort="top" class="' + (sortMode === "top" ? "is-active" : "") + '">Top</button></div>';
    var postsHtml = sortPosts(bucket.posts || [], sortMode).map(function (p) {
      var key = targetKey("post", company.slug || slug, p.id);
      var v = getVote(store, key);
      var commentsCount = Math.max(Number(p.commentCount || 0), countComments(p.comments || []));
      return '<article class="reddit-post" data-post-id="' + attr(p.id) + '"><div class="reddit-votes"><button type="button" class="reddit-vote up rd-p-up' + (v === 1 ? ' is-on' : '') + '" data-slug="' + attr(company.slug || slug) + '" data-post="' + attr(p.id) + '" aria-label="Upvote">&#9650;</button><span class="reddit-score">' + esc(p.score) + '</span><button type="button" class="reddit-vote down rd-p-down' + (v === -1 ? ' is-on' : '') + '" data-slug="' + attr(company.slug || slug) + '" data-post="' + attr(p.id) + '" aria-label="Downvote">&#9660;</button></div><div class="reddit-post-main"><div class="reddit-post-meta">Posted by ' + authorLink(p.authorName || p.author, p.authorUid) + ' ' + (p.authorHeadline ? '<span class="reddit-author-headline">' + esc(p.authorHeadline) + '</span> ' : '') + '&middot; ' + timeAgo(p.createdAt) + '</div><h3 class="reddit-post-title">' + esc(p.title) + '</h3>' + (p.body ? '<div class="reddit-post-body">' + esc(p.body) + '</div>' : '') + '<div class="reddit-post-actions"><span class="reddit-comment-count">' + commentsCount + ' comments</span></div><div class="reddit-comments"><div class="reddit-comment-form">' + (loggedIn ? '<textarea class="rd-top-comment" placeholder="What are your thoughts?"></textarea><div class="reddit-create-actions"><button type="button" class="reddit-btn-primary rd-add-top-comment" data-post="' + attr(p.id) + '">Comment</button></div>' : '<div class="reddit-login-inline">Join to comment. <button type="button" class="reddit-btn-ghost rd-login-required" data-action="comment">Join or sign in</button></div>') + '</div><div class="reddit-comment-thread">' + renderCommentTree(p.comments || [], company.slug || slug, p.id, store, 0) + '</div></div></div></article>';
    }).join("") || '<p class="reddit-empty">No threads yet. Be the first to post.</p>';
    var modeNotice = store.cloudError ? '<div class="reddit-notice">Firestore read failed, so this view is showing local fallback data.</div>' : "";
    var sidebarCopy = cloud ? "Shared forum mode: posts, comments, and votes are persisted in Firestore and visible to everyone." : "Local development mode: posts, comments, and votes are stored in this browser until Firebase is configured.";
    root.innerHTML = '<div class="reddit-top"><a href="#forums" class="reddit-back">&larr; All forums</a><a href="#" class="reddit-back">Job listings</a></div><div class="reddit-sub-header"><div class="reddit-sub-banner"></div><div class="reddit-sub-bar"><div class="forum-icon-slot reddit-sub-avatar">' + logoHtml(company, 64) + '</div><div class="reddit-sub-info"><h2><a href="#company/' + attr(company.slug || slug) + '">r/' + esc(company.slug || slug) + '</a></h2><p class="reddit-sub-meta"><strong>' + esc(company.name) + '</strong> &middot; Fortune ' + (company.rank ? '#' + esc(company.rank) : '?') + ' &middot; ' + esc(company.industry || 'Company') + ' &middot; ' + esc(company.headquarters || 'community hiring data') + '</p></div></div></div><div class="reddit-layout"><div>' + modeNotice + renderOutcomePanel(company) + create + sortBar + '<div class="reddit-feed">' + postsHtml + '</div></div><aside class="reddit-sidebar"><h4>About this forum</h4><p>' + esc(sidebarCopy) + '</p><p>Job Data Pool on Reddit: <a href="https://www.reddit.com/r/jobdatapool/" target="_blank" rel="noopener noreferrer">r/jobdatapool</a></p><p><strong>Company metadata</strong><br>Industry: ' + esc(company.industry || '-') + '<br>Revenue: $' + fmtNum(company.revenueMillions) + 'M<br>Employees: ' + fmtNum(company.employees) + '<br>HQ: ' + esc(company.headquarters || '-') + '</p><p>' + (loggedIn ? 'Signed in as <strong>' + esc(displayName()) + '</strong>.' : 'Join to post, comment, or vote.') + '</p><ul><li>Be useful</li><li>No doxxing</li><li>Share signal, not spam</li></ul></aside></div>';
    bindCompanyEvents(company.slug || slug);
  }

  function bindCompanyEvents(slug) {
    var root = $("forumRoot");
    if (!root) return;
    var submit = $("forumSubmitPost");
    if (submit) submit.addEventListener("click", function () {
      addPost(slug, $("forumNewTitle") && $("forumNewTitle").value, $("forumNewBody") && $("forumNewBody").value);
    });
    root.querySelectorAll(".rd-login-required").forEach(function (btn) {
      btn.addEventListener("click", function () { promptLogin(btn.getAttribute("data-action") || "post"); });
    });
    root.querySelectorAll("[data-sort]").forEach(function (btn) {
      btn.addEventListener("click", function () { window.__forumSort = btn.getAttribute("data-sort"); route(); });
    });
    root.querySelectorAll(".rd-p-up").forEach(function (btn) {
      btn.addEventListener("click", function () { votePost(btn.dataset.slug, btn.dataset.post, 1); });
    });
    root.querySelectorAll(".rd-p-down").forEach(function (btn) {
      btn.addEventListener("click", function () { votePost(btn.dataset.slug, btn.dataset.post, -1); });
    });
    root.querySelectorAll(".rd-c-up").forEach(function (btn) {
      btn.addEventListener("click", function () { voteComment(btn.dataset.slug, btn.dataset.post, btn.dataset.cid, 1); });
    });
    root.querySelectorAll(".rd-c-down").forEach(function (btn) {
      btn.addEventListener("click", function () { voteComment(btn.dataset.slug, btn.dataset.post, btn.dataset.cid, -1); });
    });
    root.querySelectorAll(".rd-add-top-comment").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var ta = btn.closest(".reddit-comments").querySelector(".rd-top-comment");
        addComment(slug, btn.dataset.post, ta && ta.value, null);
      });
    });
    root.querySelectorAll(".rd-reply").forEach(function (btn) {
      btn.addEventListener("click", function () {
        root.querySelectorAll(".rd-reply-box").forEach(function (box) {
          if (box.getAttribute("data-reply-for") === btn.dataset.parent) box.classList.toggle("hidden");
        });
      });
    });
    root.querySelectorAll(".rd-send-reply").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var article = btn.closest(".reddit-post");
        var box = btn.closest(".rd-reply-box");
        var ta = box && box.querySelector("textarea");
        addComment(slug, article && article.dataset.postId, ta && ta.value, btn.dataset.parent);
      });
    });
  }

  function profileFromUserDoc(uid, data) {
    data = data || {};
    var profile = data.profile || {};
    var stats = Object.assign({ posts: 0, comments: 0, votesCast: 0, karma: 0 }, profile.stats || {}, data.stats || {});
    return {
      uid: uid,
      displayName: profile.displayName || data.displayName || "Community Member",
      headline: profile.headline || "",
      bio: profile.bio || "",
      stats: stats,
      updatedAt: toMillis(profile.updatedAt || data.updatedAt)
    };
  }
  async function loadCloudUserActivity(userUid) {
    var database = db();
    var profileSnap = await database.collection("users").doc(userUid).get();
    var profile = profileFromUserDoc(userUid, profileSnap.exists ? profileSnap.data() : {});
    var posts = [];
    var comments = [];
    try {
      var postsSnap = await getWithFallback(
        database.collection("forumPosts").where("authorUid", "==", userUid).where("status", "==", "active").orderBy("createdAt", "desc").limit(20),
        database.collection("forumPosts").where("authorUid", "==", userUid).where("status", "==", "active").limit(20),
        "user forumPosts"
      );
      posts = postsSnap.docs.map(function (doc) { return normalizeCloudPost(doc.id, doc.data()); }).filter(function (p) { return p.status === "active"; }).sort(function (a, b) { return b.createdAt - a.createdAt; });
    } catch (err) {
      console.warn("Could not load user posts", err);
    }
    try {
      var commentsSnap = await getWithFallback(
        database.collection("forumComments").where("authorUid", "==", userUid).where("status", "==", "active").orderBy("createdAt", "desc").limit(20),
        database.collection("forumComments").where("authorUid", "==", userUid).where("status", "==", "active").limit(20),
        "user forumComments"
      );
      comments = commentsSnap.docs.map(function (doc) { return normalizeCloudComment(doc.id, doc.data()); }).filter(function (c) { return c.status === "active"; }).sort(function (a, b) { return b.createdAt - a.createdAt; });
    } catch (err) {
      console.warn("Could not load user comments", err);
    }
    return { profile: profile, posts: posts, comments: comments, cloud: true };
  }
  function walkLocalComments(comments, userUid, out) {
    (comments || []).forEach(function (c) {
      if (c.authorUid === userUid) out.push(c);
      walkLocalComments(c.replies || [], userUid, out);
    });
  }
  function loadLocalUserActivity(userUid) {
    var store = loadLocalStore();
    var posts = [];
    var comments = [];
    Object.keys(store.companies || {}).forEach(function (slug) {
      (store.companies[slug].posts || []).forEach(function (p) {
        p = normalizePost(p);
        p.companySlug = p.companySlug || slug;
        if (p.authorUid === userUid) posts.push(p);
        walkLocalComments(p.comments || [], userUid, comments);
      });
    });
    var current = getProfile() || {};
    var profile = {
      uid: userUid,
      displayName: current.uid === userUid ? current.displayName : "Community Member",
      headline: current.uid === userUid ? current.headline : "",
      bio: current.uid === userUid ? current.bio : "",
      stats: { posts: posts.length, comments: comments.length, votesCast: 0, karma: 0 }
    };
    posts.sort(function (a, b) { return b.createdAt - a.createdAt; });
    comments.sort(function (a, b) { return b.createdAt - a.createdAt; });
    return { profile: profile, posts: posts.slice(0, 20), comments: comments.slice(0, 20), cloud: false };
  }
  async function renderUserProfile(userUid, seq) {
    var root = $("forumRoot");
    if (!root) return;
    root.innerHTML = '<div class="reddit-empty">Loading profile...</div>';
    var data;
    try {
      data = useCloudReads() ? await loadCloudUserActivity(userUid) : loadLocalUserActivity(userUid);
    } catch (err) {
      console.warn("Could not load profile", err);
      data = loadLocalUserActivity(userUid);
      data.cloudError = err.message || String(err);
    }
    if (seq !== routeSeq) return;
    var p = data.profile;
    var stats = p.stats || {};
    var postsHtml = data.posts.length ? data.posts.map(function (post) {
      var company = companyBySlug(post.companySlug) || { name: post.companySlug || "Company" };
      return '<a class="reddit-profile-activity" href="#company/' + encodeURIComponent(post.companySlug || "") + '"><strong>' + esc(post.title) + '</strong><span>r/' + esc(post.companySlug || "company") + ' &middot; ' + esc(company.name || "") + ' &middot; ' + timeAgo(post.createdAt) + ' &middot; ' + esc(post.score) + ' pts</span></a>';
    }).join("") : '<p class="reddit-empty">No public threads yet.</p>';
    var commentsHtml = data.comments.length ? data.comments.map(function (comment) {
      return '<a class="reddit-profile-activity" href="#company/' + encodeURIComponent(comment.companySlug || "") + '"><strong>' + esc(comment.body.slice(0, 120)) + '</strong><span>Comment in r/' + esc(comment.companySlug || "company") + ' &middot; ' + timeAgo(comment.createdAt) + ' &middot; ' + esc(comment.score) + ' pts</span></a>';
    }).join("") : '<p class="reddit-empty">No public comments yet.</p>';
    var notice = data.cloudError ? '<div class="reddit-notice">Profile cloud read failed, so this view is showing local fallback data.</div>' : "";
    root.innerHTML = '<div class="reddit-top"><a href="#forums" class="reddit-back">&larr; Forums</a><a href="#" class="reddit-back">Job listings</a></div>' + notice + '<section class="reddit-profile-page"><div class="reddit-profile-hero"><div class="reddit-profile-avatar">' + esc((p.displayName || "U").slice(0, 2).toUpperCase()) + '</div><div><p class="reddit-profile-kicker">Public profile</p><h1>u/' + esc(p.displayName || "Community Member") + '</h1><p>' + esc(p.headline || "Community member") + '</p>' + (p.bio ? '<div class="reddit-profile-bio">' + esc(p.bio) + '</div>' : '') + '</div></div><div class="reddit-profile-stats"><span><strong>' + esc(stats.posts || data.posts.length || 0) + '</strong> posts</span><span><strong>' + esc(stats.comments || data.comments.length || 0) + '</strong> comments</span><span><strong>' + esc(stats.votesCast || 0) + '</strong> votes</span><span><strong>' + esc(stats.karma || 0) + '</strong> karma</span></div><div class="reddit-profile-columns"><section><h2>Threads</h2>' + postsHtml + '</section><section><h2>Comments</h2>' + commentsHtml + '</section></div></section>';
  }

  function showNotice(message, err) {
    console.warn(message, err);
    var root = $("forumRoot");
    if (!root) return;
    var old = root.querySelector(".reddit-notice");
    if (old) old.remove();
    var div = document.createElement("div");
    div.className = "reddit-notice";
    div.textContent = message;
    root.prepend(div);
  }
  function route() {
    var r = parseRoute();
    var seq = ++routeSeq;
    setShell(r);
    if (r.type === "jobs") return;
    waitForFortune().then(function () {
      if (seq !== routeSeq) return;
      if (r.type === "hub") renderHub();
      else if (r.type === "company" && r.slug) renderCompanyPage(r.slug, seq);
      else if (r.type === "user" && r.uid) renderUserProfile(r.uid, seq);
    });
  }
  function bindNav() {
    var forums = $("navForums"), jobs = $("navJobs");
    if (forums && !forums.__hcForumBound) {
      forums.__hcForumBound = true;
      forums.addEventListener("click", function (e) { e.preventDefault(); location.hash = "forums"; route(); });
    }
    if (jobs && !jobs.__hcForumBound) {
      jobs.__hcForumBound = true;
      jobs.addEventListener("click", function (e) { e.preventDefault(); location.hash = ""; route(); });
    }
    var input = $("searchInput");
    if (input && !input.__hcForumSearchBound) {
      input.__hcForumSearchBound = true;
      input.addEventListener("keydown", function (e) {
        if (e.key !== "Enter" || !location.hash.includes("forums")) return;
        var hit = searchCompanies(input.value.trim(), 1)[0];
        if (hit) {
          e.preventDefault();
          location.hash = "company/" + encodeURIComponent(hit.slug);
        }
      });
    }
  }
  function init() {
    bindNav();
    window.addEventListener("hashchange", route);
    window.addEventListener("hiringcafe:authchange", route);
    window.addEventListener("hiringcafe:showforums", route);
    window.addEventListener("hiringcafe:firebase", route);
    waitForFortune().then(route);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
