(() => {
  const FORUM_KEY = "hc_forum_state_v1";
  const rootReady = () => document.getElementById("forumRoot");
  const defaultState = () => ({ posts: {}, upvotes: {}, postVotes: {} });

  document.addEventListener("DOMContentLoaded", initForum);
  window.addEventListener("hiringcafe:authchange", renderForum);
  window.addEventListener("hiringcafe:forumstate", renderForum);

  function initForum(){
    injectForumStyles();
    renderForum();
  }

  function getAuth(){ return window.HiringCafeAuth || {}; }
  function getUser(){ return getAuth().getUser?.() || null; }
  function getProfile(){ return getAuth().getProfile?.() || null; }
  function localState(){
    try { return { ...defaultState(), ...(JSON.parse(localStorage.getItem(FORUM_KEY) || "{}")) }; }
    catch { return defaultState(); }
  }
  function state(){
    const authState = getAuth().getState?.();
    return { ...defaultState(), ...(authState || localState()) };
  }
  async function saveForumPatch(patch){
    const auth = getAuth();
    if (auth.saveForumState) return auth.saveForumState(patch);
    const next = { ...state(), ...patch };
    localStorage.setItem(FORUM_KEY, JSON.stringify(next));
    renderForum();
    return next;
  }

  function renderForum(){
    const root = rootReady();
    if (!root) return;
    const user = getUser();
    const profile = getProfile();
    const s = state();
    const posts = Object.values(s.posts || {}).sort((a,b) => (b.createdAt || 0) - (a.createdAt || 0));
    root.innerHTML = `
      <section class="forum-shell">
        <div class="forum-hero">
          <div>
            <p class="forum-kicker">Company forums</p>
            <h2>Share application stories</h2>
            <p>Posts and upvotes are tied to your local/Firebase profile so the account button actually initializes useful user data.</p>
          </div>
          <div class="forum-user-chip">${user ? esc(profile?.displayName || user.name || "Member") : "Guest"}</div>
        </div>
        <form class="forum-composer" id="forumComposer">
          <label>Company <input id="forumCompany" placeholder="Apple, Nike, Cisco…" autocomplete="off"></label>
          <label>Title <input id="forumTitle" placeholder="Interview timeline, OA notes, offer details…"></label>
          <label>Post <textarea id="forumBody" placeholder="What happened? What would help the next applicant?"></textarea></label>
          <button class="primary-btn" type="submit">Post</button>
        </form>
        <div class="forum-list" id="forumPostList">
          ${posts.length ? posts.map(postHtml).join("") : `<article class="forum-empty">No posts yet. Create an account and start the first thread.</article>`}
        </div>
      </section>
    `;
    root.querySelector("#forumComposer")?.addEventListener("submit", createPost);
    root.querySelectorAll("[data-upvote]").forEach(btn => btn.addEventListener("click", () => toggleUpvote(btn.dataset.upvote)));
  }

  async function createPost(event){
    event.preventDefault();
    const user = getUser();
    if (!user) return getAuth().promptLogin?.("Create an account to post and track your forum history.");
    const profile = getProfile();
    const company = document.getElementById("forumCompany")?.value.trim();
    const title = document.getElementById("forumTitle")?.value.trim();
    const body = document.getElementById("forumBody")?.value.trim();
    if (!company || !title || !body) return;
    const s = state();
    const id = `post-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,8)}`;
    const post = {
      id, company, title, body,
      authorUid: user.uid,
      authorName: profile?.displayName || user.name || "Member",
      createdAt: Date.now(),
      upvotes: 0
    };
    await saveForumPatch({ posts: { ...(s.posts || {}), [id]: post } });
  }

  async function toggleUpvote(postId){
    const user = getUser();
    if (!user) return getAuth().promptLogin?.("Create an account to upvote posts and keep your vote history.");
    const s = state();
    const posts = { ...(s.posts || {}) };
    if (!posts[postId]) return;
    const voteKey = `${user.uid}:${postId}`;
    const upvotes = { ...(s.upvotes || {}) };
    const postVotes = { ...(s.postVotes || {}) };
    if (upvotes[voteKey]) {
      delete upvotes[voteKey];
      postVotes[postId] = Math.max(0, (postVotes[postId] || 1) - 1);
    } else {
      upvotes[voteKey] = Date.now();
      postVotes[postId] = (postVotes[postId] || 0) + 1;
    }
    posts[postId] = { ...posts[postId], upvotes: postVotes[postId] || 0 };
    await saveForumPatch({ posts, upvotes, postVotes });
  }

  function postHtml(post){
    const s = state();
    const user = getUser();
    const voteKey = user ? `${user.uid}:${post.id}` : "";
    const voted = !!(voteKey && s.upvotes?.[voteKey]);
    return `<article class="forum-post">
      <div class="forum-post-head"><div><strong>${esc(post.title)}</strong><span>${esc(post.company)} · ${esc(post.authorName || "Member")} · ${timeAgo(post.createdAt)}</span></div><button class="upvote ${voted ? 'is-voted' : ''}" data-upvote="${escAttr(post.id)}">▲ ${Number(post.upvotes || s.postVotes?.[post.id] || 0)}</button></div>
      <p>${esc(post.body)}</p>
    </article>`;
  }

  function injectForumStyles(){
    if (document.getElementById("forumRuntimeStyles")) return;
    const style = document.createElement("style");
    style.id = "forumRuntimeStyles";
    style.textContent = `
      .profile-card,.forum-shell{border:1px solid rgba(124,45,18,.14);border-radius:24px;background:#fffaf4;padding:18px;margin-top:14px;box-shadow:0 16px 40px rgba(124,45,18,.08)}
      .profile-card label,.forum-composer label{display:grid;gap:6px;font-weight:700;color:#4a2511}.profile-card input,.forum-composer input,.forum-composer textarea{border:1px solid #ead7c3;border-radius:14px;padding:10px 12px;font:inherit;background:white}.profile-stats{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.profile-stats span,.forum-user-chip{background:#fff;border:1px solid #f0dcc8;border-radius:999px;padding:7px 10px;font-weight:800}.forum-hero{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:16px}.forum-kicker{font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:#9a3412}.forum-composer{display:grid;gap:12px}.forum-composer textarea{min-height:110px}.forum-list{display:grid;gap:12px;margin-top:18px}.forum-post,.forum-empty{background:white;border:1px solid #f0dcc8;border-radius:18px;padding:14px}.forum-post-head{display:flex;justify-content:space-between;gap:14px}.forum-post-head span{display:block;color:#7c5b45;font-size:.92rem;margin-top:4px}.upvote{border:1px solid #f0dcc8;border-radius:999px;background:#fff7ed;padding:8px 12px;font-weight:900;cursor:pointer}.upvote.is-voted{background:#ffedd5;color:#9a3412}.primary-btn{cursor:pointer}
    `;
    document.head.append(style);
  }

  function esc(s){ return String(s||"").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  function escAttr(s){ return esc(s).replace(/`/g,"&#96;"); }
  function timeAgo(t){ const h=Math.max(1, Math.round((Date.now()-Number(t||Date.now()))/36e5)); if(h<24) return `${h}h ago`; const d=Math.round(h/24); return `${d}d ago`; }
})();
