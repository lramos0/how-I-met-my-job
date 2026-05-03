const HC_COUNTER_CACHE = "hc-counter-cache-v1";
let counters = { companies: 5600000, jobs: 3100000, updatedAt: Date.now() };

self.addEventListener("install", event => {
  self.skipWaiting();
  event.waitUntil(caches.open(HC_COUNTER_CACHE));
});

self.addEventListener("activate", event => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("message", event => {
  if (event.data && event.data.type === "HC_COUNTERS") {
    counters = { ...counters, ...event.data.counters, updatedAt: Date.now() };
    caches.open(HC_COUNTER_CACHE).then(cache => {
      cache.put("/hc-counter-state.json", new Response(JSON.stringify(counters), { headers: { "Content-Type": "application/json" } }));
    });
  }
});

self.addEventListener("fetch", event => {
  const url = new URL(event.request.url);
  if (url.pathname.endsWith("/hc-counter-state.json")) {
    counters.jobs += 3;
    counters.companies += 1;
    counters.updatedAt = Date.now();
    event.respondWith(new Response(JSON.stringify(counters), { headers: { "Content-Type": "application/json" } }));
  }
});
