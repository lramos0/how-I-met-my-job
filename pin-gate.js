(() => {
  const VERIFY_ENDPOINT = "/.netlify/functions/verify-pin";
  const SESSION_KEY = "hc_pin_verified_v1";

  let accessPromise = null;

  window.HiringCafePinGate = {
    requireAccess() {
      if (!accessPromise) accessPromise = runGate();
      return accessPromise;
    },
    reset() {
      sessionStorage.removeItem(SESSION_KEY);
      location.reload();
    }
  };

  async function runGate() {
    if (sessionStorage.getItem(SESSION_KEY) === "true") {
      unlock();
      return true;
    }

    const alreadyVerified = await verifyPin("");
    if (alreadyVerified.ok) {
      sessionStorage.setItem(SESSION_KEY, "true");
      unlock();
      return true;
    }

    return showPinPrompt();
  }

  function unlock() {
    document.getElementById("pinGateOverlay")?.remove();
    document.documentElement.classList.remove("pin-gate-pending");
    document.documentElement.classList.add("pin-gate-unlocked");
  }

  function showPinPrompt() {
    return new Promise((resolve) => {
      const overlay = document.createElement("div");
      overlay.id = "pinGateOverlay";
      overlay.innerHTML = `
        <style>
          #pinGateOverlay{position:fixed;inset:0;z-index:2147483647;display:grid;place-items:center;background:#fff7e8;color:#2b2016;font-family:"IBM Plex Sans",system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:24px;}
          #pinGateOverlay *{box-sizing:border-box}
          .pin-card{width:min(420px,100%);background:#fff;border:1px solid rgba(43,32,22,.14);box-shadow:0 24px 70px rgba(43,32,22,.16);border-radius:28px;padding:28px;}
          .pin-eyebrow{margin:0 0 8px;font-size:13px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#8a5a22;}
          .pin-title{margin:0 0 8px;font-size:28px;line-height:1.1;font-weight:800;}
          .pin-copy{margin:0 0 22px;color:#6d5b4a;line-height:1.5;}
          .pin-form{display:grid;gap:12px;}
          .pin-label{font-weight:700;font-size:14px;}
          .pin-input{width:100%;font:inherit;font-size:20px;letter-spacing:.12em;border:1px solid rgba(43,32,22,.18);border-radius:16px;padding:14px 16px;outline:none;background:#fffaf2;}
          .pin-input:focus{border-color:#ca7b17;box-shadow:0 0 0 4px rgba(202,123,23,.14);}
          .pin-button{border:0;border-radius:16px;background:#2f7d32;color:#fff;font:inherit;font-weight:800;padding:14px 16px;cursor:pointer;}
          .pin-button[disabled]{opacity:.65;cursor:wait;}
          .pin-error{min-height:20px;margin:0;color:#a42b15;font-weight:700;font-size:14px;}
        </style>
        <section class="pin-card" role="dialog" aria-modal="true" aria-labelledby="pinGateTitle">
          <p class="pin-eyebrow">Private preview</p>
          <h1 class="pin-title" id="pinGateTitle">Enter your PIN</h1>
          <p class="pin-copy">This site is locked until you enter the access PIN.</p>
          <form class="pin-form" id="pinGateForm">
            <label class="pin-label" for="pinGateInput">PIN code</label>
            <input class="pin-input" id="pinGateInput" name="pin" type="password" inputmode="numeric" autocomplete="one-time-code" required autofocus />
            <button class="pin-button" id="pinGateButton" type="submit">Unlock site</button>
            <p class="pin-error" id="pinGateError" aria-live="polite"></p>
          </form>
        </section>`;

      document.body.appendChild(overlay);
      const form = overlay.querySelector("#pinGateForm");
      const input = overlay.querySelector("#pinGateInput");
      const button = overlay.querySelector("#pinGateButton");
      const error = overlay.querySelector("#pinGateError");
      input.focus();

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        error.textContent = "";
        button.disabled = true;
        button.textContent = "Checking…";
        const result = await verifyPin(input.value.trim());
        if (result.ok) {
          sessionStorage.setItem(SESSION_KEY, "true");
          unlock();
          resolve(true);
          return;
        }
        error.textContent = result.message || "Incorrect PIN. Try again.";
        input.value = "";
        input.focus();
        button.disabled = false;
        button.textContent = "Unlock site";
      });
    });
  }

  async function verifyPin(pin) {
    try {
      const response = await fetch(VERIFY_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ pin })
      });
      const data = await response.json().catch(() => ({}));
      return response.ok && data.ok ? { ok: true } : { ok: false, message: data.error };
    } catch (error) {
      return { ok: false, message: "Could not verify the PIN. Check your connection and try again." };
    }
  }
})();
