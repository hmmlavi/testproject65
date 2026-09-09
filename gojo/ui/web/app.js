/* GOJO — Phase 2 front-end logic.
   Talks to Python ONLY through window.pywebview.api (see gojo/ui/api.py).
   No state is kept on the JS side that Python doesn't own.

   BRIDGE RULE: every Python call goes through callApi(), so a broken
   bridge can never fail silently — it toasts an error and, if the bridge
   never comes up at all, the page shows a full-screen fatal message. */
"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  body: document.body,
  statusText: $("statusText"),
  heroIdle: $("heroIdle"),
  heroThinking: $("heroThinking"),
  heroText: $("heroText"),
  input: $("input"),
  btnSend: $("btnSend"),
  btnPower: $("btnPower"),
  btnNew: $("btnNew"),
  btnTranscript: $("btnTranscript"),
  btnSettings: $("btnSettings"),
  btnQuit: $("btnQuit"),
  trPanel: $("transcriptPanel"),
  trEntries: $("trEntries"),
  trEmpty: $("trEmpty"),
  trCount: $("trCount"),
  modal: $("settingsModal"),
  setBrain: $("setBrain"),
  setState: $("setState"),
  setData: $("setData"),
  setVersion: $("setVersion"),
  btnModalClose: $("btnModalClose"),
  btnModalDone: $("btnModalDone"),
  btnClearData: $("btnClearData"),
  toast: $("toast"),
  fatal: $("bridgeFatal"),
  fatalMsg: $("bridgeFatalMsg"),
};

let api = null;
let booted = false;
let busy = false;
let transcriptOpen = false;
let lastResponse = false; // has GOJO replied in this session?

const IDLE_TEXT = {
  sleeping: "· · · sleeping · · ·",
  active: "Awake. Kya chahiye, boss?",
  thinking: "",
};

/* ------------------------------------------------------------------ */
/* bridge helper — errors are NEVER silent                             */
/* ------------------------------------------------------------------ */

function callApi(name, ...args) {
  if (!api) return Promise.reject(new Error("bridge not ready"));
  return api[name](...args).catch((err) => {
    toast(`Bridge error (${name}): ${err}`, true);
    throw err;
  });
}

/* ------------------------------------------------------------------ */
/* bootstrap: wait for the pywebview bridge, then init                 */
/* ------------------------------------------------------------------ */

function boot(tries = 60) {
  if (booted) return;
  if (window.pywebview && window.pywebview.api) {
    booted = true;
    api = window.pywebview.api;
    init();
    return;
  }
  if (tries <= 0) {
    showFatal(
      "The interface couldn't reach GOJO's engine (bridge offline). " +
      "Close the app and run  python -m gojo.app  again from the terminal — " +
      "if it still happens, check the terminal output for errors."
    );
    return;
  }
  setTimeout(() => boot(tries - 1), 100);
}

/* Fast path: pywebview 6.x fires 'pywebviewready' once the API is
   injected. The polling loop above stays as the fallback. */
window.addEventListener("pywebviewready", () => boot(2));

function showFatal(msg) {
  els.fatalMsg.textContent = msg;
  els.fatal.hidden = false;
}

async function init() {
  try {
    const s = await callApi("get_status");
    await applyStatus(s.status);
  } catch (err) {
    console.warn("init: get_status failed", err);
  }
  try {
    await loadTranscript();
    await openSettingsData();
  } catch (err) {
    console.warn("init: transcript/settings failed", err);
  }
  setInterval(() => refreshStatus(), 900); // status polling (push comes in Phase 3 with voice)
  els.input.focus();
}

/* ------------------------------------------------------------------ */
/* status                                                              */
/* ------------------------------------------------------------------ */

function refreshStatus() {
  callApi("get_status")
    .then((s) => applyStatus(s.status))
    .catch(() => {}); // transport errors already toasted by callApi
}

async function applyStatus(status) {
  if (!status) return;
  els.body.dataset.state = status.state;
  els.statusText.textContent = status.state;

  const sleeping = status.state === "sleeping";
  els.input.disabled = sleeping || busy;
  els.btnSend.disabled = sleeping || busy;

  // hero placeholder only when GOJO hasn't replied yet
  if (!lastResponse) {
    els.heroIdle.textContent = IDLE_TEXT[status.state] || "";
  }
}

/* ------------------------------------------------------------------ */
/* chat                                                                */
/* ------------------------------------------------------------------ */

function onSend() {
  if (busy) return;
  const text = els.input.value.trim();
  if (!text) return;
  if (els.body.dataset.state === "sleeping") {
    toast("GOJO is sleeping — press power first");
    return;
  }
  els.input.value = "";
  busy = true;
  showThinking(true);
  applyStatus({ state: "thinking" }); // optimistic; polling confirms

  callApi("chat", text)
    .then((r) => {
      if (!r.ok) {
        toast(r.error, true);
        return;
      }
      appendEntry("you", text);
      appendEntry("gojo", r.reply);
      showReply(r.reply);
    })
    .catch(() => {}) // transport errors already toasted by callApi
    .finally(() => {
      busy = false;
      showThinking(false);
      refreshStatus();
    });
}

function showThinking(on) {
  els.heroThinking.hidden = !on;
  els.heroIdle.hidden = on;
  if (on) els.heroText.hidden = true;
}

function showReply(text) {
  els.heroThinking.hidden = true;
  els.heroIdle.hidden = true;
  els.heroText.hidden = false;
  els.heroText.textContent = text;
  lastResponse = true;
}

/* ------------------------------------------------------------------ */
/* transcript                                                          */
/* ------------------------------------------------------------------ */

function appendEntry(who, text) {
  if (els.trEmpty) els.trEmpty.style.display = "none";
  const entry = document.createElement("div");
  entry.className = "entry " + (who === "gojo" ? "gojo" : "user");

  const label = document.createElement("div");
  label.className = "who";
  label.textContent = who === "gojo" ? "Gojo" : "You";

  const body = document.createElement("div");
  body.className = "txt";
  body.textContent = text; // textContent — no HTML injection, ever

  entry.appendChild(label);
  entry.appendChild(body);
  els.trEntries.appendChild(entry);
  setCount(els.trEntries.querySelectorAll(".entry").length);
  els.trEntries.scrollTop = els.trEntries.scrollHeight;
}

function setCount(n) {
  els.trCount.textContent = String(n);
}

async function loadTranscript() {
  els.trEntries.querySelectorAll(".entry").forEach((e) => e.remove());
  const r = await callApi("get_transcript");
  if (r.ok && r.items.length) {
    els.trEmpty.style.display = "none";
    for (const item of r.items) {
      appendEntry(item.role === "model" ? "gojo" : "you", item.text);
    }
  } else {
    els.trEmpty.style.display = "";
  }
  setCount(els.trEntries.querySelectorAll(".entry").length);
}

async function onNewChat() {
  const r = await callApi("new_chat");
  if (!r.ok) return;
  await loadTranscript();
  lastResponse = false;
  els.heroText.hidden = true;
  els.heroIdle.hidden = false;
  refreshStatus();
  toast("New conversation started");
}

/* ------------------------------------------------------------------ */
/* settings                                                            */
/* ------------------------------------------------------------------ */

async function openSettingsData() {
  const s = await callApi("get_settings");
  if (!s.ok) return;
  els.setBrain.textContent = `${s.provider} · ${s.model}`;
  els.setData.textContent = s.data_dir;
  els.setData.title = s.data_dir;
  els.setVersion.textContent = s.version;
}

function openSettings() {
  openSettingsData().catch(() => {});
  refreshStatus();
  setTimeout(() => {
    els.setState.textContent = els.body.dataset.state;
  }, 50);
  els.modal.classList.add("open");
  els.modal.setAttribute("aria-hidden", "false");
}

function closeSettings() {
  els.modal.classList.remove("open");
  els.modal.setAttribute("aria-hidden", "true");
}

async function onClearData() {
  if (!window.confirm("Delete ALL saved conversations? This cannot be undone.")) return;
  const r = await callApi("clear_data");
  if (!r.ok) return;
  lastResponse = false;
  await loadTranscript();
  els.heroText.hidden = true;
  els.heroIdle.hidden = false;
  toast("All saved data cleared");
}

/* ------------------------------------------------------------------ */
/* lifecycle buttons                                                   */
/* ------------------------------------------------------------------ */

function onPower() {
  const sleeping = els.body.dataset.state === "sleeping";
  // explicit calls (not a ternary) so the wiring test can see both names
  const call = sleeping ? callApi("wake") : callApi("sleep");
  call
    .then(() => refreshStatus())
    .catch(() => {});
  if (!sleeping) toast("GOJO is going to sleep…");
}

function onQuit() {
  callApi("close_app").catch(() => {});
}

function toggleTranscript() {
  transcriptOpen = !transcriptOpen;
  els.body.classList.toggle("tr-open", transcriptOpen);
  els.btnTranscript.classList.toggle("active", transcriptOpen);
  if (transcriptOpen) els.trEntries.scrollTop = els.trEntries.scrollHeight;
}

/* ------------------------------------------------------------------ */
/* toast                                                               */
/* ------------------------------------------------------------------ */

let toastTimer = null;
function toast(msg, isError = false) {
  els.toast.textContent = msg;
  els.toast.classList.toggle("error", isError);
  els.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove("show"), 2600);
}

/* ------------------------------------------------------------------ */

document.getElementById("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  onSend();
});
els.btnPower.addEventListener("click", onPower);
els.btnNew.addEventListener("click", onNewChat);
els.btnTranscript.addEventListener("click", toggleTranscript);
els.btnSettings.addEventListener("click", openSettings);
els.btnQuit.addEventListener("click", onQuit);
els.btnModalClose.addEventListener("click", closeSettings);
els.btnModalDone.addEventListener("click", closeSettings);
els.btnClearData.addEventListener("click", onClearData);
els.modal.addEventListener("click", (e) => {
  if (e.target === els.modal) closeSettings();
});

boot();
