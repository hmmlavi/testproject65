/* GOJO — Phase 2 front-end logic.
   Talks to Python ONLY through window.pywebview.api (see gojo/ui/api.py).
   No state is kept on the JS side that Python doesn't own. */
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
};

let api = null;
let busy = false;
let transcriptOpen = false;
let lastResponse = false; // has GOJO replied in this session?

const IDLE_TEXT = {
  sleeping: "· · · sleeping · · ·",
  active: "Awake. Kya chahiye, boss?",
  thinking: "",
};

/* ------------------------------------------------------------------ */
/* bootstrap: wait for the pywebview bridge, then init                 */
/* ------------------------------------------------------------------ */

function boot(tries = 60) {
  if (window.pywebview && window.pywebview.api) {
    api = window.pywebview.api;
    init();
    return;
  }
  if (tries <= 0) {
    els.statusText.textContent = "bridge offline";
    return;
  }
  setTimeout(() => boot(tries - 1), 100);
}

async function init() {
  await applyStatus(await api.get_status());
  await loadTranscript();
  await openSettingsData();
  setInterval(() => refreshStatus(), 900); // status polling (push comes in Phase 3 with voice)
  els.input.focus();
}

/* ------------------------------------------------------------------ */
/* status                                                              */
/* ------------------------------------------------------------------ */

async function refreshStatus() {
  try {
    await applyStatus(await api.get_status());
  } catch (err) {
    console.warn("status poll failed", err);
  }
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

  api.chat(text)
    .then((r) => {
      if (!r.ok) {
        toast(r.error, true);
        return;
      }
      appendEntry("you", text);
      appendEntry("gojo", r.reply);
      showReply(r.reply);
    })
    .catch((err) => toast("Bridge error: " + err, true))
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
  const r = await api.get_transcript();
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
  const r = await api.new_chat();
  if (!r.ok) { toast(r.error, true); return; }
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
  try {
    const s = await api.get_settings();
    if (!s.ok) return;
    els.setBrain.textContent = `${s.provider} · ${s.model}`;
    els.setData.textContent = s.data_dir;
    els.setData.title = s.data_dir;
    els.setVersion.textContent = s.version;
  } catch (err) {
    console.warn("settings load failed", err);
  }
}

function openSettings() {
  openSettingsData();
  refreshStatus().then(() => {
    els.setState.textContent = els.body.dataset.state;
  });
  els.modal.classList.add("open");
  els.modal.setAttribute("aria-hidden", "false");
}

function closeSettings() {
  els.modal.classList.remove("open");
  els.modal.setAttribute("aria-hidden", "true");
}

async function onClearData() {
  if (!window.confirm("Delete ALL saved conversations? This cannot be undone.")) return;
  const r = await api.clear_data();
  if (!r.ok) { toast(r.error, true); return; }
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
  const call = sleeping ? api.wake() : api.sleep();
  call.then(refreshStatus).catch((err) => toast(String(err), true));
  if (!sleeping) toast("GOJO is going to sleep…");
}

function onQuit() {
  api.close_app().catch(() => {});
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
