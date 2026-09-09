/* GOJO — Phase 3 front-end logic.
   Talks to Python ONLY through window.pywebview.api (see gojo/ui/api.py).
   No state is kept on the JS side that Python doesn't own.

   BRIDGE RULE: every Python call goes through callApi(), so a broken
   bridge can never fail silently — it toasts an error and, if the bridge
   never comes up at all, the page shows a full-screen fatal message.

   Phase 3 additions (voice): the 🎙 Talk button (click = start/stop
   listening), voice status in the pill + avatar, note toasts (one-shot
   messages from the voice pipeline, e.g. "backup voice used"), voice
   settings in the settings modal, and a "· voice" marker on transcript
   lines that came from speech. */
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
  btnMic: $("btnMic"),
  talkHint: $("talkHint"),
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
  setVoiceOn: $("setVoiceOn"),
  setWakeWord: $("setWakeWord"),
  setAutoplay: $("setAutoplay"),
  setProvider: $("setProvider"),
  setFishVoice: $("setFishVoice"),
  setFishStatus: $("setFishStatus"),
  setSttModel: $("setSttModel"),
  btnModalClose: $("btnModalClose"),
  btnModalDone: $("btnModalDone"),
  btnClearData: $("btnClearData"),
  toast: $("toast"),
  fatal: $("bridgeFatal"),
  fatalMsg: $("bridgeFatalMsg"),
};

let api = null;
let booted = false;
let busy = false; // this client is mid-text-turn
let transcriptOpen = false;
let lastResponse = false; // has GOJO replied in this session?
let voiceEnabled = false; // from get_voice_settings
let wasVoiceBusy = false; // previous status.busy (voice turn in flight)

const IDLE_TEXT = {
  sleeping: "· · · sleeping · · ·",
  active: "Awake. Kya chahiye, boss?",
};

const PILL_TEXT = {
  sleeping: "sleeping",
  active: "awake",
  thinking: "thinking",
  listening: "listening…",
  speaking: "speaking…",
  error: "error",
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
    await loadVoiceSettings();
    await loadTranscript();
    await openSettingsData();
  } catch (err) {
    console.warn("init: transcript/settings failed", err);
  }
  setInterval(() => refreshStatus(), 900); // status polling (voice notes ride along)
  els.input.focus();
}

/* ------------------------------------------------------------------ */
/* status — pill, avatar, mic button, one-shot notes                   */
/* ------------------------------------------------------------------ */

function refreshStatus() {
  callApi("get_status")
    .then((s) => applyStatus(s.status))
    .catch(() => {}); // transport errors already toasted by callApi
}

async function applyStatus(status) {
  if (!status) return;
  const state = status.state;
  els.body.dataset.state = state;
  // "sleeping + local wake listener on" shows as one combined status
  els.statusText.textContent =
    state === "sleeping" && status.wake_listening ? "wake listening" : PILL_TEXT[state] || state;

  const sleeping = state === "sleeping";
  const listening = state === "listening";
  const voiceBusy = !!status.busy;

  // one-shot note from the voice pipeline (e.g. fallback warning) —
  // the bridge clears it once delivered, so we only ever show it once
  if (status.note && status.note.text) {
    toast(status.note.text, !!status.note.error);
  }

  // composer + mic live on the same busy flags
  els.input.disabled = sleeping || busy || voiceBusy || !voiceEnabled;
  els.btnSend.disabled = sleeping || busy || voiceBusy || !voiceEnabled;
  els.btnMic.disabled =
    sleeping || !voiceEnabled || (voiceBusy && !listening);
  els.btnMic.classList.toggle("listening", listening);
  els.talkHint.hidden = !listening;

  // hero placeholder only when GOJO hasn't replied yet
  if (!lastResponse && IDLE_TEXT[state]) {
    els.heroIdle.textContent = IDLE_TEXT[state];
  }

  // when a voice turn ends, the transcript has new lines and a new
  // reply (voice turns don't return the reply to the clicker) — refresh
  if (wasVoiceBusy && !voiceBusy) {
    try {
      const items = await loadTranscript();
      const last = items.filter((i) => i.role === "model").pop();
      if (last) showReply(last.text);
    } catch (err) {
      console.warn("post-voice transcript refresh failed", err);
    }
  }
  wasVoiceBusy = voiceBusy;
}

/* ------------------------------------------------------------------ */
/* chat (text path — unchanged from Phase 2)                           */
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
      appendEntry("you", text, "text");
      appendEntry("gojo", r.reply, "ai");
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
/* voice (Phase 3) — press-to-talk via the mic button                  */
/* ------------------------------------------------------------------ */

function onTalk() {
  const state = els.body.dataset.state;
  if (state === "sleeping") {
    toast("GOJO is sleeping — press power first");
    return;
  }
  if (!voiceEnabled) {
    toast("Voice is off — turn it on in Settings");
    return;
  }
  const listening = state === "listening";
  // explicit calls (not a ternary) so the wiring test can see both names
  const call = listening ? callApi("stop_talk") : callApi("start_talk");
  call
    .then((r) => {
      if (!r.ok && r.error) toast(r.error, true);
      refreshStatus();
    })
    .catch(() => {}); // transport errors already toasted by callApi
}

/* ------------------------------------------------------------------ */
/* transcript — lines can carry a `via` marker (text | voice | ai)     */
/* ------------------------------------------------------------------ */

function appendEntry(who, text, via) {
  if (els.trEmpty) els.trEmpty.style.display = "none";
  const entry = document.createElement("div");
  entry.className = "entry " + (who === "gojo" ? "gojo" : "user");

  const label = document.createElement("div");
  label.className = "who";
  label.textContent = who === "gojo" ? "Gojo" : "You";
  if (via === "voice") {
    const tag = document.createElement("span");
    tag.className = "via";
    tag.textContent = "· voice";
    label.appendChild(tag);
  }

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
  let items = [];
  if (r.ok && r.items.length) {
    els.trEmpty.style.display = "none";
    items = r.items;
    for (const item of items) {
      appendEntry(item.role === "model" ? "gojo" : "you", item.text, item.via);
    }
  } else {
    els.trEmpty.style.display = "";
  }
  setCount(els.trEntries.querySelectorAll(".entry").length);
  return items;
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
  loadVoiceSettings().catch(() => {});
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

/* voice settings — stored as runtime prefs (data/gojo_prefs.json),
   separate from .env (which holds keys + model choices) */

async function loadVoiceSettings() {
  const v = await callApi("get_voice_settings");
  if (!v.ok) return;
  voiceEnabled = !!v.voice_enabled;
  els.setVoiceOn.checked = voiceEnabled;
  els.setWakeWord.checked = !!v.always_listening;
  els.setAutoplay.checked = !!v.tts_autoplay;
  els.setProvider.value = v.tts_provider || "auto";
  els.setFishVoice.textContent = v.fish_voice || "—";
  els.setFishVoice.title = v.fish_voice || "";
  els.setFishStatus.textContent = v.fish_configured
    ? "ready"
    : "missing — add FISH_AUDIO_API_KEY in .env";
  els.setSttModel.textContent = v.stt_model || "—";
  refreshStatus();
}

function saveVoicePref(key, value) {
  callApi("set_voice_pref", key, value)
    .then((r) => {
      if (!r.ok && r.error) toast(r.error, true);
      refreshStatus();
    })
    .catch(() => {});
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
els.btnMic.addEventListener("click", onTalk);
els.btnModalClose.addEventListener("click", closeSettings);
els.btnModalDone.addEventListener("click", closeSettings);
els.btnClearData.addEventListener("click", onClearData);
els.setVoiceOn.addEventListener("change", (e) => {
  saveVoicePref("voice_enabled", e.target.checked);
});
els.setWakeWord.addEventListener("change", (e) => {
  saveVoicePref("always_listening", e.target.checked);
});
els.setAutoplay.addEventListener("change", (e) => {
  saveVoicePref("tts_autoplay", e.target.checked);
});
els.setProvider.addEventListener("change", (e) => {
  saveVoicePref("tts_provider", e.target.value);
});
els.modal.addEventListener("click", (e) => {
  if (e.target === els.modal) closeSettings();
});

boot();
