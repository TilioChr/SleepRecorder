// App.jsx
import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
} from "react";
import {
  fetchRecordings,
  fmtBytes,
  setTag,
  renameRecording,
  deleteRecording,
} from "./api.js";
import WavePlayer from "./WavePlayer.jsx";

const TAGS = ["Tous", "Non tagué", "Parole", "Ronflement", "Bruit"];

function safeNameHint(name) {
  return (name || "").replace(/\s+/g, "_");
}

function groupByDate(files) {
  const map = new Map();
  for (const f of files) {
    const d = f.date || "Inconnu";
    if (!map.has(d)) map.set(d, []);
    map.get(d).push(f);
  }
  return Array.from(map.entries()).map(([date, items]) => ({ date, items }));
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [err, setErr] = useState("");

  const [tagFilter, setTagFilter] = useState("Tous");
  const [selectedName, setSelectedName] = useState(null);

  const [renameValue, setRenameValue] = useState("");
  const [busy, setBusy] = useState(false);

  // refs pour éviter les "stale closures" du setInterval
  const desiredSelectionRef = useRef(null); // dernier choix utilisateur à préserver
  const selectedNameRef = useRef(null); // miroir de selectedName

  useEffect(() => {
    selectedNameRef.current = selectedName;
  }, [selectedName]);

  const selected = useMemo(() => {
    if (!selectedName) return null;
    return files.find((f) => f.name === selectedName) || null;
  }, [files, selectedName]);

  const visible = useMemo(() => {
    const sorted = [...files].sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
    if (tagFilter === "Tous") return sorted;
    return sorted.filter((f) => (f.tag || "Non tagué") === tagFilter);
  }, [files, tagFilter]);

  const grouped = useMemo(() => groupByDate(visible), [visible]);

  const refresh = useCallback(async () => {
    try {
      const data = await fetchRecordings();
      setFiles(data);
      setErr("");

      const want = desiredSelectionRef.current || selectedNameRef.current;

      // 1) si on "veut" quelque chose et que ça existe : on garde
      if (want && data.some((x) => x.name === want)) {
        setSelectedName(want);
        // on ne vide desiredSelectionRef que si on l’a effectivement appliqué
        if (desiredSelectionRef.current === want)
          desiredSelectionRef.current = null;
        return;
      }

      // 2) si la sélection actuelle existe encore : ne rien changer
      const cur = selectedNameRef.current;
      if (cur && data.some((x) => x.name === cur)) return;

      // 3) sinon fallback sur le premier
      if (data[0]) setSelectedName(data[0].name);
      else setSelectedName(null);
    } catch (e) {
      setErr(e?.message || String(e));
    }
  }, []);

  // poll léger (utilise refresh stable + refs => pas de rollback)
  useEffect(() => {
    refresh();
    const t = setInterval(() => {
      refresh();
    }, 2500);
    return () => clearInterval(t);
  }, [refresh]);

  // sync rename input quand la sélection change
  useEffect(() => {
    setRenameValue(selected?.name ? safeNameHint(selected.name) : "");
  }, [selected?.name]);

  function pick(f) {
    desiredSelectionRef.current = f.name;
    setSelectedName(f.name);
  }

  async function onSetTag(tag) {
    if (!selected || busy) return;
    try {
      setBusy(true);
      await setTag(selected.name, tag);
      desiredSelectionRef.current = selected.name;
      await refresh();
    } catch (e) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onRename() {
    if (!selected || busy) return;

    const raw = (renameValue || "").trim();
    if (!raw) return;

    const finalName = raw.toLowerCase().endsWith(".wav") ? raw : `${raw}.wav`;
    if (finalName === selected.name) return;

    try {
      setBusy(true);
      const r = await renameRecording(selected.name, finalName);
      desiredSelectionRef.current = r?.name || finalName;
      await refresh();
    } catch (e) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onDelete() {
    if (!selected || busy) return;

    const ok = window.confirm(`Supprimer définitivement "${selected.name}" ?`);
    if (!ok) return;

    try {
      setBusy(true);
      await deleteRecording(selected.name);
      desiredSelectionRef.current = null;
      await refresh();
    } catch (e) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  const selectedTag = selected?.tag || "Non tagué";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo" aria-hidden="true">
            🌙
          </div>
          <div className="titles">
            <h1>Sleep Recorder</h1>
            <p>LAN only • clips détectés</p>
          </div>
        </div>

        <button
          className="iconbtn"
          type="button"
          aria-label="Réglages"
          disabled
        >
          ⚙️
        </button>
      </header>

      <section className="controls card">
        <div className="chips" aria-label="Filtres par tag">
          {TAGS.map((t) => (
            <button
              key={t}
              className={`chip ${tagFilter === t ? "is-active" : ""}`}
              onClick={() => setTagFilter(t)}
              type="button"
            >
              {t}
            </button>
          ))}
        </div>
      </section>

      <main className="main">
        {/* LISTE */}
        <section className="card card--panel" aria-label="Liste des clips">
          <div className="sectionhead">
            <h2>Clips</h2>
            <div className="meta">
              {visible.length} élément{visible.length > 1 ? "s" : ""}
            </div>
          </div>

          {err ? <div className="alert">Erreur: {err}</div> : null}

          <div className="list">
            {grouped.map(({ date, items }) => (
              <div key={date}>
                <div className="daysep">{date}</div>

                {items.map((f) => (
                  <button
                    key={f.name}
                    className={`item ${
                      selected?.name === f.name ? "is-selected" : ""
                    }`}
                    type="button"
                    onClick={() => pick(f)}
                  >
                    <div className="item-main">
                      <div className="item-title">
                        {f.time ? `${f.time} • ` : ""}
                        {f.name}
                      </div>
                      <div className="item-sub">{fmtBytes(f.size)}</div>
                    </div>

                    <div className={`badge ${f.tag ? "" : "badge--muted"}`}>
                      {f.tag || "Non tagué"}
                    </div>
                  </button>
                ))}
              </div>
            ))}

            {visible.length === 0 ? (
              <div className="empty">Aucun clip pour ce filtre.</div>
            ) : null}
          </div>
        </section>

        {/* LECTURE */}
        <section className="card card--panel" aria-label="Lecture et actions">
          <div className="sectionhead">
            <h2>Lecture</h2>
            <div className="meta">{selected ? "1 sélection" : "—"}</div>
          </div>

          <div className="player">
            <div className="now">
              <div className="now-title">
                {selected?.date && selected?.time
                  ? `${selected.date} • ${selected.time}`
                  : "Date inconnue"}
              </div>
              <div className="now-name">{selected?.name ?? "—"}</div>
            </div>

            <div className="waveblock">
              <WavePlayer url={selected?.url || ""} />
            </div>

            <hr className="sep" />

            <div className="field">
              <label className="label">Tag</label>
              <div className="chips" aria-label="Changer le tag">
                {TAGS.filter((t) => t !== "Tous").map((t) => (
                  <button
                    key={t}
                    className={`chip ${selectedTag === t ? "is-active" : ""}`}
                    type="button"
                    onClick={() => onSetTag(t)}
                    disabled={!selected || busy}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            <div className="field">
              <label className="label" htmlFor="rename">
                Renommer
              </label>
              <div className="row-inline">
                <input
                  id="rename"
                  className="control"
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  placeholder={selected ? selected.name : "—"}
                  disabled={!selected || busy}
                />
                <button
                  className="btn btn--primary"
                  type="button"
                  onClick={onRename}
                  disabled={!selected || busy}
                >
                  Appliquer
                </button>
              </div>
            </div>

            <div className="danger">
              <button
                className="btn btn--danger btn--full"
                type="button"
                onClick={onDelete}
                disabled={!selected || busy}
              >
                Supprimer ce clip
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
