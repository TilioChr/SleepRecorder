import React, { useEffect, useMemo, useState } from "react";
import { fetchRecordings, fmtBytes, isTail } from "./api.js";

export default function App() {
  const [files, setFiles] = useState([]);
  const [err, setErr] = useState("");
  const [q, setQ] = useState("");
  const [chip, setChip] = useState("all"); // all | clips | tail
  const [selected, setSelected] = useState(null);

  async function refresh() {
    try {
      const data = await fetchRecordings();
      setFiles(data);
      setErr("");
      if (!selected && data[0]) setSelected(data[0]);
    } catch (e) {
      setErr(e?.message || String(e));
    }
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const visible = useMemo(() => {
    const qq = q.trim().toLowerCase();
    return files.filter((f) => {
      const name = (f.name || "").toLowerCase();
      if (qq && !name.includes(qq)) return false;
      if (chip === "tail") return isTail(name);
      if (chip === "clips") return !isTail(name);
      return true;
    });
  }, [files, q, chip]);

  function pick(f) {
    setSelected(f);
    // autoplay soft
    const a = document.getElementById("audio");
    if (a) {
      a.load();
      a.play().catch(() => {});
    }
  }

  function seek(delta) {
    const a = document.getElementById("audio");
    if (!a) return;
    a.currentTime = Math.max(0, (a.currentTime || 0) + delta);
  }

  function togglePlay() {
    const a = document.getElementById("audio");
    if (!a) return;
    if (a.paused) a.play();
    else a.pause();
  }

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
        <button className="iconbtn" type="button" aria-label="Réglages">
          ⚙️
        </button>
      </header>

      <section className="controls card">
        <div className="row">
          <label className="label" htmlFor="search">
            Filtre
          </label>
          <input
            id="search"
            className="control"
            type="search"
            placeholder="ronflement, parole, bruit…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>

        <div className="chips" aria-label="Filtres rapides">
          <button
            className={`chip ${chip === "all" ? "is-active" : ""}`}
            onClick={() => setChip("all")}
            type="button"
          >
            Tous
          </button>
          <button
            className={`chip ${chip === "clips" ? "is-active" : ""}`}
            onClick={() => setChip("clips")}
            type="button"
          >
            Clips
          </button>
          <button
            className={`chip ${chip === "tail" ? "is-active" : ""}`}
            onClick={() => setChip("tail")}
            type="button"
          >
            Tail
          </button>
        </div>
      </section>

      <main className="main">
        <section className="card">
          <div className="sectionhead">
            <h2>Clips</h2>
            <div className="meta">{visible.length} éléments</div>
          </div>

          {err ? <div className="meta">Erreur: {err}</div> : null}

          <div className="list">
            {visible.map((f) => (
              <button
                key={f.name}
                className={`item ${
                  selected?.name === f.name ? "is-selected" : ""
                }`}
                type="button"
                onClick={() => pick(f)}
              >
                <div className="item-main">
                  <div className="item-title">{f.name}</div>
                  <div className="item-sub">{fmtBytes(f.size)}</div>
                </div>
                <div
                  className={`badge ${isTail(f.name) ? "badge--muted" : ""}`}
                >
                  {isTail(f.name) ? "Tail" : "Clip"}
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="card">
          <div className="sectionhead">
            <h2>Lecture</h2>
            <div className="meta">
              Sélection : <b>{selected?.name ?? "—"}</b>
            </div>
          </div>

          <div className="player">
            <div className="now">
              <div className="now-title">Clip sélectionné</div>
              <div className="now-name">{selected?.name ?? "—"}</div>
            </div>

            <audio
              id="audio"
              className="audio"
              controls
              preload="metadata"
              src={selected?.url ? selected.url : ""}
            />

            <div className="player-actions">
              <button className="btn" type="button" onClick={() => seek(-5)}>
                ⏪ 5s
              </button>
              <button
                className="btn btn--primary"
                type="button"
                onClick={togglePlay}
              >
                ▶︎ / ⏸
              </button>
              <button className="btn" type="button" onClick={() => seek(5)}>
                5s ⏩
              </button>
            </div>
          </div>
        </section>
      </main>

      <nav className="bottomnav" aria-label="Navigation">
        <button className="navbtn is-active" type="button">
          <span className="navico">📄</span>
          <span className="navlbl">Clips</span>
        </button>
        <button className="navbtn" type="button">
          <span className="navico">🗂️</span>
          <span className="navlbl">Archives</span>
        </button>
        <button className="navbtn" type="button">
          <span className="navico">🔎</span>
          <span className="navlbl">Recherche</span>
        </button>
      </nav>
    </div>
  );
}
