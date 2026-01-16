import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

export default function WavePlayer({ url }) {
  const containerRef = useRef(null);
  const wsRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [cur, setCur] = useState(0);
  const [dur, setDur] = useState(0);

  useEffect(() => {
    if (!containerRef.current || !url) return;

    // cleanup ancien
    if (wsRef.current) {
      wsRef.current.destroy();
      wsRef.current = null;
    }

    const ws = WaveSurfer.create({
      container: containerRef.current,
      height: 72,
      normalize: true,
      cursorWidth: 2,
      barWidth: 2,
      barGap: 1,
      // couleurs par défaut (tu peux les styler plus tard)
      waveColor: "rgba(147,164,203,0.55)",
      progressColor: "rgba(106,163,255,0.85)",
      cursorColor: "rgba(231,239,255,0.85)",
    });

    wsRef.current = ws;
    setReady(false);
    setPlaying(false);
    setCur(0);
    setDur(0);

    ws.load(url);

    ws.on("ready", () => {
      setReady(true);
      setDur(ws.getDuration() || 0);
    });
    ws.on("audioprocess", () => setCur(ws.getCurrentTime() || 0));
    ws.on("timeupdate", () => setCur(ws.getCurrentTime() || 0));
    ws.on("finish", () => setPlaying(false));

    return () => {
      ws.destroy();
      wsRef.current = null;
    };
  }, [url]);

  function fmt(t) {
    const s = Math.max(0, Math.floor(t || 0));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}:${String(r).padStart(2, "0")}`;
  }

  function toggle() {
    const ws = wsRef.current;
    if (!ws || !ready) return;
    ws.playPause();
    setPlaying(ws.isPlaying());
  }

  function seek(delta) {
    const ws = wsRef.current;
    if (!ws || !ready) return;
    const t = Math.max(0, Math.min(dur, (ws.getCurrentTime() || 0) + delta));
    ws.setTime(t);
  }

  return (
    <div className="wavewrap">
      <div className="wave" ref={containerRef} aria-label="Forme d'onde" />
      <div className="time">
        <span className="tcur">{fmt(cur)}</span>
        <span className="tsep">/</span>
        <span className="tdur">{fmt(dur)}</span>
      </div>

      <div className="player-actions">
        <button className="btn" type="button" onClick={() => seek(-5)}>
          ⏪ 5s
        </button>
        <button className="btn btn--primary" type="button" onClick={toggle}>
          {playing ? "⏸" : "▶︎"}
        </button>
        <button className="btn" type="button" onClick={() => seek(5)}>
          5s ⏩
        </button>
      </div>
    </div>
  );
}
