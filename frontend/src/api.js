export async function fetchRecordings() {
    const r = await fetch("/api/recordings", { cache: "no-store" });
    if (!r.ok) throw new Error("API /api/recordings unreachable");
    const data = await r.json();
    if (!Array.isArray(data)) throw new Error("API format unexpected");
    return data;
}

export function fmtBytes(n) {
    if (!Number.isFinite(n)) return "";
    const u = ["B", "KB", "MB", "GB"];
    let i = 0, v = n;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(i === 0 ? 0 : 1)} ${u[i]}`;
}

export function isTail(name) {
    return (name || "").toLowerCase().includes("tail");
}
