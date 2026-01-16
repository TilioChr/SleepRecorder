// api.js
async function requestJson(url, options) {
    const r = await fetch(url, { cache: "no-store", ...options });
    if (!r.ok) {
        const txt = await r.text().catch(() => "");
        const more = txt ? ` — ${txt.slice(0, 200)}` : "";
        throw new Error(`HTTP ${r.status} ${r.statusText}${more}`);
    }
    return r.json();
}

export async function fetchRecordings() {
    const data = await requestJson("/api/recordings");
    if (!Array.isArray(data)) throw new Error("API format unexpected");
    return data;
}

export function fmtBytes(n) {
    if (!Number.isFinite(n)) return "";
    const u = ["B", "KB", "MB", "GB"];
    let i = 0;
    let v = n;
    while (v >= 1024 && i < u.length - 1) {
        v /= 1024;
        i++;
    }
    return `${v.toFixed(i === 0 ? 0 : 1)} ${u[i]}`;
}

export function isTail(name) {
    return (name || "").toLowerCase().includes("tail");
}

export async function renameRecording(name, newName) {
    return requestJson(`/api/recordings/${encodeURIComponent(name)}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_name: newName }),
    });
}

export async function deleteRecording(name) {
    return requestJson(`/api/recordings/${encodeURIComponent(name)}`, {
        method: "DELETE",
    });
}

export async function setTag(name, tag) {
    return requestJson(`/api/recordings/${encodeURIComponent(name)}/tag`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tag }),
    });
}
