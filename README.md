# Sleep Recorder

Enregistre de l’audio depuis un Raspberry Pi via TCP, découpe en fichiers WAV, traite l'audio et fournit une interface web locale pour consulter et écouter les enregistrements.

---

## Architecture

```
Raspberry Pi (arecord + nc) TCP PCM (s16le / 16kHz / mono)
▼ (via wifi)
Réception TCP Serveur Backend (Python / FastAPI)
▼
Traitement local (découpage wav, analyse VAD, tag, etc...)
▼
nginx - Frontend React localhost
```

---

## Lancer

```powershell
podman machine init
podman machine start
podman compose up -d --build
```

Interface :

```
http://localhost:8080
```

Build du frontend - (obligatoire après chaque modif React)

```powershell
.\deploy.ps1
```

Tester sans Raspberry Pi - Envoyer un WAV en PCM brut vers le backend :

```powershell
ffmpeg -i test.wav -f s16le -ar 16000 -ac 1 - | ncat 127.0.0.1 5000
```

Les fichiers apparaissent automatiquement dans l’UI.

Utiliser avec Raspberry Pi

```bash
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 | nc <IP_BACKEND> 5000
```

---

### Endpoints utiles :

Liste des wav dans le backend : http://localhost:8080/api/recordings

Télécharger un WAV précis : http://localhost:8080/api/recordings/[file].wav

### Notes :

- Le frontend est statique (React build)
- Toute modification React nécessite un rebuild
- Les fichiers audio sont stockés dans un volume (persistant)
- Projet volontairement simple et sans auth (LAN)
