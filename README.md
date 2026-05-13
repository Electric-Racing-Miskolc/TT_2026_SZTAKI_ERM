# Real2Sim — Unitree G1 webkamerás vezérlés (analitikus retargeting)

Felsőtest-utánzó "real2sim" rendszer: a laptop webkamerája figyeli az operátor
mozgását, és a karok pózát **direkt analitikus joint-angle számítás** segítségével
ülteti át egy Unitree G1 humanoid modellre MuJoCo-ban valós időben.  A demó
videó osztott képernyőn mutatja a kamerát (skeleton overlay-jel) és a szimulációt.

A SZTAKI ERM 2026 tervezési feladata.  Licensz: **CC0 1.0 Universal**.

## Architektúra

```
Webkamera → MediaPipe Pose (Tasks API)
                │ 33 világ-landmark (méter, csípő-centrált)
         One-Euro szűrő (adaptív simítás, 33×3 csatorna)
                │
         3-pózos antropometriai kalibráció (T, A, könyök-90°)
         → shoulder_width, torso_height, arm_length, anatomical_zero
                │
         retarget.compute_arm_angles()
         → direkt joint-szög számítás (zárt képletek, NO IK)
            ─ shoulder_pitch = atan2(u_y, hypot(u_x, u_z))
            ─ shoulder_roll  = atan2(u_x, -u_z)            (gimbal-lock-tolerant)
            ─ elbow          = acos(u_upper · u_fore)
            ─ shoulder_yaw, wrist_roll/pitch/yaw = 0 (V1)
         → SIGN/OFFSET tábla (anatómiai → G1 konvenció)
         → JOINT-LIMIT clamp (MJCF range)
                │
         MuJoCo G1 (pozíció-aktuátorok, torzo rögzítve)
                │
         split-screen mp4 / passzív viewer
```

- **Computer vision**: Google MediaPipe Pose (Tasks API).  33 testpont, hip-centered 3D.
- **Szűrés**: One-Euro Filter — adaptív cutoff, kevesebb lag, mint az EMA.
- **Kalibráció**: 3-pózos antropometriai mérés (`models/anthropo.json` cache).
- **Fizika**: MuJoCo 3.8 + `mujoco_menagerie/unitree_g1` MJCF.
- **Retargeting**: tisztán analitikus (numpy, ~1 ms/frame, függvénykönyvtár-mentes).
- **openclaw**: a pipeline skill-wrapper-ként elérhető (`openclaw/skills/real2sim/`).

### Miért analitikus retargeting, nem IK?

A korábbi `mink` differenciális IK megoldó instabil volt, mert csak a tenyér
**pozícióját** célozta — ugyanazt a wrist-pozíciót sok ízületi konfiguráció
elérhette, és minden frame-ben a solver mást választott (rángatás).  Az analitikus
megközelítés ehelyett **közvetlenül kiszámítja** minden ízületi szöget egy zárt
képlettel a felhasználó csontváz-vektoraiból; ugyanaz a bemenet mindig ugyanazt a
kimenetet adja, így nincs solver-konvergencia, nincs posture-cost tuning, és a
joint-limitek hard-clamp-pel kényszerítettek.  Ezt a megközelítést használja a
[Humandroid](https://github.com/vellons/Humandroid), a TonyPi pose-mimic, és az
ipari kinect-teleop projektek többsége.

## Telepítés (Windows, bash / Git Bash)

```bash
# 1. Klónozás
git clone https://github.com/<repo>/TT_2026_SZTAKI_ERM.git
cd TT_2026_SZTAKI_ERM

# 2. Unitree G1 MJCF (szükséges, nem tracked a repóban)
git clone https://github.com/google-deepmind/mujoco_menagerie.git

# 3. Python venv
python -m venv real2sim_env
source real2sim_env/Scripts/activate    # Windows Git Bash
# source real2sim_env/bin/activate       # Linux/macOS

# 4. Függőségek
pip install -r requirements.txt

# 5. MediaPipe pose modell (~9 MB)
mkdir -p models
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

## Használat

### GUI

```bash
python scripts/run_gui.py [--camera IDX]
```

A GUI ablakban két preview (kamera + szimuláció), Start/Stop/Kalibráció gombok,
oldalt csúszkák (One-Euro paraméterek, FPS, visibility threshold), jobbra az
aktuális 14 ízületi szög sávdiagramokkal.

### CLI

```bash
# Élő MuJoCo viewer + kamera-preview (első indításkor kalibráció kell)
python scripts/run_real2sim.py --viewer

# Kalibráció újrafuttatása (3 póz)
python scripts/run_real2sim.py --viewer --recalibrate

# Demó mp4 felvétele 20 másodpercig
python scripts/run_real2sim.py --record demo.mp4 --duration 20

# Csak kamera + skeleton overlay (hardver-ellenőrzés, szimuláció nélkül)
python scripts/run_real2sim.py --pose-only

# Joint-szög hibakeresés (target-ek kiírása 30 frame-enként)
python scripts/run_real2sim.py --viewer --angle-debug
```

| Flag | Leírás |
|------|--------|
| `--pose-only` | Kamera + póz overlay; nincs MuJoCo |
| `--viewer` | Élő MuJoCo viewer + kamera ablak |
| `--record PATH` | Split-screen mp4 írása |
| `--duration N` | N másodperc után megáll (0 = `q`-ig) |
| `--camera IDX` | Kamera index (default 0) |
| `--recalibrate` | 3-pózos antropometriai kalibráció újrafuttatása |
| `--angle-debug` | Joint-szögek periodikus kiírása |
| `--no-mirror` | Ne tükrözze a kamera képet |

### 3-pózos kalibráció

Első indításkor (vagy `--recalibrate` flaggel) a rendszer három pózt mér le
3-3 másodperc alatt:

1. **T-pose** — karok kinyújtva oldalra (arm length mérés)
2. **A-pose** — karok lazán lelógatva (shoulder width, torso height, *anatomical zero*)
3. **Elbow 90°** — karok előre, könyök 90°-os hajlat (validáció)

Az anatomical-zero a felhasználó természetes nyugalmi póz-eltolását rögzíti,
így a "karok leengedve" mindig pontosan a robot zero joint-szögeire képződik
le, függetlenül a MediaPipe minimális elcsúszásaitól.

Az eredmény a `models/anthropo.json`-ba kerül, és következő indításkor
automatikusan betöltődik.

### OpenClaw skill

```bash
python openclaw/skills/real2sim/scripts/run.py --action status
python openclaw/skills/real2sim/scripts/run.py --action start
python openclaw/skills/real2sim/scripts/run.py --action stop
python openclaw/skills/real2sim/scripts/run.py --action calibrate
python openclaw/skills/real2sim/scripts/run.py --action record --out demo.mp4 --duration 20
```

## Tesztelés

```bash
# Retarget joint-szög unit tesztek (kamera nélkül, szintetikus pózokon)
python -m pytest tests/test_retarget.py -v

# Antropometriai kalibráció unit tesztek
python -m pytest tests/test_calibration.py -v

# Kamera-diagnosztika
python tests/camera_diag.py

# Élő pose-detection ellenőrzés (csak kamera + skeleton overlay)
python tests/smoke_pose.py
```

A `tests/test_retarget.py` tartalmazza a kanonikus pózok (T, A, kar-előre,
könyök-90°, kar-fent) elvárt joint-szögeit szintetikus landmark vektorokon —
ez biztosítja, hogy a matematika minden CI futáson is helyes maradjon.

## Projekt szerkezet

```
real2sim/
├── config.py          konstansok (joint nevek, sign/offset/limit táblák, FPS)
├── pose.py            MediaPipe Tasks API wrapper
├── one_euro.py        One-Euro adaptív szűrő (landmark array)
├── calibration.py     3-pózos antropometriai kalibráció + anatomical_zero
├── retarget.py        ★ Analitikus joint-angle retargeting (core)
├── sim.py             MuJoCo G1 wrapper (rögzített torzo)
├── recorder.py        split-screen mp4 író
└── runner.py          fő ciklus (calibrate → capture → retarget → step → render)

scripts/
├── run_real2sim.py    CLI belépési pont
└── run_gui.py         tkinter GUI

openclaw/skills/real2sim/
├── SKILL.md           openclaw skill manifest
└── scripts/run.py     start / stop / status / record / calibrate

tests/
├── test_retarget.py     ★ retarget joint-szög unit tesztek
├── test_calibration.py  antropometriai kalibráció unit tesztek
├── smoke_pose.py        webkamera live póz teszt
└── camera_diag.py       kamera index és engedély-diagnosztika

models/
└── pose_landmarker_full.task   MediaPipe modell (nem tracked)
    anthropo.json               3-pózos kalibráció cache (generált)

mujoco_menagerie/   Google DeepMind (Apache-2.0, nem tracked)
openclaw/           MIT licensz
```

## Joint-szög számítás részletei

A `real2sim/retarget.py` modul minden frame-en:

1. **Body frame** épül a vállak + csípők alapján
   (`e_x` = subject left, `e_z` = up, `e_y` = forward).
2. A vállból elindulva a felkar és alkar irányvektorát test-frame-be forgatja.
3. A felkar irányból zárt képletekkel kihámozza:
   * `shoulder_pitch = atan2(u_y, hypot(u_x, u_z))`
   * `shoulder_roll  = atan2(u_x, -u_z)` (gimbal-lock guard: ha az arm a body-Y mentén áll, roll = 0)
4. `elbow = acos(clip(u_upper · u_fore, -1, 1))` — szög a felkar és alkar között.
5. `shoulder_yaw`, `wrist_roll/pitch/yaw` = 0 (V1).  Ezek a következő fázisban
   jönnek a hand-landmark-okból (Mediapipe 15-21 / 16-22).
6. Az anatómiai szögek **SIGN × value + OFFSET** táblán keresztül G1 joint
   konvencióra váltanak, majd a `config.G1_JOINT_LOWER/_UPPER` limitekkel
   clamp-olódnak.

A SIGN/OFFSET tábla explicit a `real2sim/config.py`-ban — ha egy ízület rossz
irányba mozog, ott kell flippelni a sign-ját, nem a matematikát hekkelni.

## Licensz

Ez a repo **CC0 1.0 Universal** licenszű (lásd [LICENSE](LICENSE)).  Külső
komponensek a saját licenszüket öröklik:

- [mujoco_menagerie](mujoco_menagerie/LICENSE) — Apache 2.0
- [unitree_g1 MJCF](mujoco_menagerie/unitree_g1/LICENSE) — BSD-3
- [openclaw](openclaw/LICENSE) — MIT
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — Apache 2.0
