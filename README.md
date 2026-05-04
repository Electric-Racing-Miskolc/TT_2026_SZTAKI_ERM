# Real2Sim — Unitree G1 webkamerás vezérlés (mink IK)

Felsőtest-utánzó "real2sim" rendszer: a laptop webkamerája figyeli az operátor
mozgását, és a karok pózát egy **mink differenciális IK** solver segítségével
ültetjük át egy Unitree G1 humanoid modellre MuJoCo-ban valós időben.  A demó
videó osztott képernyőn mutatja a kamerát (skeleton overlay-jel) és a szimulációt.

A SZTAKI ERM 2026 tervezési feladata. Licensz: **CC0 1.0 Universal**.

![G1 a szimulációban](tests/smoke_pipeline_synth_frame100.jpg)

## Architektúra

```
Webkamera → MediaPipe Pose (Tasks API)
                │ 33 világ-landmark (méter, csípő-centrált)
         One-Euro szűrő (adaptív simítás, 33×3 csatorna)
                │
         T-pose kalibráció (induláskor, egyszer)
         → kar-arány skálázás (user ↔ G1)
                │
         mink FrameTask IK
         tasks: left_palm + right_palm site célok (14 DoF / 2 kar)
         solver: daqp (CPU, ~1 ms / frame)
                │
         MuJoCo G1 (pozíció-aktuátorok, torzo rögzítve)
                │
         split-screen mp4 / passzív viewer
```

- **Computer vision**: Google MediaPipe Pose (Tasks API). 33 testpont, hip-centered 3D.
- **Szűrés**: One-Euro Filter — adaptív cutoff, kevesebb lag mint EMA.
- **Kalibráció**: T-pose indításkor → kar-arány skálázó (`models/calibration.json`).
- **Fizika**: MuJoCo 3.8 + `mujoco_menagerie/unitree_g1` MJCF.
- **IK**: mink (Apache-2.0) differenciális IK + daqp QP solver.
- **openclaw**: a pipeline skill-wrapper-ként elérhető (`openclaw/skills/real2sim/`).

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

```bash
# Első futtatás — T-pose kalibráció + élő szimuláció
python scripts/run_real2sim.py --viewer

# Kalibráció újrafuttatása
python scripts/run_real2sim.py --viewer --recalibrate

# Demo mp4 felvétele 20 másodpercig
python scripts/run_real2sim.py --record demo.mp4 --duration 20

# Csak kamera + skeleton overlay (hardver-ellenőrzés, szimuláció nélkül)
python scripts/run_real2sim.py --pose-only

# IK hibakeresés (joint-szögek 30 frame-enként)
python scripts/run_real2sim.py --viewer --ik-debug
```

### CLI kapcsolók

| Flag | Leírás |
|------|--------|
| `--pose-only` | Kamera + póz overlay; nincs MuJoCo |
| `--viewer` | Élő MuJoCo viewer + kamera ablak |
| `--record PATH` | Split-screen mp4 írása |
| `--duration N` | N másodperc után megáll (0 = `q`-ig) |
| `--camera IDX` | Kamera index (default 0) |
| `--recalibrate` | T-pose kalibráció újrafuttatása |
| `--ik-debug` | IK joint-szögek periodikus kiírása |
| `--no-mirror` | Ne tükrözze a kamera képet |

### T-pose kalibráció

Az első indításkor (vagy `--recalibrate` flaggel) a rendszer megkér, hogy 3 másodpercig tartsd T-pózban a karjaid (oldalra kinyújtva). Ezalatt méri a váll-csukló távolságodat, és kiszámít egy skálázót (`scale = G1_kar / te_kared`). Az eredmény a `models/calibration.json`-ba kerül, és következő indításkor automatikusan betöltődik.

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
# Kalibráció matematika unit tesztek (kamera nélkül)
python -m pytest tests/test_calibration.py -v

# IK integráció tesztek (szükséges: mujoco_menagerie + mink)
python -m pytest tests/test_ik.py -v

# Kamera-diagnosztika
python tests/camera_diag.py

# End-to-end szintetikus pipeline (kamera nélkül, mp4-et ír)
python tests/smoke_pipeline_synth.py
```

## Projekt szerkezet

```
real2sim/
├── config.py          konstansok (joint nevek, IK paraméterek, FPS)
├── pose.py            MediaPipe Tasks API wrapper
├── one_euro.py        One-Euro adaptív szűrő (landmark array)
├── calibration.py     T-pose kalibráció, arm-arány skálázás
├── ik.py              mink FrameTask IK (left_palm + right_palm, 14 DoF)
├── sim.py             MuJoCo G1 wrapper (runtime MJCF patch, torzo pin)
├── filter.py          limit_delta segédfüggvény
└── runner.py          fő ciklus (calibrate → capture → IK → step → render)

scripts/
└── run_real2sim.py    CLI belépési pont

openclaw/skills/real2sim/
├── SKILL.md           openclaw skill manifest
└── scripts/run.py     start / stop / status / record / calibrate

tests/
├── test_calibration.py  kalibráció unit tesztek
├── test_ik.py           mink IK integráció tesztek
├── smoke_pose.py        webkamera live póz teszt
├── smoke_pipeline_synth.py  end-to-end mp4 kamera nélkül
└── camera_diag.py       kamera index és engedély-diagnosztika

models/
└── pose_landmarker_full.task   MediaPipe modell (nem tracked)
    calibration.json            T-pose skálázó cache (generált)

mujoco_menagerie/   Google DeepMind (Apache-2.0, nem tracked)
openclaw/           MIT licensz
```

## Pose → IK leképezés

A `real2sim/ik.py` minden kép-kockán:

1. **Body frame** épül a vállak + csípők alapján (e_x=bal, e_z=fel, e_y=előre).
2. A csukló vektora a vállhoz képest **testkeretbe forgatódik**, majd **skálázódik**
   a G1/user arány szerint.
3. A skálázott vektor visszakerül a **robot MuJoCo-világ keretébe** (torso xmat
   alapján kiszámított 3×3 mátrix).
4. A `mink.FrameTask` felveszi a bal/jobb `left_palm`/`right_palm` site célokat,
   és a `daqp` solver megoldja a `qvel` vektort.
5. A `mink.Configuration.integrate_inplace()` elvégzi az integrálást; az eredmény
   qpos a pozíció-aktuátorokhoz kerül (`data.ctrl`).

Az MJCF `left_palm`/`right_palm` site-ok runtime kerülnek be (nem módosítjuk a
submodule fájlját): a `sim.py` induláskor kétszer beilleszti a site-sort a g1.xml
tartalmába és ideiglenes fájlba ment, majd betöltés után törli.

## Licensz

Ez a repo **CC0 1.0 Universal** licenszű (lásd [LICENSE](LICENSE)). Külső
komponensek a saját licenszüket örlik:

- [mujoco_menagerie](mujoco_menagerie/LICENSE) — Apache 2.0
- [unitree_g1 MJCF](mujoco_menagerie/unitree_g1/LICENSE) — BSD-3
- [openclaw](openclaw/LICENSE) — MIT
- [mink](https://github.com/kevinzakka/mink) — Apache 2.0
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — Apache 2.0
