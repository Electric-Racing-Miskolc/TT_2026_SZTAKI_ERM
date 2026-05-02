import time
import math
import mujoco
import mujoco.viewer

# A színtér betöltése (gravitáció aktív marad)
model_path = "mujoco_menagerie/unitree_g1/scene.xml"

try:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 1. A karok aktuátorainak dinamikus azonosítása a nevük alapján
    arm_actuators = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and ("shoulder" in name.lower() or "elbow" in name.lower() or "arm" in name.lower()):
            arm_actuators.append((i, name))

    print(f"Rendszer inicializálva. Azonosított felsőtest-aktuátorok száma: {len(arm_actuators)}")

    # 2. A kiindulási póz mentése a törzs rögzítéséhez
    mujoco.mj_step(model, data)
    initial_root_qpos = data.qpos[:7].copy() # A szabad ízület (freejoint) 7 koordinátája (3 pozíció, 4 kvaternió)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()

        while viewer.is_running():
            current_time = time.time() - start_time
            
            # 3. Tesztállvány-szimuláció: A törzs helyzetének kényszerített megtartása
            data.qpos[:7] = initial_root_qpos
            data.qvel[:6] = 0.0

            # 4. Valósághű, összehangolt karmozgás generálása
            for act_id, name in arm_actuators:
                # Eltérő anatómiai mozgások a különböző ízületekre
                if "pitch" in name.lower():
                    # Előre-hátra emelés (váll/könyök)
                    data.ctrl[act_id] = math.sin(current_time * 1.5) * 0.8
                elif "roll" in name.lower():
                    # Oldalirányú emelés
                    data.ctrl[act_id] = abs(math.sin(current_time * 1.0)) * 0.5
                else:
                    # Egyéb (pl. csukló, yaw) finomabb mozgatása
                    data.ctrl[act_id] = math.cos(current_time * 2.0) * 0.3

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

except Exception as e:
    print(f"Kritikus hiba lépett fel a végrehajtás során: {e}")