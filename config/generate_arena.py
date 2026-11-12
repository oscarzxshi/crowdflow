#!/usr/bin/env python3
"""
Generates `config/arena.xml`: a 20 m x 15 m indoor concert floor for 100 pedestrians.

Venue layout (world frame, metres):
    x in [-10, 10], y in [-7.5, 7.5]
    Stage on the WEST wall (x < -8.6), crowd barrier at x = -8.4.
    Bar counter on the EAST wall.

Sizing rationale: 100 people on ~270 m2 of usable floor is ~0.37 people/m2 on average
(comfortable), but the strip in front of the stage centre is only a few m2, so fans
pressing forward can push local density past 4-5 people/m2 if nothing intervenes.

Pedestrians are 70 kg cylinders (r = 0.22 m, shoulder width ~0.44 m) driven by
velocity actuators on two slide joints. Heads share one warm skin tone that no clothing
or venue surface uses, so the downward cameras can count heads by colour (bodies merge in
dense crowds; heads stay separated). Clothing uses a muted palette with no warm hues.

Run:  python config/generate_arena.py
"""

import os
import numpy as np

NUM_PEDS = 100
VENUE_X = 10.0          # half-length (x in [-10, 10])
VENUE_Y = 7.5           # half-width  (y in [-7.5, 7.5])
STAGE_X = -8.6          # stage front edge
BARRIER_X = -8.4        # crowd barrier line
PED_RADIUS = 0.22
HEAD_RGBA = "0.88 0.66 0.46 1"
SHIRT_COLOURS = [            # muted, no warm hues (the detector looks for the head tone)
    "0.16 0.20 0.32 1", "0.22 0.23 0.25 1", "0.32 0.36 0.42 1", "0.24 0.28 0.21 1",
    "0.40 0.13 0.20 1", "0.22 0.30 0.42 1", "0.60 0.61 0.64 1", "0.10 0.10 0.11 1",
]
OVERVIEW_W, OVERVIEW_H = 1280, 720


def look_at_xyaxes(pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera xyaxes for a camera at `pos` looking at `target`."""
    pos, target, up = map(np.asarray, (pos, target, up))
    d = target - pos
    d = d / np.linalg.norm(d)
    x = np.cross(d, up)
    x = x / np.linalg.norm(x)
    y = np.cross(x, d)
    return " ".join(f"{v:.4f}" for v in np.concatenate([x, y]))


def drone_xml(name, start):
    # Visual-only geometry (contype/conaffinity 0): drawn about 1.8x real size, light grey and
    # slightly self-lit so the drones stay visible in the overview camera, with a coloured
    # status light on top identifying each drone (Alpha amber, Bravo cyan).
    # Nothing here is below the downward camera, so the drone's own view is unchanged.
    # The spotlight is drawn as a red ring on the floor rather than a MuJoCo <light>:
    # on macOS, spot lights attached to the drones made the drones render black.
    return f"""
        <body name="drone_{name}" mocap="true" pos="{start}">
            <geom name="{name}_core" type="box" size="0.42 0.42 0.10" material="drone_body_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_arm1" type="capsule" fromto="-0.65 -0.65 0 0.65 0.65 0" size="0.05" material="drone_body_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_arm2" type="capsule" fromto="-0.65 0.65 0 0.65 -0.65 0" size="0.05" material="drone_body_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_r1" type="cylinder" pos="0.65 0.65 0.06" size="0.30 0.015" material="rotor_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_r2" type="cylinder" pos="0.65 -0.65 0.06" size="0.30 0.015" material="rotor_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_r3" type="cylinder" pos="-0.65 0.65 0.06" size="0.30 0.015" material="rotor_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_r4" type="cylinder" pos="-0.65 -0.65 0.06" size="0.30 0.015" material="rotor_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_led" type="sphere" pos="0 0 0.13" size="0.13" material="{name}_led_mat" contype="0" conaffinity="0"/>
            <geom name="{name}_beacon_bulb" type="sphere" pos="0 0 -0.09" size="0.11" rgba="0.2 0.2 0.2 0.5" contype="0" conaffinity="0"/>
            <camera name="cam_{name}" pos="0 0 -0.10" xyaxes="1 0 0 0 1 0" fovy="78"/>
        </body>
        <body name="beacon_disc_{name}" mocap="true" pos="0 0 -2.0">
            <geom name="disc_{name}_outer" type="cylinder" size="2.0 0.008" contype="0" conaffinity="0" rgba="1.0 0.05 0.05 0.30"/>
            <geom name="disc_{name}_inner" type="cylinder" size="1.0 0.010" contype="0" conaffinity="0" rgba="1.0 0.15 0.10 0.40"/>
        </body>"""


def main():
    world_cam = look_at_xyaxes(pos=(3.0, -19.5, 13.5), target=(-0.5, 0.8, 1.2))
    ped_bodies, actuators = [], []
    for i in range(NUM_PEDS):
        shirt = SHIRT_COLOURS[i % len(SHIRT_COLOURS)]
        ped_bodies.append(f"""
        <body name="ped_{i}" pos="0 0 0.45">
            <joint name="ped_{i}_jx" type="slide" axis="1 0 0"/>
            <joint name="ped_{i}_jy" type="slide" axis="0 1 0"/>
            <geom name="ped_{i}_geom" type="cylinder" size="{PED_RADIUS} 0.40" rgba="{shirt}" mass="70.0" friction="0.3 0.005 0.0001"/>
            <geom name="ped_{i}_head" type="sphere" pos="0 0 0.50" size="0.15" rgba="{HEAD_RGBA}" mass="0.1" contype="0" conaffinity="0"/>
        </body>""")
        for ax in ("x", "y"):
            # Velocity servo: force = kv * (v_desired - v), capped at a realistic push force.
            actuators.append(
                f'        <velocity name="act_ped_{i}_{ax}" joint="ped_{i}_j{ax}" kv="140" '
                f'ctrlrange="-2.0 2.0" forcerange="-260 260"/>')

    hx, hy = VENUE_X, VENUE_Y
    stage_cx = (-hx + STAGE_X) / 2.0
    stage_hx = (STAGE_X + hx) / 2.0
    xml = f"""<mujoco model="crowdflow_concert_floor">
    <!-- AUTO-GENERATED by config/generate_arena.py. Edit the generator, not this file. -->
    <option timestep="0.005" gravity="0 0 -9.81"/>

    <visual>
        <headlight ambient="0.40 0.40 0.42" diffuse="0.42 0.42 0.44" specular="0.08 0.08 0.08"/>
        <rgba haze="0.07 0.075 0.085 1"/>
        <global offwidth="{OVERVIEW_W}" offheight="{OVERVIEW_H}"/>
        <quality shadowsize="0"/>
    </visual>

    <asset>
        <texture name="sky_tex" type="skybox" builtin="gradient" rgb1="0.13 0.14 0.16" rgb2="0.04 0.045 0.05" width="512" height="3072"/>
        <texture name="grid_tex" type="2d" builtin="flat" mark="edge" rgb1="0.19 0.20 0.22" markrgb="0.25 0.26 0.29" width="256" height="256"/>
        <material name="floor_mat" texture="grid_tex" texrepeat="20 15" reflectance="0"/>
        <material name="wall_mat" rgba="0.27 0.28 0.31 1"/>
        <material name="stage_mat" rgba="0.17 0.18 0.22 1"/>
        <material name="barrier_mat" rgba="0.62 0.63 0.66 1"/>
        <material name="bar_mat" rgba="0.29 0.25 0.22 1"/>
        <material name="drone_body_mat" rgba="0.72 0.74 0.78 1" emission="0.35"/>
        <material name="rotor_mat" rgba="0.85 0.87 0.90 0.45" emission="0.3"/>
        <material name="alpha_led_mat" rgba="1.0 0.72 0.20 1" emission="1"/>
        <material name="bravo_led_mat" rgba="0.30 0.80 1.0 1" emission="1"/>
    </asset>

    <worldbody>
        <light name="arena_sun" pos="0 0 14" dir="0 0 -1" directional="true" castshadow="false" diffuse="0.50 0.50 0.55" specular="0.1 0.1 0.1"/>
        <geom name="floor" type="plane" size="{hx + 1} {hy + 1} 0.1" material="floor_mat"/>

        <camera name="world_cam" pos="3.0 -19.5 13.5" xyaxes="{world_cam}" fovy="42"/>

        <!-- Perimeter walls -->
        <geom name="wall_north" type="box" pos="0 {hy + 0.2} 1.5" size="{hx + 0.4} 0.2 1.5" material="wall_mat"/>
        <geom name="wall_south" type="box" pos="0 {-hy - 0.2} 0.6" size="{hx + 0.4} 0.2 0.6" material="wall_mat"/>
        <geom name="wall_east"  type="box" pos="{hx + 0.2} 0 1.5" size="0.2 {hy} 1.5" material="wall_mat"/>
        <geom name="wall_west"  type="box" pos="{-hx - 0.2} 0 1.5" size="0.2 {hy} 1.5" material="wall_mat"/>

        <!-- Stage (west) and front-of-stage crowd barrier -->
        <geom name="stage" type="box" pos="{stage_cx} 0 0.5" size="{stage_hx} 5.5 0.5" material="stage_mat"/>
        <geom name="barrier" type="box" pos="{BARRIER_X + 0.05} 0 0.55" size="0.05 5.8 0.55" material="barrier_mat"/>

        <!-- Bar counter (east) -->
        <geom name="bar" type="box" pos="{hx - 0.5} 0 0.55" size="0.5 3.0 0.55" material="bar_mat"/>
{"".join(drone_xml(n, s) for n, s in (("alpha", "-5 0 7"), ("bravo", "5 0 7")))}
{"".join(ped_bodies)}
    </worldbody>

    <actuator>
{chr(10).join(actuators)}
    </actuator>
</mujoco>
"""
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena.xml")
    with open(out, "w") as f:
        f.write(xml)
    print(f"Wrote {out} ({NUM_PEDS} pedestrians)")


if __name__ == "__main__":
    main()
