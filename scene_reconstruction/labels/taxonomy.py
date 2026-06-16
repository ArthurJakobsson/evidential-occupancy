"""Mappings between source label spaces and the Occ3D-nuScenes taxonomy.

Occ3D-nuScenes semantic indices (see scene_reconstruction/visualization/colormap.py):
0=other, 1=barrier, 2=bicycle, 3=bus, 4=car, 5=construction_vehicle, 6=motorcycle,
7=pedestrian, 8=traffic_cone, 9=trailer, 10=truck, 11=driveable_surface, 12=other_flat,
13=sidewalk, 14=terrain, 15=manmade, 16=vegetation, 17=free.

Classes 1..10 are the foreground "thing" classes that 3D bounding boxes cover; 11..16 are
"stuff" classes (only reachable from lidarseg / Occ3D / open-vocab).
"""

from __future__ import annotations

# Exact nuScenes annotation category name -> Occ3D class index (keyed on category NAME, not the
# synthesized category.index, so it is correct with or without the lidarseg expansion installed).
NUSCENES_TO_OCC3D: dict[str, int] = {
    "movable_object.barrier": 1,
    "vehicle.bicycle": 2,
    "vehicle.bus.bendy": 3,
    "vehicle.bus.rigid": 3,
    "vehicle.car": 4,
    "vehicle.emergency.ambulance": 4,  # car-like; mapped to car
    "vehicle.emergency.police": 4,
    "vehicle.construction": 5,
    "vehicle.motorcycle": 6,
    "human.pedestrian.adult": 7,
    "human.pedestrian.child": 7,
    "human.pedestrian.construction_worker": 7,
    "human.pedestrian.personal_mobility": 7,
    "human.pedestrian.police_officer": 7,
    "human.pedestrian.stroller": 7,
    "human.pedestrian.wheelchair": 7,
    "movable_object.trafficcone": 8,
    "vehicle.trailer": 9,
    "vehicle.truck": 10,
    # not a clean foreground box class -> background / unlabeled
    "animal": 0,
    "movable_object.debris": 0,
    "movable_object.pushable_pullable": 0,
    "static_object.bicycle_rack": 0,
}


def nuscenes_name_to_occ3d(name: str | None) -> int:
    """nuScenes category name -> Occ3D class (0 = background / not a box class)."""
    if not name:
        return 0
    return NUSCENES_TO_OCC3D.get(name, 0)
