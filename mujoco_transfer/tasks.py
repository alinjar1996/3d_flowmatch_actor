"""Names and independently audited fixtures for supported transfer tasks."""

from pathlib import Path


TASKS = {
    "bimanual_lift_ball": {
        "object_name": "ball", "additional_object_names": [], "static_object_names": [],
        "fixture": "peract2_scene.json",
    },
    "bimanual_lift_tray": {
        "object_name": "tray", "additional_object_names": ["item"],
        "static_object_names": ["drawer_legs", "drawer_legs0"],
        "fixture": "peract2_tray_scene.json",
    },
}


def task_spec(name="bimanual_lift_ball"):
    if name not in TASKS:
        raise ValueError(f"Unsupported transfer task {name!r}; choose from {tuple(TASKS)}")
    return TASKS[name]


def fixture_path(name="bimanual_lift_ball"):
    return Path(__file__).parent / "configs" / task_spec(name)["fixture"]
