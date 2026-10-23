import importlib.util
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools/booster_k1_retarget"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "semantics", TOOLS / "resolve_motion_semantics.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_missing_trial_not_invented_from_subject():
    category, level, hold = MODULE.resolve("", "Walking with obstacles 1")
    assert category == "source_trial_description_missing"
    assert level is None
    assert hold == "scene_required_from_subject_context"


def test_story_not_automatically_released():
    assert MODULE.resolve("Alaskan vacation")[2] is not None
    assert MODULE.resolve("monkey (human subject)")[2] is not None


def test_explicit_gestures():
    assert MODULE.resolve("Shrug") == (
        "gesture_or_upper_body_expression",
        "L2",
        None,
    )
    assert MODULE.resolve("basketball signals")[2] is None


def test_object_and_unknown_remain_held():
    assert MODULE.resolve("carry 12.5lb suitcase")[2] is not None
    assert MODULE.resolve("UpRightSequence")[2] is not None


def test_rotation_does_not_guess_direction():
    assert MODULE.resolve("180")[0] == "rotation_label_without_direction"
