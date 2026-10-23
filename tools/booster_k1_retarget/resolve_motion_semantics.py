"""Conservative labels from cached source descriptions, never invented trials."""

import html
import argparse
import json
from pathlib import Path
import re

from localize_and_salvage import dump

ROOT = Path(__file__).resolve().parents[2] / "data/holomotion_k1_v1"


def resolve(description, subject_context=""):
    text = description.strip()
    lower = text.lower()
    if re.search(r"bad amc|corrupt|bad data", lower):
        return "source_marked_bad", None, "source_explicit_bad_motion"
    if not text:
        return (
            "source_trial_description_missing",
            None,
            (
                "scene_required_from_subject_context"
                if re.search(r"obstacle|terrain|stair", subject_context, re.I)
                else "missing_trial_semantics"
            ),
        )
    if re.search(
        r"vacation|nursery rhyme|^story$|mickey|mummy|ghost|superhero|devil|human subject|"
        r"^bear$|^dog$|^cat$|^mouse$|^snake$|^fish$|^bird$|chicken|prairie dog|"
        r"penguin|dragon|squirrel|pterosaur|tyrannosaurus|^elephant$|zombie",
        lower,
    ):
        return (
            "narrative_or_character_performance",
            "L4",
            "mixed_performance_requires_clip_review",
        )
    if re.search(
        r"box|suitcase|(?:\d\s*(?:lb|lbs))|bottle|soda|wrench|hammer|umbrella|sewing|"
        r"batter|chalkboard|phone|soup|driving|headphones|paint|putt|placing tee|"
        r"swim|swing$|washing|clean|fishing|sweep|screw|shoot|gun|coffee mug|"
        r"pick up|lift|throw|frisbee|catch|mop|sawing|nail|bolt",
        lower,
    ):
        return (
            "object_or_environment_interaction",
            "L5",
            "object_or_environment_semantics",
        )
    if re.search(
        r"high five|slap hands|shaking hands|subjects|pulls up", lower
    ):
        return (
            "multi_person_or_contact_gesture",
            "L5",
            "partner_contact_semantics",
        )
    if re.search(
        r"stepping over|stepping up|step.*down|duck under|laying down|get.*up",
        lower,
    ):
        return (
            "obstacle_or_floor_transition",
            "L5",
            "scene_or_floor_contact_semantics",
        )
    if re.search(
        r"laugh|cry|shrug|wave|point|traffic|basketball signals|looking around|"
        r"wait for bus|stretch|conduct|surpris|scared|freezing|waving|giving directions|peek.?a.?boo",
        lower,
    ):
        return "gesture_or_upper_body_expression", "L2", None
    if re.search(
        r"sneak|creep|march|limp|smallforward|stop and go|curtsey|mope|"
        r"step forward$|go forward|stop ang go|pacing|shifting weight|ducking",
        lower,
    ):
        return "stylized_steps_or_postural_transition", "L3", None
    if lower in {"90", "180", "270", "360"}:
        return "rotation_label_without_direction", "L4", None
    if "charleston" in lower or "ring around the rosie" in lower:
        return "named_dance", "L4", None
    if re.search(r"swordplay|bassai|empi", lower):
        return "named_form_or_weapon_performance", "L4", "form_or_prop_review"
    return "unresolved_specific_motion", None, "manual_semantic_review"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--additional-snapshot", type=Path)
    args = parser.parse_args()
    rows = json.loads(
        (ROOT / "organized_frame_salvage_v2/all_motions.json").read_text()
    )
    if args.additional_snapshot:
        rows += json.loads(
            (args.additional_snapshot / "all_motions.json").read_text()
        )
    existing = ROOT / "remaining_repair_v1/semantic_overrides.json"
    overrides = json.loads(existing.read_text()) if existing.exists() else {}
    for row in rows:
        if row.get("semantic_hold") != "unresolved_action_description":
            continue
        name = row.get("parent_motion", row["motion"])
        key = row["source"] + ":" + name
        subject = name.split("_")[0]
        page = ROOT / "catalog/cmu_pages" / (subject + ".html")
        context = ""
        if page.exists():
            found = re.search(
                r"Subject #\d+\s*\((.*?)\)", page.read_text(), re.S
            )
            if found:
                context = html.unescape(
                    re.sub("<[^>]+>", "", found.group(1))
                ).strip()
        category, level, hold = resolve(row["action_description"], context)
        overrides[key] = {
            "category": category,
            "difficulty": level,
            "hold": hold,
            "trial_description": row["action_description"],
            "subject_context": context,
            "evidence": str(page.resolve()),
            "basis": "cached_official_trial_description; subject context never substituted as exact trial label",
        }
    dump(ROOT / "remaining_repair_v1/semantic_overrides.json", overrides)
    from collections import Counter

    print(dict(Counter(x["category"] for x in overrides.values())))


if __name__ == "__main__":
    main()
