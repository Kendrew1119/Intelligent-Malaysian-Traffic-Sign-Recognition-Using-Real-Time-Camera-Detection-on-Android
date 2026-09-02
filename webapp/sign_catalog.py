from __future__ import annotations

from typing import Any


# Plain-language summaries based on the Malaysian JKR traffic-sign categories.
# Wording is intentionally conservative: the physical sign and road markings
# remain authoritative during actual driving.
SIGN_DETAILS: dict[str, tuple[str, str]] = {
    "accident-prone-area-warning": ("Accident-prone area ahead.", "Reduce speed and watch carefully for hazards."),
    "bicycle-path": ("Designated route or lane for bicycles.", "Keep motor vehicles out of the bicycle path."),
    "bicycle-warning": ("Cyclists may enter or cross the road ahead.", "Slow down and give cyclists sufficient space."),
    "bumps-warning": ("Road bumps or an uneven raised section ahead.", "Reduce speed before reaching the bumps."),
    "bus-stop": ("Designated bus stop area.", "Watch for buses and passengers and do not obstruct the stop."),
    "camera-operation-zone": ("Traffic-enforcement camera zone.", "Follow the posted speed limit and road rules."),
    "cars-only": ("Route or lane intended for motor cars only.", "Enter only when driving a permitted vehicle."),
    "chevron-left": ("The road alignment changes sharply to the left.", "Slow down and follow the leftward road alignment."),
    "chevron-right": ("The road alignment changes sharply to the right.", "Slow down and follow the rightward road alignment."),
    "children-crossing-warning": ("Children may cross the road ahead.", "Slow down and be ready to stop."),
    "construction-ahead-warning": ("Road works or construction activity ahead.", "Reduce speed and follow temporary traffic controls."),
    "cow-nearby-warning": ("Cattle or other livestock may be on the road.", "Slow down and be prepared for animals entering the lane."),
    "crossroad-left-warning": ("A side road or junction joins from the left.", "Slow down and watch for entering or crossing traffic."),
    "crossroad-right-warning": ("A side road or junction joins from the right.", "Slow down and watch for entering or crossing traffic."),
    "gated-railway-crossing-ahead-warning": ("A gated railway crossing is ahead.", "Slow down and stop when the gate or signals require it."),
    "general-warning": ("An unspecified road hazard is ahead.", "Slow down and look for additional signs or hazards."),
    "give-way": ("Give priority to traffic on the road being entered.", "Slow down and stop if necessary before proceeding."),
    "height-limit": ("Vehicles above the displayed height are restricted.", "Do not proceed if the vehicle exceeds the shown limit."),
    "left-or-right": ("Traffic must proceed either left or right.", "Choose one of the indicated directions; do not continue straight."),
    "left-turn-only": ("Traffic must turn left.", "Move safely into position and turn left only."),
    "no-cars": ("Motor cars are prohibited beyond this point.", "Do not enter with a motor car."),
    "no-entry": ("Entry is prohibited from this direction.", "Do not enter; use an authorised route."),
    "no-horn": ("Sounding the horn is prohibited except for immediate danger.", "Avoid using the horn in this zone."),
    "no-left": ("Left turns are prohibited.", "Continue using a permitted direction."),
    "no-left-and-right": ("Both left and right turns are prohibited.", "Continue only in the permitted direction."),
    "no-overtaking": ("Overtaking is prohibited.", "Remain behind other vehicles until the restriction ends."),
    "no-parking": ("Parking is prohibited in this area.", "Do not leave the vehicle parked here."),
    "no-right": ("Right turns are prohibited.", "Continue using a permitted direction."),
    "no-straight": ("Continuing straight is prohibited.", "Turn only in a permitted direction."),
    "no-straight-or-left": ("Continuing straight and turning left are prohibited.", "Use only the permitted direction."),
    "no-uturn": ("U-turns are prohibited.", "Continue ahead and use an authorised turning location."),
    "parking-area": ("Designated parking area.", "Park only within the marked spaces and conditions."),
    "pass-obstacle-on-either-side": ("An obstacle may be passed on either side.", "Choose the safe available side and keep clear of the obstacle."),
    "pass-right": ("Traffic must pass to the right of the obstacle.", "Keep right as indicated."),
    "pedestrian-crossing-warning": ("A pedestrian crossing or pedestrian activity is ahead.", "Slow down and give way to pedestrians."),
    "railway-crossing-ahead-warning": ("A railway crossing is ahead.", "Slow down, check for trains and obey crossing signals."),
    "reverse-turn-warning": ("A reverse or successive sharp turn is ahead.", "Reduce speed and follow the road alignment carefully."),
    "right-turn-only": ("Traffic must turn right.", "Move safely into position and turn right only."),
    "road-narrows-left-warning": ("The road narrows from the left side ahead.", "Reduce speed and keep safely within the available lane."),
    "road-narrows-right-warning": ("The road narrows from the right side ahead.", "Reduce speed and keep safely within the available lane."),
    "roadway-diverges-warning": ("The roadway divides or diverges ahead.", "Choose the correct lane early and avoid sudden lane changes."),
    "roundabout": ("Traffic must follow the roundabout direction.", "Slow down, give way as required and follow the circular flow."),
    "sharp-right-turn-warning": ("A sharp right turn is ahead.", "Reduce speed before the turn."),
    "slippery-road-warning": ("The road surface may be slippery.", "Reduce speed and avoid sudden braking or steering."),
    "slowdown-warning": ("A condition ahead requires lower speed.", "Slow down and prepare for the road condition ahead."),
    "speed-limit-15": ("Maximum posted speed is 15 km/h.", "Do not exceed 15 km/h while the limit applies."),
    "speed-limit-30": ("Maximum posted speed is 30 km/h.", "Do not exceed 30 km/h while the limit applies."),
    "speed-limit-40": ("Maximum posted speed is 40 km/h.", "Do not exceed 40 km/h while the limit applies."),
    "speed-limit-5": ("Maximum posted speed is 5 km/h.", "Do not exceed 5 km/h while the limit applies."),
    "speed-limit-50": ("Maximum posted speed is 50 km/h.", "Do not exceed 50 km/h while the limit applies."),
    "speed-limit-60": ("Maximum posted speed is 60 km/h.", "Do not exceed 60 km/h while the limit applies."),
    "speed-limit-80": ("Maximum posted speed is 80 km/h.", "Do not exceed 80 km/h while the limit applies."),
    "steep-descent-warning": ("A steep downhill section is ahead.", "Reduce speed and use an appropriate lower gear."),
    "stop-for-inspection": ("Vehicles must stop for inspection or control.", "Stop at the indicated point and follow authorised instructions."),
    "stop-sign": ("A complete stop is required.", "Stop at the line, check all directions and proceed only when safe."),
    "straight-only": ("Traffic must continue straight.", "Do not turn left or right."),
    "straight-or-right": ("Traffic may continue straight or turn right.", "Use one of the indicated directions."),
    "towing-area": ("Unauthorised or improperly parked vehicles may be towed.", "Check the parking restrictions before leaving the vehicle."),
    "traffic-light-ahead": ("Traffic signals are ahead.", "Reduce speed and be ready to stop for the signal."),
    "use-horn": ("Sound the horn to warn other road users.", "Use the horn briefly where required, then proceed carefully."),
    "uturn-lane": ("This lane is intended for making a U-turn.", "Use the lane only when the U-turn is permitted and safe."),
    "village-ahead-warning": ("A village or settled area is ahead.", "Reduce speed and watch for pedestrians and local traffic."),
    "winding-road-warning": ("A series of bends is ahead.", "Reduce speed and remain within the lane through the bends."),
}


SPEECH_OVERRIDES: dict[str, str] = {
    "camera-operation-zone": "Traffic enforcement camera ahead.",
    "give-way": "Give way ahead.",
    "left-or-right": "Turn left or right ahead.",
    "left-turn-only": "Turn left only ahead.",
    "no-left": "No left turn ahead.",
    "no-left-and-right": "No left or right turn ahead.",
    "no-right": "No right turn ahead.",
    "no-straight": "Straight movement is prohibited ahead.",
    "no-straight-or-left": "Do not continue straight or turn left.",
    "no-uturn": "No U-turn ahead.",
    "pass-obstacle-on-either-side": "Pass the obstacle on either side.",
    "pass-right": "Pass the obstacle on the right.",
    "right-turn-only": "Turn right only ahead.",
    "stop-for-inspection": "Stop for inspection ahead.",
    "stop-sign": "Stop sign ahead.",
    "straight-only": "Continue straight.",
    "straight-or-right": "Continue straight or turn right.",
    "use-horn": "Use the horn ahead.",
    "uturn-lane": "U-turn lane ahead.",
}


def build_speech_phrase(class_name: str) -> str:
    """Return a short phrase suitable for browser speech synthesis."""
    if class_name.startswith("speed-limit-"):
        speed = class_name.rsplit("-", 1)[-1]
        return f"Speed limit {speed} kilometres per hour ahead."
    if class_name in SPEECH_OVERRIDES:
        return SPEECH_OVERRIDES[class_name]
    readable_name = class_name.replace("-warning", "").replace("-", " ")
    return f"{readable_name.capitalize()} ahead."


def build_catalog(class_names: dict[int, str]) -> list[dict[str, Any]]:
    missing = set(class_names.values()) - set(SIGN_DETAILS)
    extra = set(SIGN_DETAILS) - set(class_names.values())
    if missing or extra:
        raise ValueError(f"Sign catalogue mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}")

    return [
        {
            "class_id": class_id,
            "class_name": class_name,
            "meaning": SIGN_DETAILS[class_name][0],
            "action": SIGN_DETAILS[class_name][1],
            "speech": build_speech_phrase(class_name),
        }
        for class_id, class_name in sorted(class_names.items())
    ]
