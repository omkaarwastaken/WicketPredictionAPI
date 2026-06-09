import math


def clamp(value, minimum=0.0, maximum=1.0):
    return max(min(value, maximum), minimum)


def predict_wicket(data):

    # -----------------------------------------
    # PHASE MULTIPLIER
    # -----------------------------------------

    phase_multiplier = {
        "powerplay": 1.10,
        "middle": 0.95,
        "death": 1.20
    }.get(data.phase.lower(), 1.0)

    # -----------------------------------------
    # PRESSURE
    # -----------------------------------------

    dot_rate = data.dots_last_12 / 12

    run_deficit = max(
        0,
        1 - (data.runs_last_12 / (12 * 1.35))
    )

    wicket_pressure = min(
        data.wickets_last_12 / 2,
        1
    )

    pressure = (
        dot_rate * 0.40 +
        run_deficit * 0.25 +
        wicket_pressure * 0.15 +
        (1 - data.batter_consistency / 100) * 0.10 +
        (data.bowler_pressure_handling / 100) * 0.10
    )

    pressure = clamp(pressure)

    # -----------------------------------------
    # CONTROL
    # -----------------------------------------

    batter_control = (
        (data.batter_consistency / 100) * 0.35 +
        (data.batter_pressure_handling / 100) * 0.25 +
        (data.batter_rating / 100) * 0.20 +
        (1 - data.batter_aggression / 100) * 0.20
    )

    bowler_control = (
        (data.bowler_rating / 100) * 0.40 +
        (data.bowler_death_skill / 100) * 0.35 +
        (data.bowler_pressure_handling / 100) * 0.25
    )

    control = (
        batter_control * 0.50 +
        (1 - bowler_control) * 0.50
    )

    control = clamp(control)

    # -----------------------------------------
    # RISK
    # -----------------------------------------

    aggression = (
        data.boundaries_last_12 +
        (data.sixes_last_12 * 2)
    ) / 12

    risk = (
        aggression * 0.30 +
        pressure * 0.30 +
        (1 - control) * 0.20 +
        (data.bowler_death_skill / 100) * 0.10 +
        (data.batter_aggression / 100) * 0.10
    )

    risk = clamp(risk)

    # -----------------------------------------
    # XWICKET
    # -----------------------------------------

    xwicket = (
        risk * 0.35 +
        pressure * 0.25 +
        (1 - control) * 0.20 +
        (data.bowler_rating / 100) * 0.10 +
        (data.batter_pressure_handling / 100) * 0.10
    )

    xwicket *= phase_multiplier

    xwicket = clamp(xwicket)

    # -----------------------------------------
    # THREAT SCORE
    # -----------------------------------------

    threat = (
        xwicket * 0.45 +
        risk * 0.20 +
        pressure * 0.20 +
        (1 - control) * 0.15
    )

    # -----------------------------------------
    # LOGISTIC CONVERSION
    # -----------------------------------------

    z = (threat * 8) - 4

    wicket_probability = (
        1 / (1 + math.exp(-z))
    )

    # -----------------------------------------
    # LABELS
    # -----------------------------------------

    percentage = round(
        wicket_probability * 100,
        2
    )

    if percentage >= 80:
        label = "Extreme Threat"

    elif percentage >= 60:
        label = "High Threat"

    elif percentage >= 40:
        label = "Developing Threat"

    elif percentage >= 20:
        label = "Occasional Threat"

    else:
        label = "Low Threat"

    return {
        "pressure": round(pressure, 4),
        "control": round(control, 4),
        "risk": round(risk, 4),
        "xwicket": round(xwicket, 4),
        "wicket_probability": round(wicket_probability, 4),
        "wicket_percentage": percentage,
        "label": label
    }