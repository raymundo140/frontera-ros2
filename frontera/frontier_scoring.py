"""
frontier_scoring.py — candidate classes and the frontier utility function.

Utility of candidate i:

    U_i = Gn_i * exp(-lambda * L_i) * C_i * R_i * V_i

    Gn_i  normalized information gain in [0, 1]
          = visible unknown area / full sensor footprint area
    L_i   obstacle-aware path length from the robot [m]
    lambda travel-cost rate [1/m]; 1/lambda is the path length at which the
          value of a frontier drops to 37 %
    C_i   directional continuity = 1 + w_c * cos(theta_i), in [1-w_c, 1+w_c]
          theta_i = angle between the robot's recent travel direction and the
          initial direction of the path to the candidate
    R_i   recent-goal factor (< 1 near recently reached goals, else 1)
    V_i   unproductive-visit factor = penalty ** (unproductive visits)

All factors are positive and bounded, so U_i is finite and non-negative.
C, R and V are soft preferences: they re-rank candidates, they never
exclude one.
"""

import math

# Candidate classes used for selection and for the completion decision.
ELIGIBLE = 'ELIGIBLE'
TEMPORARILY_BLOCKED = 'TEMPORARILY_BLOCKED'
COOLDOWN = 'COOLDOWN'
UNREACHABLE = 'UNREACHABLE'
EXHAUSTED = 'EXHAUSTED'
LOW_INFORMATION_GAIN = 'LOW_INFORMATION_GAIN'
UNSAFE = 'UNSAFE'

ALL_CLASSES = (
    ELIGIBLE, TEMPORARILY_BLOCKED, COOLDOWN, UNREACHABLE,
    EXHAUSTED, LOW_INFORMATION_GAIN, UNSAFE,
)

# Classes that mean "exploration may still continue here later".
BLOCKING_CLASSES = (ELIGIBLE, TEMPORARILY_BLOCKED, COOLDOWN)


def continuity_factor(preferred_direction, candidate_direction, weight):
    """Compute the directional continuity factor C of a candidate.

    Args:
        preferred_direction: Unit vector (x, y) of the robot's recent travel
            direction in the map frame, or None if not yet known.
        candidate_direction: Unit vector (x, y) of the initial direction of
            the path to the candidate in the map frame, or None.
        weight: Continuity weight w_c, in [0, 1).

    Returns:
        Tuple (cosine, C): the cosine of the angle between both directions
        (None if either is unknown) and C = 1 + w_c * cosine, which is 1.0
        when no direction is known.
    """
    if preferred_direction is None or candidate_direction is None:
        return None, 1.0
    cosine = (
        preferred_direction[0] * candidate_direction[0] +
        preferred_direction[1] * candidate_direction[1]
    )
    cosine = max(-1.0, min(1.0, cosine))
    return cosine, 1.0 + float(weight) * cosine


def utility(gain_norm, path_m, travel_lambda,
            continuity=1.0, recent=1.0, visit=1.0):
    """Compute the utility U of a candidate (see the module docstring).

    Args:
        gain_norm: Normalized information gain; clipped to [0, 1].
        path_m: Obstacle-aware path length to the candidate [m]; negative
            values are treated as 0.
        travel_lambda: Travel-cost rate lambda [1/m].
        continuity: Directional continuity factor C.
        recent: Recent-goal factor R.
        visit: Unproductive-visit factor V.

    Returns:
        The non-negative utility U.
    """
    gain_norm = max(0.0, min(1.0, float(gain_norm)))
    path_m = max(0.0, float(path_m))
    # Bounded exponent: beyond ~700 the factor is negligible (< 1e-300).
    exponent = min(float(travel_lambda) * path_m, 700.0)
    return (
        gain_norm * math.exp(-exponent) *
        float(continuity) * float(recent) * float(visit)
    )
