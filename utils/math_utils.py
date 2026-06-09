
import math


def clamp_score(
    value: float,
    minimum: float = 0,
    maximum: float = 100
):
    """
    Restrict values inside a bounded range.
    """

    return max(minimum, min(value, maximum))


def normalize_distribution(values):
    """
    Convert percentages into probabilities.

    Example:
    [45, 35, 20]
    ->
    [0.45, 0.35, 0.20]
    """

    total = sum(values)

    if total <= 0:
        raise ValueError(
            "Distribution total must be positive."
        )

    return [v / total for v in values]


def shannon_entropy(probabilities):
    """
    Shannon entropy formula.

    H = -Σ(p * log2(p))
    """

    return -sum(
        p * math.log2(p)
        for p in probabilities
        if p > 0
    )


def normalize_entropy(
    entropy,
    max_entropy
):
    """
    Normalize entropy into 0–1 range.
    """

    if max_entropy == 0:
        return 0

    return entropy / max_entropy

