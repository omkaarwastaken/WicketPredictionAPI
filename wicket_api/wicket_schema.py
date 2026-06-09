from pydantic import BaseModel


class WicketInput(BaseModel):

    phase: str

    runs_last_12: int
    dots_last_12: int
    wickets_last_12: int
    boundaries_last_12: int
    sixes_last_12: int

    current_run_rate: float
    required_run_rate: float

    over_number: int
    ball_number: int

    batter_rating: float
    batter_aggression: float
    batter_consistency: float
    batter_pressure_handling: float

    bowler_rating: float
    bowler_death_skill: float
    bowler_pressure_handling: float