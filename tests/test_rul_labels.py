import pandas as pd

from src.rul_labels import add_test_rul, add_train_rul, last_cycle_per_engine


def _toy_train_df():
    # Two engines: engine 1 runs 5 cycles, engine 2 runs 3 cycles.
    rows = []
    for cycle in range(1, 6):
        rows.append({"unit_number": 1, "time_in_cycles": cycle})
    for cycle in range(1, 4):
        rows.append({"unit_number": 2, "time_in_cycles": cycle})
    return pd.DataFrame(rows)


def test_train_rul_counts_down_to_zero_at_failure():
    df = add_train_rul(_toy_train_df(), clip=None)
    engine1 = df[df.unit_number == 1].sort_values("time_in_cycles")
    assert list(engine1["rul"]) == [4, 3, 2, 1, 0]


def test_train_rul_clipping():
    df = add_train_rul(_toy_train_df(), clip=2)
    engine1 = df[df.unit_number == 1].sort_values("time_in_cycles")
    assert list(engine1["rul"]) == [2, 2, 2, 1, 0]


def test_test_rul_accounts_for_truncation():
    df = _toy_train_df()  # reuse as a "truncated" test set
    true_rul = pd.Series({1: 10, 2: 20})
    out = add_test_rul(df, true_rul, clip=None)
    engine1 = out[out.unit_number == 1].sort_values("time_in_cycles")
    # Engine 1 truncated at cycle 5 with true_rul=10 -> RUL at cycle 5 is 10,
    # and counts up going backwards: cycle 4 -> 11, cycle 1 -> 14.
    assert list(engine1["rul"]) == [14, 13, 12, 11, 10]


def test_last_cycle_per_engine_picks_max_cycle():
    df = _toy_train_df()
    latest = last_cycle_per_engine(df)
    assert set(latest["time_in_cycles"]) == {5, 3}
    assert len(latest) == 2
