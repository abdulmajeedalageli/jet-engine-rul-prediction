"""Load the three raw files, add labels, and check they fit together (notebook sections 1-4)."""
import pandas as pd

from src.config import CAP, COLS, RAW


def _read(name):
    return pd.read_csv(RAW / name, header=None, sep=r"\s+", names=COLS)


def _check_no_duplicates(d):
    assert d.groupby(["unit", "cycle"]).size().max() == 1, "duplicate (unit, cycle) rows"


def load_train():
    d = _read("train_FD001.txt")
    _check_no_duplicates(d)
    return d


def load_test():
    d = _read("test_FD001.txt")
    _check_no_duplicates(d)
    return d


def load_rul(test):
    """True RUL of each test engine at its last observed cycle."""
    rul = pd.read_csv(RAW / "RUL_FD001.txt", header=None, sep=r"\s+", names=["RUL"])
    # no engine column: line 1 is engine 1
    rul["unit"] = rul.index + 1
    assert len(rul) == 100, f"RUL file has {len(rul)} rows, expected 100"
    assert set(rul["unit"]) == set(test["unit"]), "RUL rows do not match the test engines"
    return rul


def add_labels(d):
    """Training labels. Only valid for engines that run to failure (the train set)."""
    d = d.copy()
    d["failure_cycle"] = d.groupby("unit")["cycle"].transform("max")
    d["RUL"] = d["failure_cycle"] - d["cycle"]
    d["cycles_left_capped"] = d["RUL"].clip(upper=CAP)
    assert (d.groupby("unit")["RUL"].min() == 0).all(), "a train engine does not end at RUL 0"
    return d
