"""
CSE 802 final project code.

This script reads the Serie A CSV files, turns each match into two team rows,
builds previous-game features, trains a few classifiers, and prints the tables
I use in the report.

Run:
    python soccer_momentum_project.py --data "data/*.csv" --out results
"""

import argparse
import glob
import os
from itertools import product
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


result_order = ["Win", "Loss", "Draw"]
required_columns = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
]

diff_columns = [
    "GoalDiff", "ShotDiff", "ShotTargetDiff", "CornerDiff", "FoulDiff"
]


def expand_files(patterns):
    files = []
    for pattern in patterns:
        # I use glob here so I can pass in data/*.csv and read all seasons at once.
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
    files = sorted(set(files))
    return files


def read_match_files(files):
    frames = []
    for path in files:
        df = pd.read_csv(path)

        df.columns = [str(c).strip() for c in df.columns]
        df = df[required_columns].copy()
        df["SourceFile"] = os.path.basename(path)
        frames.append(df)

    matches = pd.concat(frames, ignore_index=True)
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")

    for col in ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

    matches = matches.dropna(subset=required_columns)
    matches = matches.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    # I added MatchId so both team rows from the same match can still be connected.
    matches["MatchId"] = np.arange(len(matches))
    return matches


def team_view(matches):
    # Home team side of each match
    home = pd.DataFrame({
        "MatchId": matches["MatchId"],
        "Date": matches["Date"],
        "Team": matches["HomeTeam"],
        "Opponent": matches["AwayTeam"],
        "IsHome": 1,
        "GoalsFor": matches["FTHG"],
        "GoalsAgainst": matches["FTAG"],
        "ShotsFor": matches["HS"],
        "ShotsAgainst": matches["AS"],
        "ShotsTargetFor": matches["HST"],
        "ShotsTargetAgainst": matches["AST"],
        "FoulsFor": matches["HF"],
        "FoulsAgainst": matches["AF"],
        "CornersFor": matches["HC"],
        "CornersAgainst": matches["AC"],
        "Result": np.where(matches["FTR"].eq("H"), "Win", np.where(matches["FTR"].eq("A"), "Loss", "Draw")),
    })

    # Away team side of each match
    away = pd.DataFrame({
        "MatchId": matches["MatchId"],
        "Date": matches["Date"],
        "Team": matches["AwayTeam"],
        "Opponent": matches["HomeTeam"],
        "IsHome": 0,
        "GoalsFor": matches["FTAG"],
        "GoalsAgainst": matches["FTHG"],
        "ShotsFor": matches["AS"],
        "ShotsAgainst": matches["HS"],
        "ShotsTargetFor": matches["AST"],
        "ShotsTargetAgainst": matches["HST"],
        "FoulsFor": matches["AF"],
        "FoulsAgainst": matches["HF"],
        "CornersFor": matches["AC"],
        "CornersAgainst": matches["HC"],
        "Result": np.where(matches["FTR"].eq("A"), "Win", np.where(matches["FTR"].eq("H"), "Loss", "Draw")),
    })

    df = pd.concat([home, away], ignore_index=True)
    df = df.sort_values(["Team", "Date", "MatchId"]).reset_index(drop=True)
    return df


def add_match_metrics(df):
    df = df.copy()
    df["GoalDiff"] = df["GoalsFor"] - df["GoalsAgainst"]
    df["ShotDiff"] = df["ShotsFor"] - df["ShotsAgainst"]
    df["ShotTargetDiff"] = df["ShotsTargetFor"] - df["ShotsTargetAgainst"]
    df["CornerDiff"] = df["CornersFor"] - df["CornersAgainst"]
    df["FoulDiff"] = df["FoulsFor"] - df["FoulsAgainst"]

    # Simple score: goals matter most; extra fouls hurt the score.

    # I created this simple performance score so each match has one rough
    # number for how strongly a team played. Goals matter most, shots on
    # target matter more than total shots, corners help a little, and extra
    # fouls slightly hurt the score.
    df["PerformanceScore"] = (
        2.00 * df["GoalDiff"]
        + 0.08 * df["ShotDiff"]
        + 0.20 * df["ShotTargetDiff"]
        + 0.06 * df["CornerDiff"]
        - 0.05 * df["FoulDiff"]
    )

    df["WinFlag"] = (df["Result"] == "Win").astype(int)
    df["LossFlag"] = (df["Result"] == "Loss").astype(int)
    df["DrawFlag"] = (df["Result"] == "Draw").astype(int)

    #Big Wins and Losses split
    df["BigWinFlag"] = ((df["Result"] == "Win") & (df["GoalDiff"] >= 2)).astype(int)
    df["RegularWinFlag"] = ((df["Result"] == "Win") & (df["GoalDiff"] == 1)).astype(int)
    df["BigLossFlag"] = ((df["Result"] == "Loss") & (df["GoalDiff"] <= -2)).astype(int)
    df["RegularLossFlag"] = ((df["Result"] == "Loss") & (df["GoalDiff"] == -1)).astype(int)
    return df


def add_history_features(df, windows=(3, 5)):
    df = df.sort_values(["Team", "Date", "MatchId"]).copy()
    grouped = df.groupby("Team", group_keys=False)
    # I added this so I can remove early rows that do not have enough history.
    df["MatchesBefore"] = grouped.cumcount()

    history_cols = diff_columns + ["PerformanceScore"]
    flag_cols = [
        "WinFlag", "LossFlag", "DrawFlag",
        "BigWinFlag", "RegularWinFlag", "BigLossFlag", "RegularLossFlag",
    ]

    for w in windows:
        for col in history_cols:
            new_col = f"Prev{w}{col}Avg"
            # shift(1) makes sure the current match is not used to predict itself.
            df[new_col] = grouped[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )

        for col in flag_cols:
            clean_name = col.replace("Flag", "")
            new_col = f"Prev{w}{clean_name}Count"
            df[new_col] = grouped[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).sum()
            )

    history_feature_cols = [c for c in df.columns if c.startswith("Prev")]
    df[history_feature_cols] = df[history_feature_cols].fillna(0)
    return df


def feature_columns():
    cols = ["IsHome"]
    for w in (3, 5):
        for col in diff_columns + ["PerformanceScore"]:
            cols.append(f"Prev{w}{col}Avg")
        for col in ["Win", "Loss", "Draw", "BigWin", "RegularWin", "BigLoss", "RegularLoss"]:
            cols.append(f"Prev{w}{col}Count")
    return cols


def split_train_val_test(X, y, seed=42):
    # This gives about 70% training, 15% validation, and 15% testing.
    strat = y if y.value_counts().min() >= 2 else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=strat
    )

    strat_temp = y_temp if y_temp.value_counts().min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=seed, stratify=strat_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def param_grid(grid):
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def build_models(seed=42):
    return {
        "Logistic_standard": (
            Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed)),
            ]),
            {"model__C": [0.3, 1.0, 3.0]},
        ),
        "Logistic_minmax": (
            Pipeline([
                ("scale", MinMaxScaler()),
                ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed)),
            ]),
            {"model__C": [0.3, 1.0, 3.0]},
        ),
        "SVM_standard": (
            Pipeline([
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
            ]),
            {"model__C": [0.5, 1.0, 2.0], "model__gamma": ["scale"]},
        ),
        "RandomForest": (
            Pipeline([
                ("model", RandomForestClassifier(
                    n_estimators=150,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ))
            ]),
            {"model__max_depth": [None, 8, 14], "model__min_samples_leaf": [1, 3]},
        ),
    }


def tune_models(X_train, X_val, y_train, y_val, seed=42):
    rows = []
    best_name = None
    best_score = -1
    best_model = None
    best_params = None

    for name, (pipe, grid) in build_models(seed).items():
        for params in param_grid(grid):
            model = clone(pipe)
            model.set_params(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            acc = accuracy_score(y_val, pred)
            bal_acc = balanced_accuracy_score(y_val, pred)

            rows.append({
                "Model": name,
                "Params": str(params),
                "ValidationAccuracy": acc,
                "ValidationBalancedAccuracy": bal_acc,
            })

            # balanced accuracy handles class imbalance better than plain accuracy.
            if bal_acc > best_score:
                best_score = bal_acc
                best_name = name
                best_model = model
                best_params = params

    results = pd.DataFrame(rows).sort_values(
        ["ValidationBalancedAccuracy", "ValidationAccuracy"], ascending=False
    )
    return best_name, best_params, best_model, results


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    # After choosing the model on the validation set, I retrain it using both
    # the training and validation data before testing.
    X_final_train = pd.concat([X_train, X_val], axis=0)
    y_final_train = pd.concat([y_train, y_val], axis=0)

    final_model = clone(model)
    final_model.fit(X_final_train, y_final_train)
    pred = final_model.predict(X_test)

    cm = confusion_matrix(y_test, pred, labels=result_order)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{x}" for x in result_order],
                         columns=[f"Pred_{x}" for x in result_order])

    report = classification_report(y_test, pred, labels=result_order, zero_division=0)
    scores = {
        "TestAccuracy": accuracy_score(y_test, pred),
        "TestBalancedAccuracy": balanced_accuracy_score(y_test, pred),
    }
    return final_model, pred, cm_df, report, scores


def repeated_partition_accuracy(best_model, X, y, repeats=20, seed=42):
    # I repeat the split several times because one split might belucky
    rows = []
    for i in range(repeats):
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, seed + i)
        model = clone(best_model)
        model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        pred = model.predict(X_test)
        rows.append({
            "Repeat": i + 1,
            "Accuracy": accuracy_score(y_test, pred),
            "BalancedAccuracy": balanced_accuracy_score(y_test, pred),
        })

    out = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "Repeats": repeats,
        "MeanAccuracy": out["Accuracy"].mean(),
        "VarianceAccuracy": out["Accuracy"].var(ddof=1),
        "MeanBalancedAccuracy": out["BalancedAccuracy"].mean(),
        "VarianceBalancedAccuracy": out["BalancedAccuracy"].var(ddof=1),
    }])
    return out, summary


def bounce_back_tables(df):
    rows = []
    usable = df[df["MatchesBefore"] >= 3].copy()

    for w in (3, 5):
        groups = [
            (f"Big win in previous {w}", usable[f"Prev{w}BigWinCount"] > 0),
            (f"Regular win in previous {w}, no big win", (usable[f"Prev{w}RegularWinCount"] > 0) & (usable[f"Prev{w}BigWinCount"] == 0)),
            (f"Big loss in previous {w}", usable[f"Prev{w}BigLossCount"] > 0),
            (f"Regular loss in previous {w}, no big loss", (usable[f"Prev{w}RegularLossCount"] > 0) & (usable[f"Prev{w}BigLossCount"] == 0)),
        ]

        for label, mask in groups:
            part = usable[mask]
            rates = part["Result"].value_counts(normalize=True).reindex(result_order, fill_value=0)
            rows.append({
                "HistoryGroup": label,
                "Window": w,
                "Samples": len(part),
                "NextWinRate": rates["Win"],
                "NextLossRate": rates["Loss"],
                "NextDrawRate": rates["Draw"],
                "AvgNextGoalDiff": part["GoalDiff"].mean() if len(part) else np.nan,
                "AvgNextPerformanceScore": part["PerformanceScore"].mean() if len(part) else np.nan,
            })

    table = pd.DataFrame(rows)
    return table


def print_bounce_summary(table):
    print("\nBounce-back / momentum table")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    for w in (3, 5):
        t = table[table["Window"] == w].set_index("HistoryGroup")
        pairs = [
            (f"Big win in previous {w}", f"Regular win in previous {w}, no big win", "big win vs regular win"),
            (f"Big loss in previous {w}", f"Regular loss in previous {w}, no big loss", "big loss vs regular loss"),
        ]
        print(f"\nSimple gaps for previous {w} matches")
        for a, b, label in pairs:
            if a in t.index and b in t.index:
                win_gap = t.loc[a, "NextWinRate"] - t.loc[b, "NextWinRate"]
                perf_gap = t.loc[a, "AvgNextPerformanceScore"] - t.loc[b, "AvgNextPerformanceScore"]
                print(f"  {label}: WinRate gap={win_gap:.3f}, PerformanceScore gap={perf_gap:.3f}")


def save_feature_importance(model, X_test, y_test, out_dir, seed=42):
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=5,
        random_state=seed,
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "ImportanceMean": result.importances_mean,
        "ImportanceStd": result.importances_std,
    }).sort_values("ImportanceMean", ascending=False)

    path = os.path.join(out_dir, "top_features.csv")
    importance.to_csv(path, index=False)
    return importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    files = expand_files(args.data)
    matches = read_match_files(files)
    team_df = team_view(matches)
    team_df = add_match_metrics(team_df)
    team_df = add_history_features(team_df)

    feature_cols = feature_columns()
    model_df = team_df[team_df["MatchesBefore"] >= args.min_history].copy()

    X = model_df[feature_cols]
    y = model_df["Result"]

    print(f"Files read: {len(files)}")
    print(f"Matches read: {len(matches)}")
    print(f"Team-match rows used for modeling: {len(model_df)}")
    print("\nClass balance")
    balance = y.value_counts().reindex(result_order, fill_value=0)
    balance_pct = (balance / balance.sum()).round(3)
    print(pd.DataFrame({"Count": balance, "Percent": balance_pct}).to_string())

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, args.seed)
    best_name, best_params, best_model, val_results = tune_models(X_train, X_val, y_train, y_val, args.seed)

    final_model, pred, cm_df, report, test_scores = evaluate_model(
        best_model, X_train, X_val, X_test, y_train, y_val, y_test
    )

    print("\nValidation results")
    print(val_results.head(10).to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Best params: {best_params}")

    print("\nTest scores")
    for k, v in test_scores.items():
        print(f"{k}: {v:.3f}")

    print("\nConfusion matrix")
    print(cm_df.to_string())

    print("\nClassification report")
    print(report)

    repeat_scores, repeat_summary = repeated_partition_accuracy(
        best_model, X, y, repeats=args.repeats, seed=args.seed
    )
    print("\nRepeated partition accuracy")
    print(repeat_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    bounce_table = bounce_back_tables(team_df)
    print_bounce_summary(bounce_table)

    importance = save_feature_importance(final_model, X_test, y_test, args.out, args.seed)
    print("\nTop previous-performance features")
    print(importance.head(12).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Save outputs for the write-up.
    team_df.to_csv(os.path.join(args.out, "team_match_features.csv"), index=False)
    val_results.to_csv(os.path.join(args.out, "validation_results.csv"), index=False)
    cm_df.to_csv(os.path.join(args.out, "confusion_matrix.csv"))
    repeat_scores.to_csv(os.path.join(args.out, "repeated_partition_scores.csv"), index=False)
    repeat_summary.to_csv(os.path.join(args.out, "repeated_partition_summary.csv"), index=False)
    bounce_table.to_csv(os.path.join(args.out, "bounce_back_table.csv"), index=False)

    print(f"\nSaved result files to: {args.out}")


if __name__ == "__main__":
    main()
