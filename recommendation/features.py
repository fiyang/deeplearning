"""MovieLens metadata and strictly earlier, training-only rating histories."""
from pathlib import Path
import re

import numpy as np

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
AGES = [1, 18, 25, 35, 45, 50, 56]


def metadata(data_dir, user_ids, item_ids):
    user_map = {int(v): n + 1 for n, v in enumerate(user_ids)}
    item_map = {int(v): n + 1 for n, v in enumerate(item_ids)}
    user_categorical = np.zeros((len(user_ids) + 1, 3), dtype=np.int64)
    item_features = np.zeros((len(item_ids) + 1, len(GENRES) + 1), dtype=np.float32)
    years = np.zeros(len(item_ids) + 1, dtype=np.float32)
    for line in (Path(data_dir) / "users.dat").read_text().splitlines():
        uid, gender, age, occupation, _ = line.split("::")
        if int(uid) in user_map:
            user_categorical[user_map[int(uid)]] = [1 if gender == "F" else 2,
                                                   AGES.index(int(age)) + 1, int(occupation) + 1]
    for line in (Path(data_dir) / "movies.dat").read_text(encoding="latin-1").splitlines():
        iid, title, genres = line.split("::")
        if int(iid) in item_map:
            row = item_map[int(iid)]
            for genre in genres.split("|"):
                item_features[row, GENRES.index(genre)] = 1
            year = re.search(r"\((\d{4})\)\s*$", title)
            if year is None:
                raise ValueError(f"Missing release year for movie {iid}")
            years[row] = int(year.group(1))
    assert (user_categorical[1:] > 0).all()
    assert (item_features[1:, :len(GENRES)].sum(axis=1) > 0).all()
    year_mean, year_std = float(years[1:].mean()), float(years[1:].std())
    item_features[1:, -1] = (years[1:] - year_mean) / max(year_std, 1.0)
    schema = {"user_categorical_columns": ["gender", "age_bucket", "occupation"],
              "gender": {"F": 1, "M": 2}, "age_buckets": AGES, "occupation": "original code + 1",
              "genres": GENRES, "movie_features": "18 multi-hot genres + standardized release year",
              "release_year_mean": year_mean, "release_year_std": year_std,
              "normalization_fit": "unique training-vocabulary movies only",
              "excluded_metadata": ["zip_code", "title_text"],
              "history_numeric_columns": ["log1p_user_past_count_div_10", "user_smoothed_high_rating_rate",
                                          "log1p_item_past_count_div_10", "item_smoothed_high_rating_rate"]
                                         + [f"user_past_liked_genre_{genre}" for genre in GENRES]}
    return user_categorical, item_features, schema


def pair_history(state, users, items):
    """Use frozen per-user/item state; this function never reads target labels."""
    numeric = np.column_stack([state["user_numeric"][users, :2], state["item_numeric"][items],
                               state["user_numeric"][users, 2:]]).astype(np.float32)
    return state["recent_items"][users], numeric


def build_history(rows, users, items, item_genres, user_count, item_count, history_length=20):
    """Build train features before updates at each timestamp, then freeze end state.

    Inputs must contain TRAINING ROWS ONLY. All rows with the same timestamp
    observe the state before that timestamp. Positive-history ties use movie ID
    order only after scoring all tied events; no current rating is a feature.
    """
    assert len(rows) == len(users) == len(items) and len(rows) > 0
    assert np.all(users > 0) and np.all(items > 0)
    histories = np.zeros((len(rows), history_length), dtype=np.int32)
    numeric = np.empty((len(rows), 4 + len(GENRES)), dtype=np.float32)
    user_count_past = np.zeros(user_count + 1, dtype=np.float32)
    user_positive = np.zeros_like(user_count_past)
    item_count_past = np.zeros(item_count + 1, dtype=np.float32)
    item_positive = np.zeros_like(item_count_past)
    genre_sum = np.zeros((user_count + 1, len(GENRES)), dtype=np.float32)
    recent = np.zeros((user_count + 1, history_length), dtype=np.int32)
    # Primary sort time, then user ID and movie ID, independent of input row order.
    order = np.lexsort((rows[:, 1], rows[:, 0], rows[:, 3]))
    boundaries = np.r_[0, np.flatnonzero(np.diff(rows[order, 3])) + 1, len(rows)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        indices = order[start:end]
        u, i = users[indices], items[indices]
        histories[indices] = recent[u]
        numeric[indices, 0] = np.log1p(user_count_past[u]) / 10
        numeric[indices, 1] = (user_positive[u] + 1) / (user_count_past[u] + 2)
        numeric[indices, 2] = np.log1p(item_count_past[i]) / 10
        numeric[indices, 3] = (item_positive[i] + 1) / (item_count_past[i] + 2)
        numeric[indices, 4:] = genre_sum[u] / np.maximum(user_positive[u, None], 1)
        positive = rows[indices, 2] >= 4
        np.add.at(user_count_past, u, 1)
        np.add.at(item_count_past, i, 1)
        np.add.at(user_positive, u, positive)
        np.add.at(item_positive, i, positive)
        np.add.at(genre_sum, u[positive], item_genres[i[positive]])
        for user, item in zip(u[positive], i[positive]):
            recent[user, 1:] = recent[user, :-1].copy()
            recent[user, 0] = item
    state = {"recent_items": recent,
             "user_numeric": np.column_stack([np.log1p(user_count_past) / 10,
                                               (user_positive + 1) / (user_count_past + 2),
                                               genre_sum / np.maximum(user_positive[:, None], 1)]).astype(np.float32),
             "item_numeric": np.column_stack([np.log1p(item_count_past) / 10,
                                               (item_positive + 1) / (item_count_past + 2)]).astype(np.float32)}
    assert np.isfinite(numeric).all()
    return histories, numeric, state
