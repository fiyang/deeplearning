"""Train-only categorical vocabulary, temporal splits, no outcome/statistic inputs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PROFILE = [f"onehot_feat{i}" for i in range(18)]
ITEM = ["author_id", "video_type", "upload_type", "music_id", "music_type",
        "duration_bucket", "width_bucket", "height_bucket"]
FIELDS = ["user_id", "video_id", "tab", "hour", "weekday"] + PROFILE + ITEM + ["video_age_bucket"]
LOG_COLUMNS = ["user_id", "video_id", "date", "hourmin", "time_ms", "is_click", "is_like", "tab"]
FILES = {"early": "log_standard_4_08_to_4_21_pure.csv",
         "test": "log_standard_4_22_to_5_08_pure.csv",
         "random": "log_random_4_22_to_5_08_pure.csv"}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    if value in ("", "UNKNOWN", "nan", None):
        return "<MISSING>"
    try:
        x = float(value)
        if not np.isfinite(x) or x < 0:
            return "<MISSING>"
        return str(int(x)) if x.is_integer() else str(x)
    except (TypeError, ValueError):
        return str(value)


def bucket(value, boundaries):
    try:
        x = float(value)
        return str(np.searchsorted(boundaries, x, side="right")) if np.isfinite(x) and x >= 0 else "<MISSING>"
    except (ValueError, TypeError):
        return "<MISSING>"


def load_log(path):
    with Path(path).open() as f:
        header = next(csv.reader(f))
    a = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64,
                   usecols=[header.index(k) for k in LOG_COLUMNS], ndmin=2)
    assert np.isin(a[:, 5:7], [0, 1]).all()
    return a


def split_standard(rows):
    # 2022-04-19 / 2022-04-22 midnight in Asia/Shanghai. Released date and
    # time_ms occasionally disagree near midnight, so use one clock throughout.
    cut_train, cut_test = 1650297600000, 1650556800000
    train = np.flatnonzero(rows[:, 4] < cut_train)
    validation = np.flatnonzero((rows[:, 4] >= cut_train) & (rows[:, 4] < cut_test))
    test = np.flatnonzero(rows[:, 4] >= cut_test)
    assert len(train) and len(validation) and len(test)
    assert rows[train, 4].max() < rows[validation, 4].min() < rows[test, 4].min()
    assert rows[validation, 4].max() < rows[test, 4].min()
    return train, validation, test


class FeatureBuilder:
    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        with (data_dir / "user_features_pure.csv").open() as f:
            self.users = {int(r["user_id"]): r for r in csv.DictReader(f)}
        with (data_dir / "video_features_basic_pure.csv").open() as f:
            self.items = {int(r["video_id"]): r for r in csv.DictReader(f)}
        for r in self.items.values():
            r["duration_bucket"] = bucket(r["video_duration"], [3000, 7000, 15000, 30000, 60000, 120000, 300000])
            for name in ["width", "height"]:
                r[f"{name}_bucket"] = bucket(r[f"server_{name}"], [240, 360, 480, 720, 1080, 1440, 2160])
        self.maps = {}; self.vocabulary_size = 1

    def small_values(self, rows, field):
        if field in PROFILE:
            return {str(k): canonical(v[field]) for k, v in self.users.items()}
        if field in ITEM:
            return {str(k): canonical(v[field]) for k, v in self.items.items()}
        raise KeyError(field)

    def derived(self, rows):
        # date is already a local calendar date; no inferred timezone conversion.
        days = np.array([str(d)[:4] + "-" + str(d)[4:6] + "-" + str(d)[6:] for d in np.unique(rows[:, 2])], dtype="datetime64[D]")
        date_map = {int(str(d).replace("-", "")): int(d.astype(np.int64)) for d in days}
        day = np.array([date_map[int(x)] for x in rows[:, 2]], dtype=np.int64)
        uploaded = {}
        for k, v in self.items.items():
            try:
                uploaded[k] = int(np.datetime64(v["upload_dt"], "D").astype(np.int64))
            except (TypeError, ValueError):
                uploaded[k] = -1
        upload_day = np.array([uploaded.get(int(x), -1) for x in rows[:, 1]])
        age = day - upload_day
        age_bucket = np.searchsorted([1, 3, 7, 14, 30, 90, 180, 365], age, side="right")
        age_bucket[(upload_day < 0) | (age < 0)] = -1
        return {"user_id": rows[:, 0], "video_id": rows[:, 1], "tab": rows[:, 7],
                "hour": rows[:, 3] // 100, "weekday": (day + 3) % 7, "video_age_bucket": age_bucket}

    def fit(self, rows):
        derived = self.derived(rows)
        train_users = set(rows[:, 0]); train_items = set(rows[:, 1])
        for field in FIELDS:
            if field in derived:
                values = {canonical(v) for v in np.unique(derived[field])}
            else:
                source = self.users if field in PROFILE else self.items
                ids = train_users if field in PROFILE else train_items
                values = {canonical(source.get(int(i), {}).get(field)) for i in ids}
            values.discard("<MISSING>")
            self.maps[field] = {v: j + self.vocabulary_size for j, v in enumerate(sorted(values))}
            self.vocabulary_size += len(values)
        tags = {canonical(t) for i in train_items for t in self.items.get(int(i), {}).get("tag", "").split(",")}
        tags.discard("<MISSING>")
        self.maps["tag"] = {v: j + self.vocabulary_size for j, v in enumerate(sorted(tags))}
        self.vocabulary_size += len(tags)
        # Preserve every released tag; no truncation or last-value-only encoding.
        self.max_tags = max(len(r["tag"].split(",")) for r in self.items.values())

    def transform(self, rows):
        derived = self.derived(rows)
        x = np.zeros((len(rows), len(FIELDS)), dtype=np.int32)
        user_unique, user_inverse = np.unique(rows[:, 0], return_inverse=True)
        item_unique, item_inverse = np.unique(rows[:, 1], return_inverse=True)
        for j, field in enumerate(FIELDS):
            mapping = self.maps[field]
            if field in derived:
                unique, inverse = np.unique(derived[field], return_inverse=True)
                values = np.array([mapping.get(canonical(v), 0) for v in unique], dtype=np.int32)
                x[:, j] = values[inverse]
            else:
                source = self.users if field in PROFILE else self.items
                unique, inverse = (user_unique, user_inverse) if field in PROFILE else (item_unique, item_inverse)
                values = np.array([mapping.get(canonical(source.get(int(i), {}).get(field)), 0) for i in unique], dtype=np.int32)
                x[:, j] = values[inverse]
        tag_table = np.zeros((max(self.items) + 1, self.max_tags), dtype=np.int32)
        for i, item in self.items.items():
            tags = item["tag"].split(",")
            tag_table[i, :len(tags)] = [self.maps["tag"].get(canonical(t), 0) for t in tags]
        assert (rows[:, 1] >= 0).all() and (rows[:, 1] < len(tag_table)).all()
        return x, tag_table[rows[:, 1]]

    def schema(self):
        return dict(fields=FIELDS, maps=self.maps, vocabulary_size=self.vocabulary_size,
                    max_tags=self.max_tags, tag_pooling="sum", unknown_id=0,
                    fit_on="training rows only", profile_snapshot_caveat="Anonymous user features have no as-of timestamp; strict temporal availability cannot be proven.",
                    excluded=["video_features_statistic_pure.csv", "all current feedback except labels", "play_time_ms", "profile_stay_time", "comment_stay_time", "user activity/fan/follow counts", "visible_status"])


def prepare(data_dir, output):
    data_dir = Path(data_dir); output = Path(output)
    standard = np.concatenate([load_log(data_dir / FILES[k]) for k in ["early", "test"]])
    train_idx, val_idx, test_idx = split_standard(standard)
    rows = {"train": standard[train_idx], "validation": standard[val_idx],
            "test": standard[test_idx], "random": load_log(data_dir / FILES["random"])}
    assert rows["validation"][:, 4].max() < rows["test"][:, 4].min()
    assert rows["validation"][:, 4].max() < rows["random"][:, 4].min()
    builder = FeatureBuilder(data_dir); builder.fit(rows["train"])
    features = {k: builder.transform(v) for k, v in rows.items()}
    labels = {k: v[:, 5:7].astype(np.float32) for k, v in rows.items()}
    np.savez_compressed(output / "split_indices.npz", train=train_idx, validation=val_idx, test=test_idx)
    schema = builder.schema(); write_json(output / "feature_schema.json", schema)
    counts = {}
    for k, v in rows.items():
        y, z = v[:, 5:7].T
        counts[k] = dict(rows=len(v), clicks=int(y.sum()), likes=int(z.sum()), joint_click_likes=int((y*z).sum()),
                         like_without_click=int(((1-y)*z).sum()), first_date=int(v[:, 2].min()), last_date=int(v[:, 2].max()),
                         users=len(np.unique(v[:, 0])), items=len(np.unique(v[:, 1])),
                         min_time_ms=int(v[:, 4].min()), max_time_ms=int(v[:, 4].max()),
                         unknown_user_rows=int((features[k][0][:, 0] == 0).sum()), unknown_item_rows=int((features[k][0][:, 1] == 0).sum()))
    files = list(FILES.values()) + ["user_features_pure.csv", "video_features_basic_pure.csv"]
    manifest = dict(counts=counts, sha256={n: sha256(data_dir / n) for n in files},
                    split_sha256=sha256(output / "split_indices.npz"),
                    source="https://zenodo.org/records/10439422", rows_dropped=0,
                    split_clock="time_ms; boundaries 2022-04-19 and 2022-04-22 midnight Asia/Shanghai; raw date occasionally disagrees near midnight",
                    split_index_source=[FILES["early"], FILES["test"]],
                    sampling="No downsampling, no positive oversampling; random-policy rows held out separately.",
                    labels=dict(ctr="is_click (click or valid play depending on UI)", cvr="P(is_like=1 | is_click=1, X)",
                                ctcvr="is_click * is_like; NOT unconditional like probability or purchase CVR"))
    write_json(output / "data_manifest.json", manifest)
    return rows, features, labels, schema, manifest
