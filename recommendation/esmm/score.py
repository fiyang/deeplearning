"""Score one impression from pre-feedback fields; no label/play time required."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import FeatureBuilder, sha256
from model import FunnelModel


def restore(checkpoint, data_dir):
    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    schema_path = checkpoint.parent.parent / "feature_schema.json"
    assert sha256(schema_path) == ckpt["config"]["feature_schema_sha256"], "Wrong feature schema"
    for name in ["user_features_pure.csv", "video_features_basic_pure.csv"]:
        assert sha256(Path(data_dir)/name) == ckpt["config"]["data_sha256"][name], "Metadata changed"
    schema = json.loads(schema_path.read_text())
    builder = FeatureBuilder(data_dir)
    builder.maps = schema["maps"]; builder.max_tags = schema["max_tags"]
    builder.vocabulary_size = schema["vocabulary_size"]
    model = FunnelModel(ckpt["config"]); model.load_state_dict(ckpt["model_state"]); model.eval()
    return model, builder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent/"data/kuairand/KuaiRand-Pure/data")
    p.add_argument("--user-id", type=int, required=True)
    p.add_argument("--video-id", type=int, required=True)
    p.add_argument("--date", type=int, required=True, help="YYYYMMDD, local calendar date")
    p.add_argument("--hourmin", type=int, required=True)
    p.add_argument("--tab", type=int, required=True)
    args = p.parse_args(); torch.set_num_threads(2)
    if not (0 <= args.hourmin//100 < 24 and 0 <= args.hourmin % 100 < 60 and 0 <= args.tab <= 14):
        p.error("Invalid hourmin or tab")
    model, builder = restore(args.checkpoint, args.data_dir)
    if args.video_id not in builder.items:
        p.error("Video metadata is missing; add metadata explicitly before scoring new videos")
    row = np.array([[args.user_id, args.video_id, args.date, args.hourmin, 0, 0, 0, args.tab]], dtype=np.int64)
    x = tuple(torch.as_tensor(a, dtype=torch.long) for a in builder.transform(row))
    with torch.inference_mode():
        ctr, cvr, joint = model.probabilities(*x)[0].tolist()
    print(json.dumps(dict(user_id=args.user_id, video_id=args.video_id, p_click_or_valid_play=ctr,
                         p_like_given_click=cvr, p_click_and_like=joint,
                         unknown_user=bool(x[0][0, 0] == 0), unknown_video=bool(x[0][0, 1] == 0)), indent=2))


if __name__ == "__main__":
    main()
