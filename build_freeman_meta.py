import argparse
import json
import os
import re
from collections import defaultdict


KINDS = ["ref", "tgt", "complement", "kps2d", "kps3d"]


def build_meta(input_dir: str, output_json: str, missing_json: str, ext: str | None) -> tuple[int, int]:
    file_pattern = re.compile(r"^(.*)_(tgt|ref|complement|kps2d|kps3d)_(c0[1-8])\.([^.]+)$")
    records: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)

    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        matched = file_pattern.match(name)
        if not matched:
            continue
        prefix, kind, camera, extension = matched.groups()
        if ext and extension.lower() != ext.lower():
            continue
        records[(prefix, camera)][kind] = path

    complete_items = []
    missing_items = []

    for (prefix, camera), item in records.items():
        missing = [kind for kind in KINDS if kind not in item]
        if missing:
            missing_items.append(
                {
                    "prefix": prefix,
                    "camera": camera,
                    "missing": missing,
                    "existing": {kind: item[kind] for kind in KINDS if kind in item},
                }
            )
            continue
        complete_items.append(
            {
                "ref_img": item["ref"],
                "tgt_img": item["tgt"],
                "complement": item["complement"],
                "kps2d": item["kps2d"],
                "kps3d": item["kps3d"],
            }
        )

    complete_items.sort(key=lambda x: x["tgt_img"])
    missing_items.sort(key=lambda x: (x["prefix"], x["camera"]))

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(complete_items, f, ensure_ascii=False, indent=2)
    with open(missing_json, "w", encoding="utf-8") as f:
        json.dump(missing_items, f, ensure_ascii=False, indent=2)

    return len(complete_items), len(missing_items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="/home/yanghaotian/server_data/yanghaotian/data/FreeMan_frames",
    )
    parser.add_argument(
        "--output-json",
        default="/home/yanghaotian/server_data/yanghaotian/data/FreeMan_frames_meta.json",
    )
    parser.add_argument(
        "--missing-json",
        default="/home/yanghaotian/server_data/yanghaotian/data/FreeMan_frames_missing.json",
    )
    parser.add_argument("--ext", default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.missing_json), exist_ok=True)

    complete_count, missing_count = build_meta(
        input_dir=args.input_dir,
        output_json=args.output_json,
        missing_json=args.missing_json,
        ext=args.ext,
    )
    print(f"saved: {args.output_json}")
    print(f"saved: {args.missing_json}")
    print(f"complete_items: {complete_count}")
    print(f"missing_items: {missing_count}")


if __name__ == "__main__":
    main()
