import argparse
from pathlib import Path


def strip_frame_suffix(stem: str) -> str:
    if stem.endswith("_frame1"):
        return stem[: -len("_frame1")]
    if stem.endswith("_frame300"):
        return stem[: -len("_frame300")]
    return stem


def collect_by_prefix(folder: Path):
    mapping = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        prefix = strip_frame_suffix(p.stem)
        mapping[prefix] = p
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="从 ubc_test10 的 frame1/frame300/complex 生成 pairs.txt (ref,pose,complex)"
    )
    parser.add_argument(
        "--data-root",
        default="/home/yanghaotian/server_data/yanghaotian/data/ubc_test10",
        help="包含 frame1/frame300/complex 三个子目录的根目录",
    )
    parser.add_argument(
        "--output",
        default="/home/yanghaotian/server_data/yanghaotian/data/ubc_frames/pairs.txt",
        help="输出 txt 路径，每行格式: ref,pose,complex",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    frame1_dir = data_root / "frame1"
    frame300_dir = data_root / "frame300"
    complex_dir = data_root / "complex"

    if not frame1_dir.exists() or not frame300_dir.exists() or not complex_dir.exists():
        raise FileNotFoundError(
            f"目录不存在，请检查: {frame1_dir}, {frame300_dir}, {complex_dir}"
        )

    ref_map = collect_by_prefix(frame1_dir)
    pose_map = collect_by_prefix(frame300_dir)
    complex_map = collect_by_prefix(complex_dir)

    common_prefixes = sorted(set(ref_map) & set(pose_map) & set(complex_map))
    missing_ref = sorted((set(pose_map) & set(complex_map)) - set(ref_map))
    missing_pose = sorted((set(ref_map) & set(complex_map)) - set(pose_map))
    missing_complex = sorted((set(ref_map) & set(pose_map)) - set(complex_map))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for key in common_prefixes:
            f.write(f"{ref_map[key]},{pose_map[key]},{complex_map[key]}\n")

    print(f"已写入 {len(common_prefixes)} 条到: {output_path}")
    if missing_ref:
        print(f"缺失 frame1(ref) 的前缀数量: {len(missing_ref)}")
    if missing_pose:
        print(f"缺失 frame300(pose) 的前缀数量: {len(missing_pose)}")
    if missing_complex:
        print(f"缺失 complex 的前缀数量: {len(missing_complex)}")


if __name__ == "__main__":
    main()
