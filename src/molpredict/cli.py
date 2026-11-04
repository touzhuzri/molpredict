"""命令行入口。"""

import argparse
import sys
from pathlib import Path

from molpredict import __version__
from molpredict.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs
from molpredict.modeling import predict_smiles, train
from molpredict.molecules import process_csv


def check_environment() -> int:
    """检查 Python 版本和核心依赖是否可用，返回退出码。"""
    if sys.version_info < (3, 10):
        print(f"[FAIL] 需要 Python >= 3.10，当前为 {sys.version.split()[0]}")
        return 1
    print(f"[OK] Python {sys.version.split()[0]}")

    ok = True
    for package in ("rdkit", "numpy", "sklearn", "catboost"):
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {package} {version}")
        except ImportError:
            print(f"[FAIL] 未安装 {package}")
            ok = False
    for package in ("torch", "transformers", "padelpy"):
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {package} {version} (V3)")
        except ImportError:
            print(f"[WARN] 未安装 {package}；V3 训练/预测不可用")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    # Windows 下管道/Git Bash 默认使用 GBK，统一按 UTF-8 输出，避免中文乱码
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="molpredict",
        description="轻量级分子性质预测工具",
    )
    parser.add_argument("--version", action="version", version=f"molpredict {__version__}")
    parser.add_argument("--check", action="store_true", help="检查环境与依赖")

    subparsers = parser.add_subparsers(dest="command")
    process = subparsers.add_parser("process", help="批量处理分子 CSV（至少含 name、smiles 列）")
    process.add_argument("csv_path", type=Path, help="输入 CSV 路径")
    process.add_argument("--out", type=Path, default=PROCESSED_DIR, help="输出目录")

    train_cmd = subparsers.add_parser("train", help="训练杀菌活性分类模型")
    train_cmd.add_argument("csv_path", type=Path, help="带 label 列的 CSV（1=杀菌剂，0=非杀菌剂）")
    train_cmd.add_argument("--out", type=Path, default=MODELS_DIR, help="模型输出目录")
    train_cmd.add_argument(
        "--features", default="morgan,descriptors,combined",
        help="逗号分隔: morgan,descriptors,combined,confseq,all",
    )
    train_cmd.add_argument("--embeddings", type=Path, help="ConfSeq embedding .npy（与 CSV 行序一致）")
    train_cmd.add_argument("--seed", type=int, default=42)

    predict_cmd = subparsers.add_parser("predict", help="预测单个 SMILES 的杀菌活性概率")
    predict_cmd.add_argument("--smiles", required=True)
    predict_cmd.add_argument("--model", type=Path, required=True)
    predict_cmd.add_argument(
        "--feature", required=True, choices=["morgan", "descriptors", "combined"]
    )
    predict_cmd.add_argument("--threshold", type=float, default=0.5)

    v3_predict = subparsers.add_parser(
        "v3-predict",
        help="用 V3 统一多模态模型批量预测（需要 smiles、confseq、24 项 PaDEL）",
    )
    v3_predict.add_argument("--model", type=Path, required=True,
                            help="V3 run 目录（含 config.json/model.pt）")
    v3_predict.add_argument("--input", type=Path, required=True,
                            help="CSV：name、smiles、confseq 和 24 个描述符列")
    v3_predict.add_argument("--out", type=Path, required=True)
    v3_predict.add_argument("--threshold", type=float, default=None,
                            help="覆盖 checkpoint 中的验证集阈值")

    args = parser.parse_args(argv)

    if args.check:
        return check_environment()

    if args.command == "process":
        if not args.csv_path.is_file():
            print(f"[FAIL] 找不到输入文件: {args.csv_path}")
            return 1
        ensure_dirs()
        try:
            summary = process_csv(args.csv_path, args.out)
        except Exception as exc:
            print(f"[FAIL] 处理失败: {exc}")
            return 1
        print(
            f"\n处理完成: 共 {summary['total']}，"
            f"成功 {summary['success']}，失败 {summary['failed']}"
        )
        print(f"成功记录: {summary['out_dir'] / 'molecules.csv'}")
        print(f"失败记录: {summary['out_dir'] / 'failed.csv'}")
        print(f"2D 结构 : {summary['out_dir'] / 'structures_2d'}")
        print(f"3D 结构 : {summary['sdf_path']}")
        return 0 if summary["failed"] == 0 else 1

    if args.command == "train":
        if not args.csv_path.is_file():
            print(f"[FAIL] 找不到输入文件: {args.csv_path}")
            return 1
        features = [f.strip() for f in args.features.split(",") if f.strip()]
        try:
            results = train(
                args.csv_path, args.out, features,
                seed=args.seed, embeddings_path=args.embeddings,
            )
        except Exception as exc:
            print(f"[FAIL] 训练失败: {exc}")
            return 1
        print(f"\n模型与指标已保存: {args.out}")
        return 0

    if args.command == "predict":
        try:
            result = predict_smiles(
                args.smiles, args.model, args.feature, threshold=args.threshold
            )
        except Exception as exc:
            print(f"[FAIL] 预测失败: {exc}")
            return 1
        print(f"SMILES   : {result['smiles']}")
        print(f"InChIKey : {result['inchikey']}")
        print(f"杀菌活性概率: {result['probability']:.4f}")
        print(f"判断     : {result['prediction']}（阈值 {result['threshold']}）")
        print("注意: 该分数是模型预测值，不是实验活性概率。")
        return 0

    if args.command == "v3-predict":
        if not args.input.is_file():
            print(f"[FAIL] 找不到输入文件: {args.input}")
            return 1
        if not (args.model / "config.json").is_file():
            print(f"[FAIL] 不是 V3 模型目录: {args.model}")
            return 1
        try:
            from molpredict.v3_predict import predict_csv

            result = predict_csv(
                args.model, args.input, args.out, threshold=args.threshold
            )
        except Exception as exc:
            print(f"[FAIL] V3 预测失败: {exc}")
            return 1
        print(f"预测完成: {len(result)} 条 -> {args.out}")
        print("注意: 输出是 PPDB 杀菌剂分类模型的分数，不是实验抗真菌活性概率。")
        return 0

    ensure_dirs()
    print(f"molpredict {__version__}")
    print("项目目录:")
    print(f"  原始数据   {RAW_DIR}")
    print(f"  处理后数据 {PROCESSED_DIR}")
    print(f"  模型文件   {MODELS_DIR}")
    print("可用命令: molpredict process <csv>；--check 检查依赖。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
