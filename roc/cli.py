# roc/cli.py

"""
single entrypoint
"""

import argparse
from roc.data.extract import cmd_extract
from roc.data.index_frames import cmd_index
from roc.data.split import cmd_split
from roc.train.train import cmd_train
from roc.export.torchscript import cmd_export
from roc.infer.batch import cmd_infer
from roc.data.label_cli import cmd_label  # we’ll adapt your label tool as a callable


def build_parser():
    p = argparse.ArgumentParser(
        prog="roc", description="Road Object Classification CLI"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # extract
    pe = sub.add_parser(
        "extract", help="Extract frames from day/night videos (resume-safe)"
    )
    pe.add_argument("--video-root", required=True)
    pe.add_argument("--output-root", default="raw_frames")
    pe.add_argument(
        "--lighting", choices=["daytime", "nighttime", "all"], default="all"
    )
    pe.add_argument("--interval", type=int, default=30)
    pe.set_defaults(func=cmd_extract)

    # index -> labels.csv (unlabeled)
    pi = sub.add_parser("index", help="Index frames to labels.csv")
    pi.add_argument("--day", required=False)
    pi.add_argument("--night", required=False)
    pi.add_argument("--out", default="labels.csv")
    pi.set_defaults(func=cmd_index)

    # split (csv)
    ps = sub.add_parser("split", help="Build dataset/train|val from labels.csv")
    ps.add_argument("--labels", required=True)
    ps.add_argument("--raw-frames", required=True)
    ps.add_argument("--dest", default="dataset")
    ps.add_argument("--train-ratio", type=float, default=0.8)
    ps.add_argument("--hardlinks", action="store_true")
    ps.set_defaults(func=cmd_split)

    # label (interactive)
    pl = sub.add_parser("label", help="Interactive labeling UI (1..5 keys)")
    pl.add_argument("--csv", required=True)
    pl.add_argument("--autosave-every", type=int, default=1)
    pl.set_defaults(func=cmd_label)

    # train
    pt = sub.add_parser("train", help="Train MobileNetV3 Small")
    pt.add_argument("--data", default="dataset")
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--out", default="models/mobilenetv3_road5.pt")
    pt.set_defaults(func=cmd_train)

    # export
    px = sub.add_parser("export", help="Export TorchScript for mobile")
    px.add_argument("--weights", required=True)
    px.add_argument("--out", default="mobile/mobilenetv3_road5_torchscript.pt")
    px.add_argument("--num-classes", type=int, default=5)
    px.set_defaults(func=cmd_export)

    # infer (desktop sanity check on folder)
    pf = sub.add_parser("infer", help="Batch infer on a folder of images")
    pf.add_argument("--model", required=True)
    pf.add_argument("--images", required=True)
    pf.set_defaults(func=cmd_infer)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
