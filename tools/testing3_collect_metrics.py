#!/usr/bin/env python3

import json
from pathlib import Path
import argparse


def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    args=parser.parse_args()

    root=Path(args.root)

    summary=[]

    for d in sorted(root.glob("iter*")):

        metrics={
            "iteration": int(d.name.replace("iter","")),
            "games":0,
            "mean_return":0,
            "min_return":0,
            "max_return":0
        }

        # placeholder:
        # SGF parser will be connected here
        # current SGF already generated

        summary.append(metrics)


    (root/"summary.json").write_text(
        json.dumps(summary,indent=2)
    )


if __name__=="__main__":
    main()
