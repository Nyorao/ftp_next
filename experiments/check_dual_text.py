"""Real 224px, 8-image, three-update correctness smoke for all eight dual arms."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.check_encoder_text import run
from experiments.run_dual_text import ARMS, arm_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/v3/context_efficiency_reference.yaml")
    parser.add_argument("--bank",type=Path,default=ROOT/"outputs/hierarchical_text_assets_20261108/text.pt")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:2")
    args = parser.parse_args()
    run(args.config,args.bank,args.output.resolve(),args.device,arms=ARMS,make_config=arm_config)


if __name__ == "__main__":
    main()
