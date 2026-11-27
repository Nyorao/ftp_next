"""Real-size paired-center short correctness check for MT, N0 and N7."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.check_encoder_text import run
from experiments.run_dual_text_long import ARMS, arm_config, verify_pairing
from freqssl.config import load_config
from freqssl.engine import write_json
from experiments.train_long_v3 import resolve_long_config
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/v3/context_efficiency_reference.yaml')
    parser.add_argument('--bank',type=Path,default=ROOT/'outputs/hierarchical_text_assets_20261108/text.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:2')
    args=parser.parse_args(); base=load_config(args.config); torch.set_num_threads(base.cpu_threads)
    resolved,_,_=resolve_long_config(base,args.output,(100,200),5)
    configs={a:arm_config(resolved,a,args.bank,args.output/a) for a in ARMS}
    pairing=verify_pairing(configs)
    def make_config(base,arm,bank,output):
        return replace(configs[arm],output_dir=str(output)).validate()
    run(args.config,args.bank,args.output.resolve(),args.device,arms=ARMS,make_config=make_config,center_pairing=True)
    write_json(args.output/'pairing.json',pairing)


if __name__=='__main__': main()
