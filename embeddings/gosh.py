import torch
import subprocess
import numpy as np
from pathlib import Path


def create_GOSH_Embedding(args, data, A_fname):

	if Path(data.U_fname).exists() and not args.recompute_embedding:
		print("GOSH embedding exists. Loading...")
		U = np.load(data.U_fname)
	else:
		# GOSH returns a garbage embedding with no warning
		if not torch.cuda.is_available():
				raise ValueError("GOSH requires cuda.")

		# execute GOSH generation, saves embedding directly in np binary format
		print("Creating GOSH Embedding...")
		exec_str = f"./embeddings/GOSH/execs/gosh.out --input-graph {A_fname} --output-embedding {data.U_fname} --directed 0 --epochs {args.gosh_epochs} -d {args.dim} -s {args.gosh_s} --negative-weight {args.gosh_neg_wt} --binary-output --sampling-algorithm 0 -a 0 -l {args.gosh_lr} --learning-rate-decay-strategy 0 --coarsening-stopping-threshold 1000 --coarsening-stopping-precision 0.8 --coarsening-matching-threshold-ratio 400 --coarsening-min-vertices-in-graph 100 --epoch-strategy s-fast --smoothing-ratio 0.5"
  
		print(exec_str)
		_ = subprocess.run(exec_str, shell=True)
		U = np.load(data.U_fname)

	return U