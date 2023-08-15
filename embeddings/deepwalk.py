import subprocess
import numpy as np
from pathlib import Path


def convert_to_numpy_binary(args, data):
    flag = 1
    with open(data.U_fname) as fp:
        for line in fp:
            line = line.strip().split()
            if flag:
                flag = 0
                n, d = int(line[0]), int(line[1])
                U = np.zeros((n,d))
                continue
            node_id = int(line[0])
            emb_vec = [float(x) for x in line[1:]]
            U[node_id,:] = emb_vec
    return U


def create_DeepWalk_Embedding(args, data, A_fname):

	if Path(data.U_fname).exists() and not args.recompute_embedding:
		print("Deepwalk embedding exists. Loading...")
		U = np.load(data.U_fname)
	else:
		# execute deepwalk generation
		print("Creating DeepWalk Embedding...")
		exec_str = f"./embeddings/snap/examples/node2vec/node2vec -i:{A_fname} -o:{data.U_fname} -d:{args.dim}"
		print(exec_str)
		_ = subprocess.run(exec_str, shell=True)
		U = convert_to_numpy_binary(args, data)
		np.save(data.U_fname, U)
	return U