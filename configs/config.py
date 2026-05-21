import numpy as np

class Config:
	class data:
		radius = 5.0
		k_neighbors = 32
		dt = 0.001

	class model:
		hidden_dim = 32
		num_layers = 2
		dropout = 0.1

	class training:
		batch_size = 1
		lr = 1e-3
		epochs = 200

	class system:
		num_classes = 3

	class vec:
		r_max = 5
		M = 10
		SIGMA_SCALES = np.linspace(0.5, r_max, M)
		SUBSAMPLE = 5
		DT = 1e-3
		MAX_STEP = 200
		TRAIN_SEEDS = range(0, 70)
		VAL_SEEDS = range(70, 85)
		TEST_SEEDS = range(85, 100)
		DATA_DIR = "data"

cfg = Config