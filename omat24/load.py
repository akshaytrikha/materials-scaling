from fairchem.core.datasets import AseDBDataset

dataset_path = "rattled-1000-subsampled"
config_kwargs = {}  # see tutorial on additional configuration

dataset = AseDBDataset(config=dict(src=dataset_path, **config_kwargs))

# atoms objects can be retrieved by index
atoms = dataset.get_atoms(0)
