# ruff: noqa: D100, D101, D102, D103
import logging
import os
import pathlib
import time
from edgeimpulse import data
from edgeimpulse.data import Sample

# Settings
DATASET_PATH = os.path.join("sample_data", "gestures", "training")
TIMEOUT = 1200.0  # 20 min

# just have logging enabled for dev
logging.getLogger().setLevel(logging.INFO)

# Build dataset
current_dir = pathlib.Path(__file__).parent.resolve()
dataset = []
for filename in os.listdir(os.path.join(current_dir, DATASET_PATH)):
    if filename.endswith(".cbor"):
        dataset.append(
            Sample(
                filename=filename,
                data=open(os.path.join(current_dir, DATASET_PATH, filename), "rb"),
                category="training",
                label=filename.split(".")[0],
                metadata={"source": "accelerometer"},
            )
        )

# Upload dataset
timestamp = time.time()
resp = data.upload_samples(
    dataset,
    allow_duplicates=False,
    max_workers=4,
    show_progress=True,
    timeout_sec=TIMEOUT,
)
print(f"Uploaded {len(dataset)} samples in {time.time() - timestamp:.2f} seconds")

# Delete data

resp = data.delete_all_samples(timeout_sec=TIMEOUT)
if resp is None:
    logging.warning("Could not delete samples")
