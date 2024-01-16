import logging
import os
import pathlib
import time

from edgeimpulse.data.sample_type import (
    Sample,
)
from edgeimpulse.data._functions.upload import (
    upload_samples,
)
from edgeimpulse.data._functions.delete import (
    delete_all_samples,
)

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
resp = upload_samples(
    dataset,
    allow_duplicates=False,
    max_workers=4,
    show_progress=True,
    timeout_sec=TIMEOUT,
)
print(f"Uploaded {len(dataset)} samples in {time.time() - timestamp:.2f} seconds")

# Delete data
resp = delete_all_samples(timeout_sec=TIMEOUT)
if resp is None:
    logging.warning("Could not delete samples")
