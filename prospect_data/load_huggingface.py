
from datasets import load_dataset

train_dataset = load_dataset("Wilhelmlab/prospect-ptms-ms2", split="train")
val_dataset = load_dataset("Wilhelmlab/prospect-ptms-ms2", split="val")
test_dataset  = load_dataset("Wilhelmlab/prospect-ptms-ms2", split="test")

train_dataset.to_parquet("prospect_data/train_dataset.parquet")
val_dataset.to_parquet("prospect_data/val_dataset.parquet")
test_dataset.to_parquet("prospect_data/test_dataset.parquet")

from dlomix.data import FragmentIonIntensityDataset

TRAIN_DATAPATH = "prospect_data/train_dataset.parquet"
VAL_DATAPATH = "prospect_data/val_dataset.parquet"
TEST_DATAPATH = "prospect_data/test_dataset.parquet"

# For PTMs
BATCH_SIZE = 8
from dlomix.constants import PTMS_ALPHABET

ptm_processed = FragmentIonIntensityDataset(
    #data_source=TRAIN_DATAPATH,
    data_source=TRAIN_DATAPATH,
    val_data_source=VAL_DATAPATH,
    test_data_source=TEST_DATAPATH,
    max_seq_len=32,
    batch_size=BATCH_SIZE,
    val_ratio=0.2,
    model_features=["collision_energy_aligned_normed", "precursor_charge_onehot"],
    sequence_column="modified_sequence",
    label_column="intensities_raw",
    features_to_extract=["mod_loss", "delta_mass"],
    dataset_type="pt",
    with_termini=True,
    encoding_scheme="naive-mods", # Was missing in original code
    alphabet=PTMS_ALPHABET, # Was missing in original code
)

ptm_processed.save_to_disk(path="prospect_data/pre_processed")

'''
# For unmodified sequences
TRAIN_DATAPATH = "prospect_data/train_dataset.parquet"
VAL_DATAPATH = "prospect_data/val_dataset.parquet"
TEST_DATAPATH = "prospect_data/test_dataset.parquet"

BATCH_SIZE = 128

d = FragmentIonIntensityDataset(
    data_source=TRAIN_DATAPATH,
    val_data_source=VAL_DATAPATH,
    test_data_source=TEST_DATAPATH,
    max_seq_len=30,
    batch_size=BATCH_SIZE,
    val_ratio=0.2,
    model_features=["collision_energy_aligned_normed", "precursor_charge_onehot"],
    sequence_column="modified_sequence",
    label_column="intensities_raw",
    # features_to_extract=["mod_loss", "delta_mass"],
    dataset_type="pt",
    # alphabet=ALPHABET_UNMOD,
    with_termini=False,
)

d.save_to_disk(path="prospect_data")
'''

