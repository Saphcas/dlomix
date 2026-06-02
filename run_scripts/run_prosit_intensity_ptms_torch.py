"""
    To run this script, use the following command:
    DLOMIX_BACKEND=torch python run_scripts/run_prosit_intensity_ptms_torch.py
"""
from dlomix.data.dataset import load_processed_dataset

import logging

import torch
from tqdm import tqdm

from dlomix.data import FragmentIonIntensityDataset
from dlomix.losses.intensity_torch import masked_spectral_distance, gaussian_nll
from dlomix.models import PrositIntensityPredictor, PrositIntensityUncertaintyPredictor

from sklearn.metrics import accuracy_score

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

BATCH_SIZE = 8
N_EPOCHS = 20
UNCERTAINTY_AWARE = True
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using cuda:0")
else:
    device = torch.device("cpu")
    print("Using cpu")

TRAIN_DATAPATH = "example_dataset/intensity/third_pool_processed_sample.parquet"
#TRAIN_DATAPATH = "prospect_data/train_dataset.parquet"
#VAL_DATAPATH = "prospect_data/val_dataset.parquet"
#TEST_DATAPATH = "prospect_data/test_dataset.parquet"

d = FragmentIonIntensityDataset(
    data_source=TRAIN_DATAPATH,
    #data_source=TRAIN_DATAPATH,
    #val_data_source=VAL_DATAPATH,
    #test_data_source=TEST_DATAPATH,
    max_seq_len=32,
    batch_size=BATCH_SIZE,
    val_ratio=0.2,
    model_features=["collision_energy_aligned_normed", "precursor_charge_onehot"],
    sequence_column="modified_sequence",
    label_column="intensities_raw",
    features_to_extract=["mod_loss", "delta_mass"],
    dataset_type="pt",
    with_termini=True,
    encoding_scheme="naive-mods", # Was missing in original code, setting encoding scheme to naive-mods is what allows modifications to exist (default is Un-modified (UNMOD))
)

#print(d)

# If this can work there's no need for pre-processing
#d = load_processed_dataset("prospect_data/processed")

if UNCERTAINTY_AWARE:
    model = PrositIntensityUncertaintyPredictor(
        seq_length=32,
        use_prosit_ptm_features=True,
        input_keys={
            "SEQUENCE_KEY": "modified_sequence",
        },
        meta_data_keys={
            "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
            "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
        },
        with_termini=True,
    )

    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    loss_criterion = gaussian_nll

    for epoch in tqdm(range(0, N_EPOCHS)):
        epoch_loss = 0
        mae = 0
        mv = 0
        accuracy = 0
        model.train()
        data_size = len(d.tensor_train_data)
        for batch in d.tensor_train_data:
            optimizer.zero_grad()

            output_mean, output_var, output_missingness = model(batch)
            loss = loss_criterion(batch["intensities_raw"], output_mean, output_var, output_missingness, batch["modified_sequence"])
        
            # print(loss.item())
            epoch_loss += loss.item()

            loss.backward()

            # Add before optimizer.step()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1, norm_type=2, error_if_nonfinite=False
            )
            optimizer.step()
            y_true = batch["intensities_raw"].detach()
            present = y_true > 0
            y_true[~present] = 0
            mae += torch.mean(torch.abs(torch.sub(output_mean, y_true))).item()
            mv += torch.mean(output_var).item()

            missing_tensor = torch.clone(output_missingness).detach()
            missing_tensor = torch.sigmoid(missing_tensor) # Turns logits into probabilities
            missing = missing_tensor > 0.5
            missing_tensor[missing] = 1
            missing_tensor[~missing] = 0
            missing_target = torch.clone(y_true).detach()
            missing_target[present] = 0
            missing_target[~present] = 1

            missing_tensor = torch.flatten(missing_tensor).detach()
            missing_target = torch.flatten(missing_target).detach()
            accuracy += accuracy_score(missing_target, missing_tensor, normalize=True)

        print(
            f"Epoch {epoch} Summary: Training Loss: {epoch_loss / data_size:.4f}\nMean absolute error: {mae / data_size:.4f},"
            f"\nMean variance: {mv / data_size:.4f},\nMissingness mean accuracy {accuracy / data_size:.4f}"
            )

        # Validation phase.
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            val_data_size = len(d.tensor_val_data)
            for batch in d.tensor_val_data:

                val_pred_cs = model(batch)
                val_loss = loss_criterion(batch["intensities_raw"], val_pred_cs[0], val_pred_cs[1], val_pred_cs[2], batch["modified_sequence"]) # Mean, var, missingness
                val_loss_total += val_loss.item()

            avg_val_loss = val_loss_total / val_data_size
        print(f"Epoch {epoch} Summary:  Validation Loss: {avg_val_loss:.4f}")
else:
    model = PrositIntensityPredictor(
        seq_length=32,
        use_prosit_ptm_features=True,
        input_keys={
            "SEQUENCE_KEY": "modified_sequence",
        },
        meta_data_keys={
            "COLLISION_ENERGY_KEY": "collision_energy_aligned_normed",
            "PRECURSOR_CHARGE_KEY": "precursor_charge_onehot",
        },
        with_termini=True,
    )

    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    
    loss_criterion = masked_spectral_distance

    for epoch in tqdm(range(0, N_EPOCHS)):
        epoch_loss = 0
        mae = 0
        model.train()
        data_size = len(d.tensor_train_data)
        for batch in d.tensor_train_data:
            optimizer.zero_grad()

            # output = model(batch["modified_sequence"])
            output = model(batch)
            # print("output: ", output)
            # print("label: ", batch["intensities_raw"])
            loss = loss_criterion(batch["intensities_raw"], output)
        
            # print(loss.item())
            epoch_loss += loss.item()

            loss.backward()

            # Add before optimizer.step()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1, norm_type=2, error_if_nonfinite=False
            )
            optimizer.step()

            y_true = batch["intensities_raw"].detach()
            present = y_true > 0
            y_true[~present] = 0
            mae += torch.mean(torch.abs(torch.sub(output, y_true))).item()
        print(f"Epoch {epoch} Summary: Training Loss: {epoch_loss / data_size:.4f}\nMean absolute error: {mae / data_size:.4f}")

        # Validation phase.
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            val_data_size = len(d.tensor_val_data)
            for batch in d.tensor_val_data:

                val_pred_cs = model(batch)
                val_loss = loss_criterion(batch["intensities_raw"], val_pred_cs)
                val_loss_total += val_loss.item()

            avg_val_loss = val_loss_total / val_data_size
        print(f"Validation Loss: {avg_val_loss:.4f}")

    print(val_pred_cs.shape)
    print(val_pred_cs[0])