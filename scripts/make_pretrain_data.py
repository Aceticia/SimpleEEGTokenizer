from functools import partial
from pathlib import Path

import click
import mne
import numpy as np
import pandas as pd
from litdata import optimize


def exponential_moving_standardize(
    data,
    factor_new: float = 0.001,
    init_block_size: int | None = None,
    eps: float = 1e-4,
):
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(data[0:init_block_size], axis=i_time_axis, keepdims=True)
        init_std = np.std(data[0:init_block_size], axis=i_time_axis, keepdims=True)
        init_block_standardized = (data[0:init_block_size] - init_mean) / np.maximum(
            eps, init_std
        )
        standardized[0:init_block_size] = init_block_standardized
    return standardized.T


def load_one_file(
    record: dict,
    min_duration_s: int = 10,
    min_n_sensors: int = 16,
    patch_size: int = 1000,
):
    file_path, dataset_type = record["filename"], record["dataset_type"]

    try:
        raw = mne.io.read_raw(file_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    if raw.times[-1] < min_duration_s:
        return None

    if len(raw.ch_names) < min_n_sensors:
        return None

    if raw.times[-1] > 3600:
        # If exceeds 1 hour, randomly take one hour
        start_time = np.random.uniform(0, raw.times[-1] - 3600)
        end_time = start_time + 3600
        raw.crop(tmin=start_time, tmax=end_time)

    data = exponential_moving_standardize(raw.get_data())
    sfreq = raw.info["sfreq"]

    # Pad data to multiple of patch_size
    length = data.shape[1]
    if length % patch_size != 0:
        pad_length = patch_size - (length % patch_size)
        data = np.pad(data, ((0, 0), (0, pad_length)), mode="constant")

    length = data.shape[1]
    for i in range(0, length, patch_size):
        patch = data[:, i : i + patch_size]
        for j in range(len(patch)):
            yield (patch[j], sfreq, dataset_type)


@click.command()
@click.option("--catalog", type=click.Path(exists=True))
@click.option("--output_dir", type=click.Path())
@click.option("--min_n_sensors", type=int, default=16)
@click.option("--min_duration_s", type=int, default=10)
@click.option("--shuffle_seed", type=int, default=42)
@click.option("--subsample", type=float, default=1.0)
@click.option("--patch_size", type=int, default=1000)
@click.option("--num_workers", type=int, default=1)
def main(
    catalog,
    output_dir,
    min_n_sensors,
    min_duration_s,
    shuffle_seed,
    subsample,
    patch_size,
    num_workers,
):
    mne.set_log_level("ERROR")
    # Load and shuffle catalog
    catalog_df = pd.read_csv(catalog)
    catalog_df = catalog_df.sample(frac=subsample, random_state=shuffle_seed)

    # Optimize
    optimize(
        fn=partial(
            load_one_file,
            min_duration_s=min_duration_s,
            min_n_sensors=min_n_sensors,
            patch_size=patch_size,
        ),
        inputs=catalog_df.to_dict(orient="records"),
        output_dir=output_dir,
        num_workers=num_workers,
        chunk_bytes="100MB",
        start_method="fork",
    )


if __name__ == "__main__":
    main()
