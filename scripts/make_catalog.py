from pathlib import Path

import click
import pandas as pd

SEARCH_PATTERNS = ["*.bdf", "*.edf", "*.vhdr", "*.set", "*.fif"]


@click.command()
@click.option("--cache_directory", type=click.Path(file_okay=False))
@click.option("--dataset_type_table", type=click.Path(exists=True))
@click.option("--output_file_dir", type=click.Path(file_okay=True))
@click.option("--train_ratio", type=float, default=0.95)
def main(cache_directory, dataset_type_table, output_file_dir, train_ratio):
    cache_directory = Path(cache_directory)
    dataset_type_table = pd.read_csv(dataset_type_table)

    # Take the last part of dataset_type: "research_eeg" -> "eeg"
    dataset_type_table["dataset_type"] = dataset_type_table["dataset_type"].map(
        lambda x: x.split("_")[-1]
    )

    # Convert "eeg" to 0, "ieeg" to 1, "meg" to 2
    modality_mapper = {"eeg": 0, "ieeg": 1, "meg": 2}
    dataset_type_table["dataset_type"] = dataset_type_table["dataset_type"].map(
        lambda x: modality_mapper[x]
    )

    # Create a mapping from dataset name to dataset type
    dataset_type_mapper = dataset_type_table.set_index("dataset_name")[
        "dataset_type"
    ].to_dict()

    catalog = []
    for search_pattern in SEARCH_PATTERNS:
        for file_path in cache_directory.rglob(search_pattern):
            # Find the path relative to the cache directory and take first part as dataset name
            dataset_name = file_path.relative_to(cache_directory).parts[0]

            if dataset_name not in dataset_type_mapper:
                # Dataset skipped
                continue

            catalog.append((str(file_path), dataset_type_mapper[dataset_name]))

    # Make the catalog
    file_catalog = pd.DataFrame(catalog, columns=["filename", "dataset_type"])

    # Save the catalog
    output_file_dir = Path(output_file_dir)
    output_file_dir.mkdir(parents=True, exist_ok=True)
    train_size = int(len(file_catalog) * train_ratio)

    # Shuffle and split the catalog
    file_catalog = file_catalog.sample(frac=1, random_state=42).reset_index()
    file_catalog.iloc[:train_size].to_csv(output_file_dir / "train.csv", index=False)
    file_catalog.iloc[train_size:].to_csv(output_file_dir / "test.csv", index=False)


if __name__ == "__main__":
    main()
