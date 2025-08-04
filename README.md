# Simple EEG Tokenizer
## Make dataset
First make catalog of all training files:
```bash
python scripts/make_catalog.py --cache_directory /gpfs/data/oermannlab/public_data/eeg_datasets/cache --dataset_type_table outputs/dataset_type_df.csv --output_file_dir outputs/file_catalogs/
```  

Then make the pretraining dataset
```bash
python scripts/make_pretrain_data.py --catalog xxx --output_dir xxx ...
```

## Train tokenizer
```bash
python train_hydra.py data.batch_size=768 {other overrides}
```