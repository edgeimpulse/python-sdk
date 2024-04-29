# Changelog

Description of notable changes to the [Edge Impulse Python SDK](https://pypi.org/project/edgeimpulse/).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.9] - 2024-04-26

Work in progress: adding EON Tuner functionality to the Python SDK.

### Added

- `experimental.tuner` submodule for working with the eon tuner for hyperparameter optimization
- `infer_category_and_label_from_filename` upload transform under `experimental.data`. It will autodetect the category from the file path if it
  contains testing or training. It will use the first part of the filename as a label where the separator is a period.
  For instance: myfiles/testing/wave.1.cbor where the category is testing and the label is wave.

## [1.0.8] - 2024-01-15

### Added

- `experimental` submodule for testing new features
- `experimental.data` submodule for testing data upload and download features
- `download_samples_by_ids()` function in `experimental.data` to download sample data and metadata from Edge Impulse project
- `upload_directory()` function in `experimental.data` to upload all files in a directory to an Edge Impulse project
- `upload_numpy()` function in `experimental.data` to upload Numpy arrays
- `upload_pandas_sample()`, `upload_pandas_dataframe_wide()`, `upload_pandas_dataframe()`, `upload_pandas_dataframe_with_group()` functions in `experimental.data` to uplaod pandas (and pandas-like) dataframes

## [1.0.7] - 2023-11-01

### Added

- Now using [ruff](https://github.com/astral-sh/ruff) tool to perform linting. Black will continue to be used for formatting.
- `upload_sample()` function that allows you to upload individual samples to a project. You must wrap each sample data with the `Sample` class, as it holds the filename, metadata, etc.
- `get_filename_by_id()` and `get_ids_by_filename()` in data to get IDs or filename from samples that have been uploaded to a project
- `delete_all_samples()` and `delete_sample_by_id()` in data so users can delete samples from a project

### Fixed

- Minor linting (syntax) throughout the code to adhere to ruff rules

## [1.0.6] - 2023-08-24

### Added

- First time adding separate changelog
- ONNX ModelProto format support for `profile()` and `deploy()`
- Added `timeout_sec` parameter to `profile()` and `deploy()`

### Changed

- Reorganized directory structure for easier deployment
- Switched from Sphinx markdown to Sphinx HTML (furo theme) to generate the API reference guide
