# Changelog

Description of notable changes to the [Edge Impulse Python SDK](https://pypi.org/project/edgeimpulse/).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.18] - 2024-11-20

- Fix internal version setting

## [1.0.17] - 2024-11-19

- Added 19 extra datasets

## [1.0.16] - 2024-11-18

- Added support for Keras 3

## [1.0.15] - 2024-08-13

- Moved `data` out of `experimental`
- Moved `tuner` out of `experimental`
- Fixed various typos and spelling errors

## [1.0.14] - 2024-08-29

- Added support for `fetch_samples`, `convert_sample_to_dataframe` under `edgeimpulse.experimental.util` 

## [1.0.13] - 2024-07-16

- Added optional polling to tuner.set_impulse_from_trial() to allow the function to block until the job is done.
- Added impulse.build() to build and deploy an existing/trained impulse in a project.
- Refactored model.deploy() to use the impulse.build() function.
- Changed from Sphinx to pdoc3 for API doc generation

## [1.0.12] - 2024-07-06

- Fixed bug where `upload_pandas_dataframe_with_group` would fail if `timestamp_col` was named something other than `"timestamp"`
- Fixed `Too many open files` bug when trying to upload a directory. Added `batch_size` parameter in `upload_directory()` to address this.
- Updated link in README to point to new API docs location.

## [1.0.11] - 2024-05-06

- Fixed spelling errors in various DocStrings
- Added `allow_duplicates=False` to `upload_directory` to allow uploading of duplicates
- Added `show_progress=False` to upload_directory to show progress
- Added datasets to the `__init__.py` so it's accessible from top namespace via ei.datasets etc.
- Fixed the basic logger in datasets because it was polluting with messages when no one asked for it.
- Fixed upload sample logging level to INFO from WARN in order to prevent pollution.
- Added `show_progress=False` to `download_dataset` to hide download progress by default.
- Added more friendly exceptions when API keys aren't set.

## [1.0.10] - 2024-05-01 - Hotfix

- Fixed broken access of experimental packages when accessing in direct manner i.e. `edgeimpulse.experimental.data.upload_samples()` without importing.
- Updated tests in order to catch this direct access style.

## [1.0.9] - 2024-04-26

Work in progress: adding EON Tuner functionality to the Python SDK.

### Added

- Added `run_project_job_until_completion` and `run_organization_job_until_completion` in the `util` module to stream job logs over websockets.
- Multi-label upload support for Edge Impulse upload format.
- Added MyPy static type checker.
- `EdgeImpulseApi()` in the `edgeimpulse.experimental` module for conveniently accessing all Edge Impulse APIs.
- `experimental.tuner` submodule for working with the eon tuner for hyperparameter optimization for your models.
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
- `upload_pandas_sample()`, `upload_pandas_dataframe_wide()`, `upload_pandas_dataframe()`, `upload_pandas_dataframe_with_group()` functions in `experimental.data` to upload pandas (and pandas-like) dataframe

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
