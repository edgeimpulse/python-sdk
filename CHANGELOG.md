# Changelog

Description of notable changes to the [Edge Impulse Python SDK](https://pypi.org/project/edgeimpulse/).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
