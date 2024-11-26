# ruff: noqa: D100, D101, D102, D103
import unittest
from edgeimpulse import util
from tests.util import delete_all_samples, assert_uploaded_samples
import logging

# import modin.pandas as pd
# import dask.dataframe as pd
# import polars as pd

import pandas as pd
from edgeimpulse import data

logging.getLogger().setLevel(logging.DEBUG)


class TestUploadPandas(unittest.TestCase):
    @unittest.skipUnless(
        util.pandas_installed(), "Test requires pandas but it was not available"
    )
    def setUp(self):
        delete_all_samples()

    def tearDown(self):
        delete_all_samples()

    # ----------------------------------------------------------------
    # One dataframe per sample
    # ----------------------------------------------------------------

    #
    # single sample (time-series) with sample rate of 1, axis will be called '0'
    #
    def test_upload_pandas_sample(self):
        # DAN: DataFrame with just data. What should be done here?
        # Is this a timeseries? Since it will be using a range index here

        df = pd.DataFrame([40, 2, 3, 2])
        res = data.upload_pandas_sample(df=df)
        self.assertEqual(len(res.successes), 1)

    #
    # Single sample (time-series) single axis with sample rate of 100.
    #
    def test_upload_pandas_sample2(self):
        df = pd.DataFrame([30, 2, 3, 20])
        res = data.upload_pandas_sample(
            df=df, sample_rate_ms=100, label="UP", metadata={"test": "true"}
        )

        self.assertEqual(len(res.successes), 1)
        assert_uploaded_samples(self, res.successes)

    #
    # single sample (time-series) with x,y axis with given sample rate of 100
    #
    def test_upload_pandas_sample_axis(self):
        df = pd.DataFrame([[1, 1, 2], [1, 3, 5], [1, 6, 1]], columns=["X", "Y", "Z"])
        res = data.upload_pandas_sample(
            df=df, filename="ODP200", axis_columns=["X", "Y"], sample_rate_ms=100
        )

        self.assertEqual(len(res.successes), 1)
        assert_uploaded_samples(self, res.successes)

    #
    # single sample (time-series) with single but infer sample_rate from index
    #
    def test_upload_pandas_single_sample_infer_sample_rate_from_index(self):
        # DataFrame with timestamp column/index.
        # In this case we can infer the sample rate from the index.
        df = pd.DataFrame(
            [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1],
            index=pd.timedelta_range(0, periods=11, freq="1T"),
        )
        res = data.upload_pandas_sample(df=df, filename="ODP300")

        self.assertEqual(len(res.successes), 1)
        assert_uploaded_samples(self, res.successes)

    #
    # single sample (non-timeseries) with a,b,c without timestamp
    #
    def test_upload_pandas_sample_single_sample_non_timeseries(self):
        df = pd.DataFrame([[1, 4, 2]], columns=["A", "B", "C"])
        res = data.upload_pandas_sample(df=df, filename="ODP400")

        self.assertEqual(len(res.successes), 1)
        assert_uploaded_samples(self, res.successes)

    # ----------------------------------------------------------------
    # One dataframe multiple samples
    # ----------------------------------------------------------------

    #
    # multiple samples that are not time series. Here it will just create
    # data that doesn't have a timeseries component to it.
    #
    def test_upload_pandas_sample_per_row(self):
        values = [
            [1, "nominal", 45, 80, 0.3, 0.7, 0.2, "amsterdam"],
            [2, "nominal", 50, 76, 0.4, 0.7, 0.3, "utrecht"],
            [3, "anomaly", 75, 40, 0.8, 0.5, 0.6, "delft"],
        ]

        df = pd.DataFrame(
            values,
            columns=[
                "id",
                "label",
                "temperature",
                "humidity",
                "x",
                "y",
                "z",
                "location",
            ],
        )

        res = data.upload_pandas_dataframe(
            df,
            label_col="label",
            metadata_cols=["location"],
            feature_cols=["temperature", "humidity", "x", "y", "z"],
        )

        self.assertEqual(len(res.successes), 3)
        self.assertEqual(len(res.fails), 0)
        assert_uploaded_samples(self, res.successes)

    #
    # multiple samples with timeseries per row where the column is a part of
    # the timeseries sample into a timeseries.
    #

    def test_upload_pandas_dataframe_timeseries_columns_wide_single_axis(self):
        # Csv file per row (transposed from wide to long)
        #
        # timestamp, 0
        # 0,         0.8
        # 100,       0.7
        # 200,       0.8
        # ....

        values = [
            [1, "idle", 0.8, 0.7, 0.8, 0.9, 0.8, 0.8, 0.7, 0.8],  # ...continued
            [2, "motion", 0.3, 0.9, 0.4, 0.6, 0.8, 0.9, 0.5, 0.4],  # ...continued
        ]

        df = pd.DataFrame(
            values, columns=["id", "label", "0", "1", "2", "3", "4", "5", "6", "7"]
        )

        res = data.upload_pandas_dataframe_wide(
            df,
            label_col="label",
            sample_rate_ms=100,
            metadata_cols=["id"],
            data_col_start=2,
            data_col_length=8,
        )

        self.assertEqual(len(res.fails), 0)
        self.assertEqual(len(res.successes), 2)

        assert_uploaded_samples(self, res.successes)

    #
    # Here we have even more. Multiple axis in the columns.
    #
    def test_upload_pandas_dataframe_timeseries_columns_wide_multi_axis(self):
        # Csv file per row (transposed from wide to long)

        # Csv file looks like this
        #
        # timestamp, x,     y,      z
        # 100,       0.8,   07,     0.8
        # 200,       0.8,   07,     0.8
        # 200,       0.8,   07,     0.8
        #

        values = [
            [1, "idle", 0.8, 0.7, 0.8, 0.9, 0.8, 0.8, 0.7, 0.8, 0.8],  # ...continued
            [2, "idle", 0.7, 0.8, 0.8, 0.8, 0.7, 0.8, 0.8, 0.9, 0.8],  # ...continued
            [3, "motion", 0.2, 0.8, 0.8, 0.8, 0.7, 0.8, 0.8, 0.9, 0.8],  # ...continued
        ]

        df = pd.DataFrame(
            values,
            columns=[
                "id",
                "label",
                "x0",
                "y0",
                "z0",
                "x1",
                "y1",
                "z1",
                "x2",
                "y2",
                "z2",
            ],
        )

        # In this case we can specify the axis names, allowing studio to parse the data properly.
        res = data.upload_pandas_dataframe_wide(
            df,
            label_col="label",
            metadata_cols=["id"],
            data_col_length=2,
            data_axis_cols=[
                "x",
                "y",
                "z",
            ],
            sample_rate_ms=100,
        )
        self.assertEqual(len(res.fails), 0)
        self.assertEqual(len(res.successes), 3)

        assert_uploaded_samples(self, res.successes)

    #
    # One dataframe that contains multiple samples, but is also timeseries.
    # The group is identified in this case by the location column
    #
    def test_upload_pandas_dataframe_timeseries_grouped(self):
        #
        # city,     country, date,                        location,  parameter, value,   unit
        # Paris,    FR,      2019-06-21 00:00:00+00:00,   FR04014,   no2,       20.0,    µg/m³
        # Paris,    FR,      2019-06-20 23:00:00+00:00,   FR04014,   no2,       21.8,    µg/m³
        # Paris,    FR,      2019-06-20 22:00:00+00:00,   FR04014,   no2,       26.5,    µg/m³
        # Paris,    FR,      2019-06-20 21:00:00+00:00,   FR04014,   no2,       24.9,    µg/m³
        # Paris,    FR,      2019-06-20 20:00:00+00:00,   FR04014,   no2,       21.4,    µg/m³
        #

        df = pd.read_csv("tests/sample_data/air_quality_no2_long.csv")

        # Add a timestamp column (not named "timestamp")
        df["time"] = pd.to_datetime(df["date"])

        # Rename the "date" column to "timestamp" to test upload
        df.rename(columns={"date": "timestamp"}, inplace=True)

        # upload samples
        res = data.upload_pandas_dataframe_with_group(
            df=df,
            group_by="location",
            timestamp_col="time",
            feature_cols=["value"],
        )
        self.assertEqual(len(res.fails), 0)
        assert_uploaded_samples(self, res.successes)

    #
    # Test non-dataframe input (to simulate pandas not being installed)
    #
    def test_upload_pandas_non_dataframe(self):
        # Check exception message from upload_pandas_dataframe
        with self.assertRaises(AttributeError) as context:
            data.upload_pandas_dataframe(
                df=3, feature_cols=["a", "b"], label_col="label"
            )
        self.assertTrue(
            str(context.exception).startswith("DataFrame methods on input object")
        )

        # Check exception message from data.upload_pandas_dataframe_wide
        with self.assertRaises(AttributeError) as context:
            data.upload_pandas_dataframe_wide(
                df=3,
                data_col_start=2,
                data_col_length=8,
                label_col="label",
                sample_rate_ms=100,
            )
        self.assertTrue(
            str(context.exception).startswith("DataFrame methods on input object")
        )

        # Check exception message from data.upload_pandas_sample
        with self.assertRaises(AttributeError) as context:
            data.upload_pandas_sample(df=3, sample_rate_ms=100)
        self.assertTrue(
            str(context.exception).startswith("DataFrame methods on input object")
        )

        # Check exception message from data.upload_pandas_dataframe_with_group
        with self.assertRaises(AttributeError) as context:
            data.upload_pandas_dataframe_with_group(
                df={"timestamp": 3},
                group_by="loc",
                timestamp_col="timestamp",
                feature_cols=["a"],
            )
        self.assertTrue(
            str(context.exception).startswith("DataFrame methods on input object")
        )
