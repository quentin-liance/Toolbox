import pandas as pd
from typing import List


class DataCleaner:
    """Class which proposes several methods to clean a raw pandas DataFrame."""

    def __init__(self, raw_df: pd.DataFrame):
        """Initializes the DataCleaner class with a pandas DataFrame

        Args:
            raw_df (pd.DataFrame): raw pandas DataFrame to be cleaned
        """
        self.raw_df = raw_df

    def downcast_columns(self):
        """Downcasts columns with the most appropriate data type

        Returns:
            self
        """

        float_columns = self.raw_df.select_dtypes(include="float").columns
        integer_columns = self.raw_df.select_dtypes(include="int").columns
        object_columns = self.raw_df.select_dtypes(include="object").columns

        for column in self.raw_df.columns:
            if column in float_columns:
                self.raw_df[column] = pd.to_numeric(
                    self.raw_df[column], downcast="float"
                )
            if column in integer_columns:
                self.raw_df[column] = pd.to_numeric(
                    self.raw_df[column], downcast="unsigned"
                )
            if (column in object_columns) and (
                self.raw_df[column].nunique() / len(self.raw_df) < 0.5
            ):
                self.raw_df[column] = self.raw_df[column].astype("category")
            if (column in object_columns) and (
                self.raw_df[column].nunique() / len(self.raw_df) >= 0.5
            ):
                self.raw_df[column] = self.raw_df[column].astype("str")

        return self

    def sort_columns(self, columns_to_sort: List[str], ascending: List[bool]):
        """Sorts columns ascendingly or descendingly

        Args:
            columns_to_sort (List[str]): list of columns to be sorted
            ascending (List[bool]): define how to sort them

        Returns:
            self
        """

        self.raw_df = self.raw_df.sort_values(
            by=columns_to_sort,
            ascending=ascending,
            ignore_index=True,
        )

        return self

    def uppercase_column_names(self):
        """Uppercases column names

        Returns:
            self
        """

        self.raw_df.columns = self.raw_df.columns.str.upper().str.replace(" ", "_")

        return self

    def reorder_columns(self, columns_order: List[str]):
        """Reorders columns in the DataFrame

        Args:
            columns_order (List[str]): list of columns to be reordered

        Returns:
            self
        """

        self.raw_df = self.raw_df[columns_order]

        return self

    def rename_columns(self, column_map: dict):
        """Renames columns in the DataFrame

        Args:
            column_map (dict): dictionary of old column name keys
            and new column name values

        Returns:
            self
        """
        self.raw_df = self.raw_df.rename(columns=column_map)
        return self

    def replace_values(self, column: str, value_map: dict):
        """Replaces values in a specific column of the DataFrame

        Args:
            column (str): name of the column to replace values in
            value_map (dict): dictionary of old value keys
            and new value values

        Returns:
            self
        """
        self.raw_df[column] = self.raw_df[column].replace(value_map)
        return self

    def drop_duplicate(self):
        """Drops the duplicate rows from DataFrame

        Returns:
            self
        """
        self.raw_df = self.raw_df.drop_duplicates()
        return self

    def fill_na(self, value):
        """Fills missing values in the DataFrame with a specific value

        Args:
            value: value to fill missing values with

        Returns:
            self
        """
        self.raw_df = self.raw_df.fillna(value)
        return self

    def drop_rows(self, column_name: str, value_to_drop):
        """Drop rows from the DataFrame that meet a certain condition.

        Args:
            column_name (str): column name.
            value_to_drop: value to remove


        Returns:
            self
        """

        self.raw_df = self.raw_df[
            ~(self.raw_df[column_name] == value_to_drop)
        ].reset_index(drop=True)

        return self

    def keep_rows(self, column_name: str, value_to_keep):
        """Select rows from the DataFrame that meet a certain condition.

        Args:
            column_name (str): column name.
            value_to_keep: value to keep


        Returns:
            self
        """

        self.raw_df = self.raw_df[
            (self.raw_df[column_name] == value_to_keep)
        ].reset_index(drop=True)

        return self

    def get_cleaned_df(self):
        """Returns the cleaned DataFrame

        Returns:
            pd.DataFrame: cleaned dataframe
        """

        return self.raw_df
