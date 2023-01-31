import pandas as pd
from typing import List
from dataclasses import dataclass


@dataclass
class DataCleaner:
    """Class which proposes several methods to clean a pandas DataFrame."""

    df: pd.DataFrame

    def downcast_columns(self):
        """Downcasts columns with the most appropriate data type

        Returns:
            self: returns the same instance of the class, allowing method chaining.
        """

        float_cols = self.df.select_dtypes(include="float").columns
        integer_cols = self.df.select_dtypes(include="int").columns
        object_cols = self.df.select_dtypes(include="object").columns

        for col in self.df.columns:
            if col in float_cols:
                self.df[col] = pd.to_numeric(self.df[col], downcast="float")
            if col in integer_cols:
                self.df[col] = pd.to_numeric(self.df[col], downcast="unsigned")
            if (col in object_cols) and (self.df[col].nunique() / len(self.df) < 0.5):
                self.df[col] = self.df[col].astype("category")
            if (col in object_cols) and (self.df[col].nunique() / len(self.df) >= 0.5):
                self.df[col] = self.df[col].astype("str")

        return self

    def sort_columns(self, columns_to_sort: List[str], ascending: List[bool]):
        """Sorts columns ascendingly or descendingly

        Args:
            columns_to_sort (List[str]): list of columns to be sorted
            ascending (List[bool]): define how to sort them

        Returns:
            self: returns the same instance of the class, allowing method chaining.
        """

        self.df = self.df.sort_values(
            by=columns_to_sort,
            ascending=ascending,
            ignore_index=True,
        )

        return self

    def uppercase_column_names(self):
        """Uppercases column names

        Returns:
            self: returns the same instance of the class, allowing method chaining
        """

        self.df.columns = self.df.columns.str.upper().str.replace(" ", "_")

        return self

    def reorder_columns(self, columns_order: List[str]):
        """Reorders columns in the DataFrame

        Args:
            columns_order (List[str]): list of columns to be reordered

        Returns:
            self: returns the same instance of the class, allowing method chaining
        """

        self.df = self.df[columns_order]

        return self

    def rename_columns(self, column_map: dict):
        """Renames columns in the DataFrame

        Args:
            column_map (Dict[str]): dictionary of old column name keys
            and new column name values

        Returns:
            self: returns the same instance of the class, allowing method chaining
        """

        self.df = self.df.rename(columns=column_map)
        return self

    def replace_values(self, **kwargs):
        """Replace values in one or multiple columns of a Pandas DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            **kwargs: One or multiple dictionaries in which keys are the values to be
            replaced and values are the new values. Columns and dictionaries are
            specified as keyword arguments, e.g. column_name=replace_dict.

        Returns:
            pd.DataFrame: The DataFrame with
        """

        for column, replace_dict in kwargs.items():
            self.df[column].replace(replace_dict, inplace=True)
        return self

    def drop_duplicates(self):
        """Drops the duplicate rows from DataFrame

        Returns:
            self: returns the same instance of the class, allowing method chaining
        """

        self.df = self.df.drop_duplicates(keep="first", ignore_index=True)
        return self

    def fill_na(self, value):
        """Fills missing values in the DataFrame with a specific value

        Args:
            value: value to fill missing values with

        Returns:
            self: returns the same instance of the class, allowing method chaining
        """
        
        self.df = self.df.fillna(value)
        return self

    def drop_rows(self, condition: str):
        """Drop rows corresponding to the given condition

        Args:
            condition: indicates rows to be dropped

        Returns:
            self: returns the same instance of the class, allowing method chaining
        """

        self.df = self.df.query("not ({})".format(condition)).reset_index(drop=True)
        return self

    def keep_rows(self, condition: str):
        """Keep rows corresponding to the given condition

        Args:
            condition: indicates rows to be kept

        Returns:
            self: returns the same instance of the class, allowing method chaining
        """

        self.df = self.df.query(condition).reset_index(drop=True)
        return self

    def create_column(
        self,
        new_column_name: str,
        conditions: List[str],
        logic_operator: str,
        value_if_true,
        value_if_false,
    ):
        """
        Creates a new column in the dataframe by evaluating a set of conditions.

        Args:
            new_column_name (str): the name of the new column to be created.
            conditions (List(str)): a list of conditions to be evaluated.
            logic_operator (str): the logical operator to be used
            in evaluating the conditions ('AND' or 'OR').
            value_if_true: the value to be set in the new column
            if the conditions are True.
            value_if_false: the value to be set in the new column
            if the conditions are False.

        Returns:
            self: returns the same instance of the class, allowing method chaining
        """

        condition = (
            " & ".join(conditions)
            if logic_operator == "AND"
            else " | ".join(conditions)
        )
        self.df[new_column_name] = self.df.eval(f"({condition})")
        self.df[new_column_name] = (
            self.df[new_column_name]
            .astype(int)
            .replace({1: value_if_true, 0: value_if_false})
        )
        return self

    def get_cleaned_dataframe(self) -> pd.DataFrame:
        """Returns the cleaned DataFrame

        Returns:
            self.df (pd.DataFrame): cleaned dataframe
        """

        return self.df
