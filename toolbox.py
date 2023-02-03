import pandas as pd
import plotly.graph_objects as go
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

    def sort_columns(self, columns_to_sort: list[str], ascending: list[bool]):
        """Sorts columns ascendingly or descendingly

        Args:
            columns_to_sort (list[str]): list of columns to be sorted
            ascending (list[bool]): define how to sort them

        Returns:
            self: returns the same instance of the class, allowing method chaining
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

    def reorder_columns(self, columns_order: list[str]):
        """Reorders columns in the DataFrame

        Args:
            columns_order (list[str]): list of columns to be reordered

        Returns:
            self: returns the same instance of the class, allowing method chaining
        """

        self.df = self.df[columns_order]

        return self

    def rename_columns(self, column_map: dict[str, str]):
        """Renames columns in the DataFrame

        Args:
            column_map (dict[str, str]): dictionary of old column name keys
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
            self: returns the same instance of the class, allowing method chaining
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
        conditions: list[str],
        logic_operator: str,
        value_if_true,
        value_if_false,
    ):
        """
        Creates a new column in the dataframe by evaluating a set of conditions

        Args:
            new_column_name (str): the name of the new column to be created.
            conditions (list(str)): a list of conditions to be evaluated.
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


@dataclass
class DataVisualiser:

    data_to_plot: pd.DataFrame

    def get_barplot(self, x_col: str, y_col: str, bar_color: str) -> go.Figure:
        """Plots barplot

        Args:
            x_col (str): X axis
            y_col (str): Y axis
            bar_color (str): color of the bars

        Returns:
            go.Figure: the barplot to display
        """

        barplot = go.Figure(
            [
                go.Bar(
                    x=self.data_to_plot[x_col],
                    y=self.data_to_plot[y_col],
                    marker=dict(color=bar_color),
                    text=self.data_to_plot[y_col],
                    textfont=dict(family="Avenir LT Std"),
                    texttemplate="%{text:.3s}",
                    textposition="outside",
                )
            ]
        )

        return barplot

    def get_lineplot(self, x_col: str, y_col: str, line_color: str) -> go.Figure:
        """Plots lineplot

        Args:
            x_col (str): X axis
            y_col (str): Y axis

        Returns:
            go.Figure: the lineplot to display
        """

        def improve_text_position(value_to_display: pd.Series) -> list[str]:
            """Improves text position

            Args:
                value_to_display (pd.Series): series containing values to displau

            Returns:
                list[str]: best possible position for each value to display
            """
            positions = [
                "top center",
                # "top left",
                # "top right",
                "bottom center",
                # "bottom left",
                # "bottom right",
            ]
            return [positions[i % len(positions)] for i in range(len(value_to_display))]

        lineplot = go.Figure(
            data=go.Scatter(
                x=self.data_to_plot[x_col],
                y=self.data_to_plot[y_col],
                marker=dict(color=line_color),
                text=self.data_to_plot[y_col],
                textfont=dict(family="Avenir LT Std"),
                textposition=improve_text_position(self.data_to_plot[y_col]),
                texttemplate="%{text:.3s}",
                mode="lines+markers+text",
            )
        )

        return lineplot


def format_graph(
    figure_to_format: DataVisualiser,
    graph_width: int,
    graph_height: int,
    x_title: str,
    y_title: str,
    y_suffix: str = "",
) -> DataVisualiser:

    figure_to_format.update_layout(
        title=dict(
            text=f"",
            x=0.5,
        ),
        width=graph_width,
        height=graph_height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        titlefont=dict(
            family="Avenir LT Std",
            size=21,
        ),
        font=dict(
            family="Avenir LT Std",
            color="rgb(89, 89, 89)",
            size=20,
        ),
        margin=go.layout.Margin(
            l=100,
            r=50,
            b=50,
            t=80,
            pad=4,
        ),
        showlegend=False,
    ),

    figure_to_format.update_xaxes(
        title_text=f"<b>{x_title}</b>",
        titlefont=dict(
            family="Avenir LT Std",
            size=25,
        ),
        tickfont=dict(
            family="Avenir LT Std",
            size=20,
        ),
        zeroline=True,
        linecolor="rgb(89, 89, 89)",
    )

    figure_to_format.update_yaxes(
        title_text=f"<b>{y_title}</b>",
        titlefont=dict(
            family="Avenir LT Std",
            color="rgb(89, 89, 89)",
            size=25,
        ),
        tickfont=dict(
            family="Avenir LT Std",
            size=20,
            color="rgb(89, 89, 89)",
        ),
        ticksuffix=y_suffix,
        zeroline=False,
        linecolor="rgb(89, 89, 89)",
    )

    return figure_to_format
