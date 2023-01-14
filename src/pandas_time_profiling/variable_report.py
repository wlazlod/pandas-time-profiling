"""Auxilary module which generates summary of each column
"""
from typing import Dict

import io
import base64
import hashlib
import warnings
import jinja2

import matplotlib.pyplot as plt
import pandas as pd
from pretty_html_table import build_table

from pandas_time_profiling.utils import get_project_root


class VariableReport:
    """Auxiliary class used to create "summary" of single column
    """

    bool_relative_stats = ('true_pct', 'false_pct', 'NA_pct')
    bool_absolute_stats = ('count', 'true', 'false', 'NA')
    bool_stats = ('true', 'false', 'NA')

    basic_stats_dict = {'count': 'size',
                        'distinct': 'nunique',
                        'NA' : lambda x: pd.isna(x).sum(),
                        'NA_pct' : lambda x: pd.isna(x).sum()/len(x)
                        }

    bool_functions_dict = {'false' : lambda x: (x == 0).sum(),
                           'true' : lambda x: (x == 1).sum(),
                           'false_pct' : lambda x: (x == 0).sum()/len(x),
                           'true_pct' : lambda x: (x == 1).sum()/len(x),
                          }

    descriptive_functions_dict = {'mean' : 'mean',
                                  'std' : 'std',
                                  'skewness' : 'skew'
    }

    quantile_functions_dict = {'max' : "max",
                              'q99' : lambda x: x.quantile(0.99),
                              'Q3' : lambda x: x.quantile(0.75),
                              "median": "median",
                              'Q1' : lambda x: x.quantile(0.25),
                              'q01' : lambda x: x.quantile(0.01),
                              "min": "min",
    }

    numeric_functions_dict = {'positive' : lambda x: (x >0).sum(),
                              'zero' : lambda x: (x == 0).sum(),
                              'negative' : lambda x: (x < 0).sum()
    }

    numeric_functions_pct_dict = {
                              'positive_pct' : lambda x: (x > 0).sum()/len(x),
                              'zero_pct' : lambda x: (x == 0).sum()/len(x),
                              'negative_pct' : lambda x: (x < 0).sum()/len(x),
    }

    def __init__(
        self,
        var_grouped: pd.core.groupby.generic.SeriesGroupBy,
        var: pd.Series,
        max_cat_variables: int = 5
    ):
        """Generate a summary of a column

        Args:
            var_grouped (pd.core.groupby.generic.SeriesGroupBy): Grouped column.
            var (pd.Series): Raw column.
            max_cat_variables (int, optional): Maximum number of values considered categorical.
                Defaults to 5.
            mapped_variables (Dict[str, str], optional): Dictionary with some mapped types
                (can override automatic mapping). Defaults to None.
        """
        self.var_grouped = var_grouped
        self.var = var
        self.var_type = check_variable_type(variable = var,
                                            max_cat_variables = max_cat_variables)
        self.var_name = self.var.name
        self.max_cat_variables = max_cat_variables

        #VARIABLE_ID - used to identify variables
        self.var_id = hashlib.sha1(self.var_name.encode("UTF-8")).hexdigest()[:10]

        if self.var_type not in ("constant", "bool"):
            features = self.var.value_counts().index.to_list()
            self.distinct = len(features)
            self.frequent_features = features[:self.max_cat_variables]
            self.value_functions_dict = \
                {f"{feat if feat != '' else '<EMPTY_STR>'}" : \
                    lambda x, feat=feat: (x == feat).sum() for feat in self.frequent_features}
            self.value_pct_functions_dict = \
                {f"{feat if feat != '' else '<EMPTY_STR>'}_pct" :\
                    lambda x, feat=feat: (x == feat).sum()/len(x) for feat in self.frequent_features}

            if self.distinct > self.max_cat_variables:
                self.value_functions_dict["OTHER_VALUE"] = \
                    lambda x: (~x.isin(self.frequent_features)).sum()
                self.value_pct_functions_dict["OTHER_VALUE_PCT"] = \
                    lambda x: (~x.isin(self.frequent_features)).sum()/len(x)

        self.var_df = self.calculate_stats()

        pd.options.display.float_format = "{:,.3f}".format

        if self.var_type in ("bool", "string_categorical","string", "categorical", "numerical"):
            self.basic_stats_table = build_table(
                self.var_df[list(self.basic_stats_dict.keys())].T,
                'grey_light',
                index = True)

        if self.var_type == 'bool':
            self.relative_distribution_table = build_table(
                self.var_df[list(self.bool_relative_stats)].T,
                'grey_light',
                index = True)
            self.relative_distribution_graph = self.create_relative_distribution_graph(
                list(self.bool_relative_stats),
                list(self.bool_stats))
            self.absolute_distribution_table = build_table(
                self.var_df[list(self.bool_absolute_stats)].T,
                'grey_light',
                index = True)
            self.absolute_distribution_graph = self.create_absolute_distribution_graph(list(self.bool_stats))
        elif self.var_type in ("string_categorical", "string", "categorical", "numerical"):
            self.relative_distribution_table = build_table(
                self.var_df[list(self.value_pct_functions_dict.keys())].T,
                'grey_light',
                index = True)
            self.relative_distribution_graph = self.create_relative_distribution_graph(
                list(self.value_pct_functions_dict.keys()),
                list(self.value_functions_dict.keys()))
            self.absolute_distribution_table = build_table(
                self.var_df[list(self.value_functions_dict.keys())].T,
                'grey_light',
                index = True)
            self.absolute_distribution_graph = self.create_absolute_distribution_graph(
                list(self.value_functions_dict.keys()))
            if self.var_type in ("categorical", "numerical"):
                self.descriptive_functions_table = build_table(
                    self.var_df[list(self.descriptive_functions_dict.keys())].T,
                    'grey_light',
                    index = True)
                self.descriptive_functions_graph = self.create_absolute_distribution_graph(['mean'])
                self.quantile_functions_table = build_table(
                    self.var_df[list(self.quantile_functions_dict.keys())].T,
                    'grey_light',
                    index = True)
                self.quantile_functions_graph = self.create_box_whiskers_graph()
                self.numeric_functions_table = build_table(
                    self.var_df[list(self.numeric_functions_dict.keys()) \
                        + list(self.numeric_functions_pct_dict.keys())].T,
                    'grey_light',
                    index = True)
                self.numeric_functions_graph = self.create_relative_distribution_graph(
                    list(self.numeric_functions_pct_dict.keys()),
                    list(self.numeric_functions_dict.keys()))


    def calculate_stats(self):
        """Calculates statistics of column, regarding it's tyle

        Returns:
            pd.DataFrame: DataFrame with calculated statistics.
        """
        if self.var_type == 'bool':
            agg_functions_dict = {**self.basic_stats_dict,
                                  **self.bool_functions_dict
                                 }
        elif self.var_type in ('categorical', 'numerical'):
            agg_functions_dict = {**self.basic_stats_dict,
                                  **self.descriptive_functions_dict,
                                  **self.quantile_functions_dict,
                                  **self.numeric_functions_dict,
                                  **self.numeric_functions_pct_dict,
                                  **self.value_functions_dict,
                                  **self.value_pct_functions_dict
                                 }
        elif self.var_type in ("string", "string_categorical"):
            agg_functions_dict = {**self.basic_stats_dict,
                                  **self.value_functions_dict,
                                  **self.value_pct_functions_dict
                                 }
        else:
            return pd.DataFrame()

        return self.var_grouped.agg(**agg_functions_dict)

    def create_relative_distribution_graph(self,
                                           relative_stats: tuple,
                                           labels: tuple):
        """Generates relative distribution (0-100%) of values graph

        Args:
            absolute_stats (tuple): column names

        Returns:
            str: base64 encoded SVG picture with graph
        """
        buffer = io.StringIO()
        fig, ax = plt.subplots(1,1,figsize=(20,6))
        self.var_df.plot(y=relative_stats, kind='bar', stacked=True,
                         colormap=plt.cm.Accent, ax = ax)
        legend = ax.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, ncol=len(relative_stats))

        ax.set_yticks(ax.get_yticks().tolist())
        ax.set_yticklabels([f'{x:,.0%}' for x in ax.get_yticks()])

        plt.savefig(buffer, format = "svg", bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buffer.getvalue().encode('utf-8')).decode('utf-8')

    def create_absolute_distribution_graph(self,
                                           absolute_stats: tuple):
        """Generates absolute distribution of values graph

        Args:
            absolute_stats (tuple): column names

        Returns:
            str: base64 encoded SVG picture with graph
        """
        buffer = io.StringIO()
        fig, ax = plt.subplots(1,1,figsize=(20,6))
        self.var_df.plot(y=absolute_stats, kind='line',
                         stacked=False, colormap=plt.cm.Accent, ax = ax)
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=len(absolute_stats))
        plt.savefig(buffer, format = "svg", bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buffer.getvalue().encode('utf-8')).decode('utf-8')

    def create_box_whiskers_graph(self):
        """Generates box and whiskers distribution plot

        Returns:
            str: base64 encoded SVG picture with graph
        """
        var_df_box = self.var_df.copy().rename_axis('label').reset_index()
        stats_dict = {'median': 'med',
                      'q99': "whishi",
                      'Q3': "q3",
                      'Q1': "q1",
                      'q01': "whislo"}
        box = var_df_box.rename(columns=stats_dict)[stats_dict.values()].to_dict('records')
        for graph in box:
            graph['fliers'] = []

        buffer = io.StringIO()
        fig, ax = plt.subplots(1,1,figsize=(20,6))
        ax.bxp(box)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xticklabels(var_df_box['label'].to_list())
        plt.savefig(buffer, format = "svg", bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buffer.getvalue().encode('utf-8')).decode('utf-8')

    def create_report(self):
        """Write the report to a file.
        By default a name is generated.
        Returns:
            output_file: The name or the path of the file to generate including the extension.
        """
        path = get_project_root() / "templates" / "variable_templates"
        template_loader = jinja2.FileSystemLoader(searchpath=path)
        template_env = jinja2.Environment(loader=template_loader)
        if self.var_type == 'bool':
            template_file = "bool.jinja"
        elif self.var_type in ("string_categorical","string", "categorical", "numerical"):
            template_file = "numeric.jinja"
        else:
            template_file = "rejected.jinja"
        template = template_env.get_template(template_file)
        if self.var_type == 'bool':
            output_text = template.render(VARIABLE_ID=self.var_id,
                                    VARIABLE_NAME=self.var_name,
                                    TYPE = self.var_type,
                                    RELATIVE_DISTRIBUTION_TABLE = self.relative_distribution_table,
                                    RELATIVE_DISTRIBUTION_IMG = self.relative_distribution_graph,
                                    ABSOLUTE_DISTRIBUTION_TABLE = self.absolute_distribution_table,
                                    ABSOLUTE_DISTRIBUTION_IMG = self.absolute_distribution_graph
                                    )
        elif self.var_type in ("string_categorical", "string"):
            output_text = template.render(VARIABLE_ID=self.var_id,
                                    VARIABLE_NAME=self.var_name,
                                    TYPE = self.var_type,
                                    RELATIVE_DISTRIBUTION_TABLE = self.relative_distribution_table,
                                    RELATIVE_DISTRIBUTION_IMG = self.relative_distribution_graph,
                                    ABSOLUTE_DISTRIBUTION_TABLE = self.absolute_distribution_table,
                                    ABSOLUTE_DISTRIBUTION_IMG = self.absolute_distribution_graph,
                                    BASIC_STATS_TABLE = self.basic_stats_table
                                    )
        elif self.var_type in ("categorical", "numerical"):
            output_text = template.render(VARIABLE_ID=self.var_id,
                                    VARIABLE_NAME=self.var_name,
                                    TYPE = self.var_type,
                                    RELATIVE_DISTRIBUTION_TABLE = self.relative_distribution_table,
                                    RELATIVE_DISTRIBUTION_IMG = self.relative_distribution_graph,
                                    ABSOLUTE_DISTRIBUTION_TABLE = self.absolute_distribution_table,
                                    ABSOLUTE_DISTRIBUTION_IMG = self.absolute_distribution_graph,
                                    BASIC_STATS_TABLE = self.basic_stats_table,
                                    QUANTILES_TABLE = self.quantile_functions_table,
                                    QUANTILES_IMG = self.quantile_functions_graph,
                                    DESCRIPTIVE_TABLE = self.descriptive_functions_table,
                                    DESCRIPTIVE_IMG = self.descriptive_functions_graph,
                                    POSITIVE_NEGATIVE_TABLE = self.numeric_functions_table,
                                    POSITIVE_NEGATIVE_IMG = self.numeric_functions_graph
                                    )
        else:
            output_text = template.render(VARIABLE_ID=self.var_id,
                                          VARIABLE_NAME=self.var_name,
                                          TYPE = self.var_type,)

        return output_text

def check_variable_type(variable: pd.Series,
                        max_cat_variables: int = 5,
                        mapped_variables: Dict[str, str] = None) -> str:
    """Function that checks the variable type

    Args:
        variable (pd.Series): Examined Series.
        max_cat_variables (int, optional): Maximum number of values considered categorical.
            Defaults to 5.
        mapped_variables (Dict[str, str], optional): Dictionary with some mapped types
            (can override default mapping). Defaults to None.

    Returns:
        str: type of column
    """

    var_types = ["constant", "bool", "string_categorical",
                "string", "categorical", "numerical", "date", "object"]

    assert isinstance(variable, pd.Series), "variable must be pd.Series"
    assert (1 <= max_cat_variables <= 10), "max_cat_variables must be a number between 1 and 10"

    if mapped_variables is not None:
        if variable.name in mapped_variables:
            if mapped_variables[variable.name] in var_types:
                return mapped_variables[variable.name]
            else:
                warnings.warn(f"Warning: variable {variable.name} was in mapped_variables,"
                            f" but with wrong variable type. Possible types are: {var_types}")

    features_number = len(variable.value_counts(dropna=False))

    if features_number <= 1:
        return "constant"

    if pd.api.types.is_bool_dtype(variable):
        return "bool"

    if pd.api.types.is_string_dtype(variable):
        if features_number <= max_cat_variables:
            return "string_categorical"
        else:
            return "string"

    if pd.api.types.is_integer_dtype(variable) or pd.api.types.is_float_dtype(variable):
        if features_number <= max_cat_variables:
            return "categorical"
        else:
            return "numerical"

    if pd.api.types.is_datetime64_any_dtype(variable):
        return "date"
    else:
        return "object"
