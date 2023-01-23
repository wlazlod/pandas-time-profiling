"""Profile Time Report

This tool allows the user to create the report of all columns in the
dataset. User have to pass the dataset and the name of time column.

This tool accepts pandas DataFrame.
"""
from datetime import datetime
from typing import Dict, Optional, List, Union
from pathlib import Path
import warnings
from tqdm import tqdm
import jinja2
from textwrap import dedent
from htmlmin.main import minify

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas_time_profiling import version
from pandas_time_profiling.utils import get_project_root
from pandas_time_profiling.variable_report import VariableReport

class ProfileTimeReport:
    """Generate a profile report from a Dataset stored as a pandas `DataFrame`.
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        time_variable: str = None,
        time_unit: Optional[str] = "M",
        title: Optional[str] = "Pandas Time Profiling Report"
    ):
        """Generate a ProfileTimeReport based on a pandas DataFrame

        Args:
            df (pd.DataFrame): the pandas DataFrame.
            time_variable (str): name of the column regarding to which the analysis will be carried out.
            time_unit (str, optional): "Y" - year, "M" - month, "W" - week, "D" - day. Defaults to "M".
            title (str, optional): title of report page. Defaults to "Pandas Time Profiling Report".

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Can init a ProfileTimeReport with no DataFrame.")

        if time_variable not in df.columns:
            raise ValueError(f"Column: {time_variable} not found in the DataFrame.")

        if not is_datetime(df[time_variable]):
            raise ValueError(f"Column: {time_variable} is not datetime type. Please parse the variable.")

        if time_unit not in ("Y", "M", "W", "D"):
            raise ValueError("Variable `time_unit` can only take values: 'Y', 'M', 'W', 'D' "
                             "for year, month, week and day respectively")

        self.df = df
        self.time_variable = time_variable
        self.time_unit = time_unit
        self.title = str(title)
        self.html = ""
        self.create_report_assets()

    def create_report_assets(self):
        """Generate assets for report
        """
        start = datetime.now()
        self.date = start.strftime("%Y-%m-%d")
        self.start_time = start.strftime("%Y-%m-%d %H:%M:%S")
        self.version = version.__version__

        df_grouped = self.df.groupby(self.df[self.time_variable]\
                         .dt.to_period(self.time_unit))

        self.variable_list = [VariableReport(df_grouped[column], self.df[column]) \
            for column in tqdm(self.df, total=len(self.df.columns))]

        self.variable_reports = [variable.create_report() for variable in self.variable_list]
        self.types = calculate_variable_types(self.variable_list)
        end = datetime.now()
        self.date = end.strftime("%Y-%m-%d")
        self.end_time = end.strftime("%Y-%m-%d %H:%M:%S")

    def to_html(self) -> str:
        """Generate and return complete template as lengthy string
            for using with frameworks.
        Returns:
            Profiling report html.
        """
        path = get_project_root() / "templates"
        template_loader = jinja2.FileSystemLoader(searchpath=path, followlinks=True)
        template_env = jinja2.Environment(loader=template_loader)
        template_file = "template.jinja"
        template = template_env.get_template(template_file)

        rendered_template = template.render(VERSION = self.version,
                                    DATE = self.date,
                                    TITLE = self.title,
                                    N_VARIABLES = str(len(self.variable_list)),
                                    N_OBSERVATIONS = str(len(self.df)),
                                    ANALYSIS_START = self.start_time,
                                    ANALYSIS_FINISH = self.end_time,
                                    TIME_VARIABLE = self.time_variable,
                                    TIME_UNIT_LONG = self.time_unit,
                                    VARIABLE_TYPES = self.types,
                                    VARIABLE_REPORTS = self.variable_reports
                                   )

        self.html = minify(rendered_template, remove_all_empty_space=True, remove_comments=True)
    
        return self.html

    def to_file(self, output_file: Union[str, Path]) -> None:
        """Write the report to a file.
        By default a name is generated.
        Args:
            output_file: The name or the path of the file to generate including the extension.
        """
        if not isinstance(output_file, Path):
            output_file = Path(str(output_file))

        data = self.to_html()

        if output_file.suffix != ".html":
            suffix = output_file.suffix
            output_file = output_file.with_suffix(".html")
            warnings.warn(
                f"Extension {suffix} not supported. For now we assume .html was intended. "
                f"To remove this warning, please use .html"
            )

        output_file.write_text(data, encoding="utf-8")

def calculate_variable_types(variable_list: List[VariableReport]) -> Dict[str, int]:
    """Sums numbers of each variable type

    Args:
        variable_list (List[VariableReport]): List of Variable Summaries

    Returns:
        Dict[str, int]: Dictionary which key is a variable type
            and value is a number of variables of this type
    """
    types_list = dict()
    for variable in variable_list:
        if variable.var_type not in types_list:
            types_list[variable.var_type] = 1
        else:
            types_list[variable.var_type] += 1

    return types_list
