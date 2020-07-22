"""Utilities for analysis results export"""
import pandas as pd
import json

from google.protobuf.json_format import MessageToJson
from collections import OrderedDict
from typing import List, Union

from ml_eda.constants import COMMON_ORDER, NUMERICAL_ORDER, CATEGORICAL_ORDER
from ml_eda.metadata import run_metadata_pb2


def create_df_descriptive_row_from_analysis(
    attribute_name: str,
    analysis: run_metadata_pb2.Analysis
) -> pd.DataFrame:
  # pylint: disable-msg=too-many-locals
  """Creates pandas.DataFrame storing descriptive analysis result.

  Args:
      attribute_name: (string), name of the attribute
      analysis: (run_metadata_pb2.Analysis), analysis holding
      all the metrics

  Returns:
      pd.DataFrame
  """
  metrics = analysis.smetrics
  attribute_type = analysis.features[0].type

  # Make sure the display order of each attribute is consistent
  common_order = COMMON_ORDER
  if attribute_type == run_metadata_pb2.Attribute.NUMERICAL:
    detail_order = NUMERICAL_ORDER
  else:
    detail_order = CATEGORICAL_ORDER
  # Use a OrderedDict to store the result
  result_holder = OrderedDict(
      [(item, 0) for item in common_order + detail_order])
  for item in metrics:
      name = run_metadata_pb2.ScalarMetric.Name.Name(item.name)
      result_holder[name] = item.value

  return pd.DataFrame(
      data=result_holder,
      index=[attribute_name]
  )


def create_df_from_TableMetric(
    table_metric: run_metadata_pb2.TableMetric) -> pd.DataFrame:
    """Creates a DataFrame for a TableMetric object. Currently, this function is
    used for CONTINGENCY_TABLE and TABLE_DESCRIPTIVE.

  Args:
      table_metric: (run_metadata_pb2.TableMetric)

  Returns:
      pd.DataFrame
  """
    supported_metric = {
        run_metadata_pb2.TableMetric.CONTINGENCY_TABLE,
        run_metadata_pb2.TableMetric.TABLE_DESCRIPTIVE
    }

    assert table_metric.name in supported_metric

    table_metric_json = MessageToJson(table_metric)
    table_metric_data = json.loads(table_metric_json)

    columns = table_metric_data['columnIndexes']
    table_data = OrderedDict()

    for row in table_metric_data['rows']:
        row_header = str(row['rowIndex']).strip()
        table_data[row_header] = [item['value'] for item in row['cells']]

    return pd.DataFrame.from_dict(
        data=table_data,
        columns=columns,
        orient='index'
    )


def create_df_from_simple_TableMetric(
    analysis: run_metadata_pb2.Analysis) -> pd.DataFrame:
    """Creates a DataFrame for an Analysis object containing a single-row TableMetric. Currently, this function is
    used for HISTOGRAM and VALUE_COUNTS.

  Args:
      analysis: (run_metadata_pb2.Analysis), the analysis should be one of the following:
      - HISTOGRAM for histogram of numerical attribute
      - VALUE_COUNTS for bar chart of categorical attributes

  Returns:
      pd.DataFrame
  """
    supported_analysis = {
        run_metadata_pb2.Analysis.HISTOGRAM,
        run_metadata_pb2.Analysis.VALUE_COUNTS
    }

    assert analysis.name in supported_analysis

    table_metric_json = MessageToJson(analysis)
    table_metric_dict = json.loads(table_metric_json)
    table_metric_data = table_metric_dict['tmetrics'][0]

    columns = table_metric_data['columnIndexes']
    table_data = OrderedDict()

    for row in table_metric_data['rows']:
        row_header = table_metric_dict['features'][0]['name']
        table_data[row_header] = [item['value'] for item in row['cells']]

    return pd.DataFrame.from_dict(
        data=table_data,
        columns=columns,
        orient='index'
    )


def create_no_order_pair_metric_df(
    analysis_list: List[run_metadata_pb2.Analysis],
    same_match_value: Union[str, float],
    table_name: str = "NA") -> pd.DataFrame:
  """Creates metric table for pairwise comparison

  Args:
      analysis_list: (List[run_metadata_pb2.Analysis])
      same_match_value: (Union[str, float])
      table_name: (str)

  Returns:
      pd.DataFrame
  """
  attribute_list = set()
  # a dictionary with {(attributeone, attributetwo): metric_value}
  analysis_name_value_map = {}
  for item in analysis_list:
    value = item.smetrics[0].value
    name_list = [att.name for att in item.features]
    attribute_list.update(name_list)
    analysis_name_value_map[tuple(name_list)] = value
    analysis_name_value_map[tuple(reversed(name_list))] = value
    analysis_name_value_map.update({(att.name, att.name): same_match_value for att in item.features})

  return pd.DataFrame(
    data=analysis_name_value_map,
    index=[table_name]).T.unstack()


def create_order_pair_metric_df(
    analysis_list: List[run_metadata_pb2.Analysis],
    same_match_value: Union[str, float],
    table_name: str = "NA") -> pd.DataFrame:
  """Creates metric DataFrame for pairwise comparison

  Args:
      analysis_list: (List[run_metadata_pb2.Analysis])
      same_match_value: (Union[str, float])
      table_name: (str)

  Returns:
      pd.DataFrame
  """
  # a dictionary with {attributeone-attributetwo: metric_value}
  analysis_name_value_map = {}
  for item in analysis_list:
    value = item.smetrics[0].value
    name_list = [att.name for att in item.features]
    analysis_name_value_map[tuple(name_list)] = value
    if name_list[0] == name_list[1]:
      analysis_name_value_map[tuple(name_list)] = same_match_value

  return pd.DataFrame(
    data=analysis_name_value_map,
    index=[table_name]).T.unstack()
