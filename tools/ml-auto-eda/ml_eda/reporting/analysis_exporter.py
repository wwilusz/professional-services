"""Inventory holding helper function of building visualization for report."""
# pylint: disable-msg=wrong-import-position
import pandas as pd
import os

from typing import Union, Tuple, Dict, Optional
from pathlib import Path

from ml_eda.metadata import run_metadata_pb2
from ml_eda.reporting import exporter_utils as eutils
from ml_eda.orchestration.analysis_tracker import AnalysisTracker


class AnalysisExporter:
    """Allows exporting analysis results to files."""

    def __init__(self,
                 analysis_tracker: AnalysisTracker,
                 export_base_path: str):
        self.analysis_tracker = analysis_tracker
        self.export_base_path = export_base_path

    def export(self) -> Dict[str, str]:
      """Exports analysis results and returns paths to all data files

      Returns:
          Dict[str, str], (section_name, path to a data file)
      """
      # create export_base_path if it does not exist:
      Path(self.export_base_path).mkdir(parents=True, exist_ok=True)

      # holder
      output_paths = {}


      descriptive = AnalysisExporter.get_descriptive(self.analysis_tracker)
      for name, data in descriptive.items():
          export_path = os.path.join(
              self.export_base_path, '{}.csv'.format(name))
          output_paths[name] = export_path
          data.to_csv('{}{}'.format(export_path, '.csv'))

      pearson_correlation = AnalysisExporter.get_pearson_correlation(self.analysis_tracker)
      export_path = os.path.join(
          self.export_base_path, 'PEARSON_CORRELATION.csv')
      output_paths['PEARSON_CORRELATION'] = export_path
      pearson_correlation.to_csv(export_path)

      information_gain = AnalysisExporter.get_information_gain(self.analysis_tracker)
      export_path = os.path.join(
          self.export_base_path, 'INFORMATION_GAIN.csv')
      output_paths['INFORMATION_GAIN'] = export_path
      information_gain.to_csv(export_path)

      anova = AnalysisExporter.get_anova(self.analysis_tracker)
      export_path = os.path.join(
          self.export_base_path, 'ANOVA.csv')
      output_paths['ANOVA'] = export_path
      anova.to_csv(export_path)

      chi_square = AnalysisExporter.get_chi_square(self.analysis_tracker)
      export_path = os.path.join(
          self.export_base_path, 'CHI_SQUARE.csv')
      output_paths['CHI_SQUARE'] = export_path
      chi_square.to_csv(export_path)

      table_descriptive = AnalysisExporter.get_table_descriptive(self.analysis_tracker)
      for name, data in table_descriptive.items():
          export_path = os.path.join(
              self.export_base_path, '{}.csv'.format(name))
          output_paths[name] = export_path
          data.to_csv(export_path)

      return output_paths

    @staticmethod
    def get_descriptive(analysis_tracker: AnalysisTracker
                        ) -> Dict[str, pd.DataFrame]:
        """Exports descriptive section of the analysis

        Args:
            analysis_tracker: (AnalysisTracker)
            export_base_path: (string), the folder for holding exported data

        Returns:
            List[str], list of paths to exported data files (CSV)
        """
        numerical_attributes = analysis_tracker.get_numerical_attributes()
        categorical_attributes = analysis_tracker.get_categorical_attributes()

        # holder for the paths of exported data files
        output_paths = []

        # holders descriptive analysis results
        numerical_analysis_contents = []
        categorical_analysis_contents = []
        analysis_contents = {}

        for att in numerical_attributes:
            # base analysis is one holding basic descriptive statistics
            base_analysis = analysis_tracker.get_attribute_analysis(
                att, run_metadata_pb2.Analysis.Name.Name(
                    run_metadata_pb2.Analysis.DESCRIPTIVE))[0]
            numerical_analysis_contents.append(eutils.create_df_descriptive_row_from_analysis(att, base_analysis))
            # additional analysis is one holding histogram data for numerical attribute
            additional_analysis = analysis_tracker.get_attribute_analysis(
                att, run_metadata_pb2.Analysis.Name.Name(
                    run_metadata_pb2.Analysis.HISTOGRAM))[0]
            analysis_contents['{}-{}'.format('HISTOGRAM', att)] = eutils.create_df_from_simple_TableMetric(
                additional_analysis) # zmienic nazwe tej metody
        analysis_contents['DESCRIPTIVE-numerical'] = pd.concat(numerical_analysis_contents)

        for att in categorical_attributes:
            # base analysis is one holding basic descriptive statistics
            base_analysis = analysis_tracker.get_attribute_analysis(
                att, run_metadata_pb2.Analysis.Name.Name(
                    run_metadata_pb2.Analysis.DESCRIPTIVE))[0]
            categorical_analysis_contents.append(eutils.create_df_descriptive_row_from_analysis(att, base_analysis))
            # additional analysis is one holding value counts
            # for categorical attribute
            additional_analysis = analysis_tracker.get_attribute_analysis(
                att, run_metadata_pb2.Analysis.Name.Name(
                    run_metadata_pb2.Analysis.VALUE_COUNTS))[0]
            analysis_contents['{}-{}'.format('VALUE_COUNTS', att)] = eutils.create_df_from_simple_TableMetric(
                additional_analysis)
        analysis_contents['DESCRIPTIVE-categorical'] = pd.concat(categorical_analysis_contents)

        return analysis_contents

    @staticmethod
    def get_pearson_correlation(analysis_tracker: AnalysisTracker
                                ) -> Union[pd.DataFrame, None]:
        """Exports correlation-related data for numerical attributes

        Args:
            analysis_tracker: (AnalysisTracker), holder for all the analysis

        Returns:
            Union[str, None], a path to exported data file (CSV)
        """
        corr_analysis = analysis_tracker.get_analysis(
            run_metadata_pb2.Analysis.Name.Name(
                run_metadata_pb2.Analysis.PEARSON_CORRELATION))

        if corr_analysis:
            corr_analysis_df = eutils.create_no_order_pair_metric_df(
                analysis_list=corr_analysis,
                same_match_value=1.0,
                table_name="Pearson Correlation"
            )
            return corr_analysis_df

        return None

    @staticmethod
    def get_information_gain(analysis_tracker: AnalysisTracker
                             ) -> Union[pd.DataFrame, None]:
        """
        Exports information gain-related data for categorical attributes

        Args:
            analysis_tracker: (AnalysisTracker), holder for all the analysis

        Returns:
            Union[str, None], a path to exported data file (CSV)
        """

        # extract the information gain analysis result
        # each pair of categorical attributes will have one corresponding analysis
        info_analysis = analysis_tracker.get_analysis(
            run_metadata_pb2.Analysis.Name.Name(
                run_metadata_pb2.Analysis.INFORMATION_GAIN))

        if info_analysis:
            info_analysis_df = eutils.create_no_order_pair_metric_df(
                analysis_list=info_analysis,
                same_match_value=0.0,
                table_name="Information-Gain"
            )
            return info_analysis_df

        return None

    @staticmethod
    def get_anova(analysis_tracker: AnalysisTracker
                  ) -> Union[pd.DataFrame, None]:
        anova_analysis = analysis_tracker.get_analysis(
            run_metadata_pb2.Analysis.Name.Name(
                run_metadata_pb2.Analysis.ANOVA))

        if anova_analysis:
            anova_analysis_df = eutils.create_order_pair_metric_df(
                analysis_list=anova_analysis, same_match_value='NA')
            return anova_analysis_df

        return None

    @staticmethod
    def get_chi_square(analysis_tracker: AnalysisTracker
                       ) -> Union[pd.DataFrame, None]:

        chi_square_analysis = analysis_tracker.get_analysis(
            run_metadata_pb2.Analysis.Name.Name(
                run_metadata_pb2.Analysis.CHI_SQUARE))

        if chi_square_analysis:
            chi_square_analysis_df = eutils.create_no_order_pair_metric_df(
                analysis_list=chi_square_analysis,
                same_match_value=0.0,
                table_name="Chi-Square"
            )
            return chi_square_analysis_df

        return None

    @staticmethod
    def get_contingency_table(analysis_tracker: AnalysisTracker
                              ) -> Optional[Dict[Tuple[str, str], pd.DataFrame]]:
        # extract the contingency table analysis result
        # each pair of categorical attributes will have one corresponding analysis
        analysis_results = analysis_tracker.get_analysis(
            run_metadata_pb2.Analysis.Name.Name(
                run_metadata_pb2.Analysis.CONTINGENCY_TABLE))

        if analysis_results:
            analysis_contents = {}
            for analysis in analysis_results:
                attributes = tuple([item.name for item in analysis.features])
                analysis_results_df = eutils.create_df_from_TableMetric(
                    analysis.tmetrics[0])
                analysis_results_df.index.name = "{} / {}".format(attributes[0], attributes[1])
                analysis_contents['CONTINGENCY_TABLE-{}-{}'.format(
                    attributes[0], attributes[1])] = analysis_results_df
            return analysis_contents

        return None

    @staticmethod
    def get_table_descriptive(analysis_tracker: AnalysisTracker
                              ) -> Optional[Dict[Tuple[str, str], pd.DataFrame]]:
        # extract the contingency table analysis result
        # each pair of categorical attributes will have one corresponding analysis
        analysis_results = analysis_tracker.get_analysis(
            run_metadata_pb2.Analysis.Name.Name(
                run_metadata_pb2.Analysis.TABLE_DESCRIPTIVE))

        if analysis_results:
            analysis_contents = {}
            for analysis in analysis_results:
                attributes = tuple([item.name for item in analysis.features])
                analysis_results_df = eutils.create_df_from_TableMetric(
                    analysis.tmetrics[0])
                analysis_results_df.index.name = "{} / {}".format(attributes[0], attributes[1])
                analysis_contents['TABLE_DESCRIPTIVE-{}-{}'.format(
                    attributes[0], attributes[1])] = analysis_results_df
            return analysis_contents

        return None
