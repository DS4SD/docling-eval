#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
import os
import glob
import statistics
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import datasets

from docling_core.types.doc.document import DoclingDocument, TableItem
from lxml import html
from pydantic import BaseModel

from docling_eval.utils.teds import TEDScorer

from datasets import Dataset
from datasets import load_dataset

from docling_eval.benchmarks.constants import BenchMarkColumns

_log = logging.getLogger(__name__)


class TableEvaluation(BaseModel):
    # filename: str
    TEDS: float
    is_complex: bool = False


class DatasetTableEvaluation(BaseModel):
    evaluations: list[TableEvaluation]
    total_mean_TEDS: float = 0.0
    simple_mean_TEDS: float = 0.0
    complex_mean_TEDS: float = 0.0
    total_std_TEDS: Optional[float] = None
    simple_std_TEDS: Optional[float] = None
    complex_std_TEDS: Optional[float] = None


def is_complex_table(table: TableItem) -> bool:
    r"""
    Implement the logic to check if table is complex
    """
    for cell in table.data.table_cells:
        if cell.row_span > 1 or cell.col_span > 1:
            return True
    return False


class TableEvaluator:
    r"""
    Evaluate table predictions from HF dataset with the columns:
    """

    def __init__(self) -> None:
        self._teds_scorer = TEDScorer()
        self._stopwords = ["<i>", "</i>", "<b>", "</b>", "<u>", "</u>"]

    def __call__(self, ds_path: Path, split: str="test") -> DatasetTableEvaluation:
        r"""
        Load a dataset in HF format. Expected columns with DoclingDocuments
        "GTDoclingDocument"
        "PredictionDoclingDocument"
        """
        logging.info(f"loading from: {ds_path}")

        # Load the Parquet file
        #dataset = Dataset.from_parquet("benchmarks/dpbench-tableformer/test/shard_000000_000000.parquet")
        #dataset.save_to_disk("benchmarks/dpbench-tableformer-dataset")

        test_path = str(ds_path / "test" / "*.parquet")
        train_path = str(ds_path / "train" / "*.parquet")
        
        test_files = glob.glob(test_path)
        train_files = glob.glob(train_path)
        logging.info(f"test-files: {test_files}, train-files: {train_files}")
        
        # Load all files into the `test`-`train` split
        ds = None
        if len(test_files)>0 and len(train_files)>0:
            ds = load_dataset("parquet", data_files={"test": test_files, "train": train_files})
        elif len(test_files)>0 and len(train_files)==0:
            ds = load_dataset("parquet", data_files={"test": test_files})
            
        logging.info(f"oveview of dataset: {ds}")
        
        table_evaluations = []
        #ds = datasets.load_from_disk(ds_path)
        ds = ds[split]
        for i, data in tqdm(enumerate(ds), desc="Table evaluations", ncols=120, total=len(ds)):
            #gt_doc_dict = data["GroundTruthDoclingDocument"]
            gt_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
            gt_doc = DoclingDocument.model_validate_json(gt_doc_dict)
            #pred_doc_dict = data["PredictedDoclingDocument"]
            pred_doc_dict = data[BenchMarkColumns.PREDICTION]
            pred_doc = DoclingDocument.model_validate_json(pred_doc_dict)

            results = self._evaluate_tables_in_documents(doc_id=data[BenchMarkColumns.DOC_ID],
                                                         gt_doc=gt_doc,
                                                         pred_doc=pred_doc)
            
            table_evaluations.extend(results)

        # Compute TED statistics for the entire dataset
        teds_simple = []
        teds_complex = []
        teds_all = []
        for te in table_evaluations:
            if te.is_complex:
                teds_complex.append(te.TEDS)
            else:
                teds_simple.append(te.TEDS)
            teds_all.append(te.TEDS)

        all_mean_TEDS = statistics.mean(teds_all) if len(teds_simple) > 0 else None
        simple_mean_TEDS = (
            statistics.mean(teds_simple) if len(teds_simple) > 0 else None
        )
        complex_mean_TEDS = (
            statistics.mean(teds_complex) if len(teds_complex) > 0 else None
        )
        all_std_TEDS = statistics.stdev(teds_all) if len(teds_simple) >= 2 else None
        simple_std_TEDS = (
            statistics.stdev(teds_simple) if len(teds_simple) >= 2 else None
        )
        complex_std_TEDS = (
            statistics.stdev(teds_complex) if len(teds_complex) >= 2 else None
        )

        dataset_evaluation = DatasetTableEvaluation(
            evaluations=table_evaluations,
            total_mean_TEDS=all_mean_TEDS,
            simple_mean_TEDS=simple_mean_TEDS,
            complex_mean_TEDS=complex_mean_TEDS,
            total_std_TEDS=all_std_TEDS,
            simple_std_TEDS=simple_std_TEDS,
            complex_std_TEDS=complex_std_TEDS,
        )
        return dataset_evaluation

    def _evaluate_tables_in_documents(
        self,
        doc_id: str,
        gt_doc: DoclingDocument,
        pred_doc: DoclingDocument,
        structure_only: bool = False,
    ) -> list[TableEvaluation]:
        r""" """
        table_evaluations = []
        gt_tables = gt_doc.tables
        pred_tables = pred_doc.tables

        assert len(gt_tables)==len(pred_tables), "len(gt_tables)!=len(pred_tables)"
        
        for table_id in range(len(gt_tables), len(pred_tables)):

            try:
                gt_table = gt_tables[table_id]
                is_complex = is_complex_table(gt_table)
                gt_html = gt_table.export_to_html()
                predicted_html = pred_tables[table_id].export_to_html()

                # Filter out tags that may be present in GT but not in prediction to avoid penalty
                for stopword in self._stopwords:
                    predicted_html = predicted_html.replace(stopword, "")
                for stopword in self._stopwords:
                    gt_html = gt_html.replace(stopword, "")

                gt_html_obj = html.fromstring(gt_html)
                predicted_html_obj = html.fromstring(predicted_html)
                teds = self._teds_scorer(gt_html_obj, predicted_html_obj, structure_only)
                teds = round(teds, 3)
                table_evaluation = TableEvaluation(TEDS=teds, is_complex=is_complex)
                table_evaluations.append(table_evaluation)
            except Exception as exc:
                logging.error(f"Table {table_id} from document {doc_id} could not be compared!")
                
        return table_evaluations

    # def _dump_full_table_html(self, image_filename: str, full_table_html: str):
    #     r"""
    #     Save the full_table_html as a file
    #     """
    #     Path(self._viz_dir).mkdir(parents=True, exist_ok=True)
    #     html_filename = "{}.html".format(Path(image_filename).stem)
    #     html_fn = os.path.join(self._viz_dir, html_filename)
    #     with open(html_fn, "w") as fd:
    #         fd.write(full_table_html)
