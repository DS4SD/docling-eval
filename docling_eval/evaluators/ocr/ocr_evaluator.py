import glob
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List

import evaluate
from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns  # type: ignore


class DatasetStatistics(BaseModel):
    mean: float
    median: float
    min: float
    max: float
    std: float
    count: int


class PageEvaluation(BaseModel):
    doc_id: str
    gd_text: str
    extracted_text: str
    cer: float
    source_file: str


class FileEvaluation(BaseModel):
    file_name: str
    evaluations: List[PageEvaluation]
    cer_stats: DatasetStatistics


class DatasetTextractEvaluation(BaseModel):
    file_evaluations: List[FileEvaluation]


def compute_stats(cers_list: List[float]) -> DatasetStatistics:
    """Compute statistics for CER scores"""
    result = {
        "mean": statistics.mean(cers_list) if cers_list else 0.0,
        "median": statistics.median(cers_list) if cers_list else 0.0,
        "min": min(cers_list) if cers_list else 0.0,
        "max": max(cers_list) if cers_list else 0.0,
        "std": statistics.stdev(cers_list) if len(cers_list) > 1 else 0.0,
        "count": len(cers_list),
    }

    return DatasetStatistics(**result)


def load_jsonl_dataset(jsonl_path: str) -> Dict[str, List[Dict]]:
    """Load dataset from JSONL file(s) and organize by source file"""
    data_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for file_path in glob.glob(jsonl_path):
        file_name = os.path.basename(file_path)
        data_by_file[file_name] = []

        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item["_source_file"] = file_path
                    data_by_file[file_name].append(item)

    return data_by_file


class OCREvaluator:
    def __init__(self):
        # https://huggingface.co/spaces/evaluate-metric/cer
        self._cer_eval = evaluate.load("cer")

    def __call__(
        self, input_ds_path: Path, output_path: Path, split: str = "test"
    ) -> DatasetTextractEvaluation:
        jsonl_files = str(input_ds_path / "*.jsonl")
        logging.info(f"Loading JSONL files from pattern: {jsonl_files}")

        dataset_by_file = load_jsonl_dataset(jsonl_files)
        if not dataset_by_file:
            raise ValueError(f"Failed to load dataset from {jsonl_files}")

        file_count = len(dataset_by_file)
        total_records = sum(len(records) for records in dataset_by_file.values())
        logging.info(f"Loaded {total_records} records from {file_count} JSONL files")

        file_evaluations = []

        # Process each file separately
        for file_name, dataset in dataset_by_file.items():
            logging.info(f"Processing file: {file_name} with {len(dataset)} records")

            file_evaluations_list = []
            file_cers_list = []

            for i, data in tqdm(
                enumerate(dataset),
                desc=f"Processing {file_name}",
                ncols=120,
                total=len(dataset),
            ):
                doc_id = data[BenchMarkColumns.DOC_ID]
                true_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
                pred_doc_dict = data[BenchMarkColumns.PREDICTION]

                true_doc = DoclingDocument.model_validate(true_doc_dict)
                pred_doc = DoclingDocument.model_validate(pred_doc_dict)

                # Extract text from documents
                gd_text = self._extract_text(true_doc.export_to_dict())
                extracted_text = self._extract_text(pred_doc.export_to_dict())

                if gd_text and extracted_text:
                    cer = self._compute_cer_score(gd_text, extracted_text)
                else:
                    cer = 1.0  # max error when text is missing

                file_cers_list.append(cer)

                text_evaluation = PageEvaluation(
                    doc_id=doc_id,
                    gd_text=gd_text,
                    extracted_text=extracted_text,
                    cer=cer,
                    source_file=file_name,
                )

                file_evaluations_list.append(text_evaluation)
                logging.debug(f"File: {file_name}, Document {doc_id} CER: {cer:.4f}")

            # Compute statistics for this file
            file_cer_stats = compute_stats(file_cers_list)

            file_evaluation = FileEvaluation(
                file_name=file_name,
                evaluations=file_evaluations_list,
                cer_stats=file_cer_stats,
            )

            file_evaluations.append(file_evaluation)

            logging.debug(f"\n{file_name} CER Statistics:")
            logging.debug(f"  - Mean CER: {file_cer_stats.mean:.4f}")
            logging.debug(f"  - Median CER: {file_cer_stats.median:.4f}")
            logging.debug(f"  - Min CER: {file_cer_stats.min:.4f}")
            logging.debug(f"  - Max CER: {file_cer_stats.max:.4f}")
            logging.debug(f"  - Std Dev: {file_cer_stats.std:.4f}")
            logging.debug(f"  - Count: {file_cer_stats.count}")

        output_path.mkdir(parents=True, exist_ok=True)

        results_path = output_path / "evaluation_results.json"
        with open(results_path, "w") as f:
            results_data = {
                "file_evaluations": [
                    {
                        "file_name": eval_item.file_name,
                        "statistics": eval_item.cer_stats.model_dump(),
                        "evaluations": [e.model_dump() for e in eval_item.evaluations],
                    }
                    for eval_item in file_evaluations
                ]
            }
            json.dump(results_data, f, indent=2)

        logging.info(f"Saved evaluation results to {results_path}")

        chart_path = output_path / "ocr_comparison_chart.png"
        self._create_comparison_chart(file_evaluations, chart_path)

        return DatasetTextractEvaluation(file_evaluations=file_evaluations)

    def _compute_cer_score(self, true_txt: str, pred_txt: str):
        """Compute CER score with the HF evaluate and the default Tokenizer"""
        # separator = "-" * 40

        # logging.debug(f"\n{separator}\nPredicted Text:\n{pred_txt}\n{separator}\n")
        # logging.debug(f"\n{separator}\nTrue Text:\n{true_txt}\n{separator}\n")
        result = self._cer_eval.compute(predictions=[pred_txt], references=[true_txt])

        return result

    def _extract_text(self, json_data: Dict[str, Any]) -> str:
        """Extract text from document JSON structure"""
        extracted_text = ""
        if "texts" in json_data:
            for text_item in json_data["texts"]:
                if "text" in text_item:
                    extracted_text += text_item["text"] + " "
        return extracted_text.strip()

    def _create_comparison_chart(
        self, file_evaluations: List[FileEvaluation], output_path: Path
    ):
        """Create a bar chart comparing CER across different OCR engines"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            engines = []
            mean_cers = []
            median_cers = []

            for eval_item in file_evaluations:
                engine_name = eval_item.file_name.split("_")[0]  # Extract engine name
                engines.append(engine_name)
                mean_cers.append(eval_item.cer_stats.mean)
                median_cers.append(eval_item.cer_stats.median)

            # Set up bar chart
            x = np.arange(len(engines))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width / 2, mean_cers, width, label="Mean CER")
            ax.bar(x + width / 2, median_cers, width, label="Median CER")

            ax.set_xlabel("OCR Engine")
            ax.set_ylabel("Character Error Rate (CER)")
            ax.set_title("OCR Engine Comparison - Lower is Better")
            ax.set_xticks(x)
            ax.set_xticklabels(engines)
            ax.legend()

            for i, v in enumerate(mean_cers):
                ax.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center")

            for i, v in enumerate(median_cers):
                ax.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center")

            plt.tight_layout()
            plt.savefig(output_path)
            logging.info(f"Saved comparison chart to {output_path}")

        except ImportError:
            logging.warning(
                "Could not create comparison chart: matplotlib is not installed"
            )
