import os
import json
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Annotated, Optional

import typer

from docling_eval.benchmarks.constants import BenchMarkNames

from docling_eval.benchmarks.dpbench.create import create_dpbench_e2e_dataset, create_dpbench_tableformer_dataset

from docling_eval.evaluators.table_evaluator import TableEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

app = typer.Typer(
    name="docling-eval",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


class EvaluationTask(str, Enum):
    CREATE = "create"
    EVALUATE = "evaluate"

class EvaluationModality(str, Enum):
    END2END = "end-to-end"
    LAYOUT = "layout"
    TABLEFORMER = "tableformer"
    CODEFORMER = "codeformer"


def create(modality:EvaluationModality, benchmark:BenchMarkNames, idir:Path, odir:Path=None, image_scale:float=1.0):
    r""""""
    if not os.path.exists(idir):
        log.error(f"Benchmark directory not found: {idir}")
        return

    if odir is None:
        odir = Path("./benchmarks") / benchmark.value / modality.value
        
    match benchmark:
        case BenchMarkNames.DPBENCH:
            if(modality==EvaluationModality.END2END or
               modality==EvaluationModality.LAYOUT):
                create_dpbench_e2e_dataset(dpbench_dir=idir, output_dir=odir, image_scale=image_scale)
            elif(modality==EvaluationModality.TABLEFORMER):
                create_dpbench_tableformer_dataset(dpbench_dir=idir, output_dir=odir, image_scale=image_scale)
            else:
                log.error(f"{modality} is not yet implemented for {benchmark}")

        case _:
            log.error(f"{benchmark} is not yet implemented")


def evaluate(modality:EvaluationModality, benchmark:BenchMarkNames, idir:Path, odir:Path):
    r""""""
    if not os.path.exists(idir):
        log.error(f"Benchmark directory not found: {idir}")
    
    match modality:
        case EvaluationModality.END2END:
            pass
        
        case EvaluationModality.LAYOUT:
            pass

        case EvaluationModality.TABLEFORMER:
            table_evaluator = TableEvaluator()
            ds_evaluation = table_evaluator(idir, split="test")
            
        case EvaluationModality.CODEFORMER:
            pass

    # Save the evaluation
    save_fn = odir / f"evaluation_{benchmark}_{modality}.json"
    with open(save_fn, "w") as fd:
        json.dump(ds_evaluation.model_dump(), fd, indent=2, sort_keys=True)
        _log.info("The evaluation has been saved in '%s'", save_fn)
        
    """
    ds_evaluation = None
    if evaluation_task == EvaluationTask.TABLES:
        table_evaluator = TableEvaluator()
        ds_evaluation = table_evaluator(ds_path, split)
    else:
        _log.info("Unsupported evaluation task")

    if ds_evaluation is None:
        _log.error("No evaluation has been produced")
        return

    # Save the evaluation
    save_fn = save_path / "evaluation.json"
    with open(save_fn, "w") as fd:
        json.dump(ds_evaluation.model_dump(), fd, indent=2, sort_keys=True)
        _log.info("The evaluation has been saved in '%s'", save_fn)
    """

@app.command(no_args_is_help=True)
def main(
    task: Annotated[
        EvaluationTask,
        typer.Option(
            ..., #EvaluationTask.CREATE,
            "-t",  # Short name
            "--task",  # Long name
            help="Evaluation task",
        ),
    ],
    modality: Annotated[
        EvaluationModality,
        typer.Option(
            ..., #EvaluationModality.TABLEFORMER,
            "-m",  # Short name
            "--modality",  # Long name
            help="Evaluation modality",
        ),
    ],        
    benchmark: Annotated[
        BenchMarkNames,
        typer.Option(
            ..., #BenchMarkNames.DPBENCH,
            "-b",  # Short name
            "--benchmark",  # Long name
            help="Benchmark name",
        ),
    ],        
    idir: Annotated[
        Path,
        typer.Option(
            ...,
            "-i",  # Short name
            "--input-dir",  # Long name
            help="Input directory",
        ),
    ],
    odir: Annotated[
        Path,
        typer.Option(
            ...,
            "-o",  # Short name
            "--output-dir",  # Long name
            help="Output directory",
        ),
    ],        
):
    # Dispatch the command
    if task == EvaluationTask.CREATE:
        create(modality, benchmark, idir, odir)

    elif task == EvaluationTask.EVALUATE:
        evaluate(modality, benchmark, idir, odir)

    else:
        _log.error("Unsupported command: '%s'", command)


if __name__ == "__main__":
    app()
