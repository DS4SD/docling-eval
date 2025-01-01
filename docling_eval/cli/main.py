import json
import logging
import os
from enum import Enum, auto
from pathlib import Path
from typing import Annotated, Optional

from tabulate import tabulate

import matplotlib.pyplot as plt
import typer

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.benchmarks.dpbench.create import (
    create_dpbench_e2e_dataset,
    create_dpbench_tableformer_dataset,
)
from docling_eval.evaluators.layout_evaluator import (
    DatasetLayoutEvaluation,
    LayoutEvaluator,
)
from docling_eval.evaluators.table_evaluator import (
    DatasetTableEvaluation,
    TableEvaluator,
)

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
    VISUALIZE = "visualize"


def create(
    modality: EvaluationModality,
    benchmark: BenchMarkNames,
    idir: Path,
    odir: Path,
    image_scale: float = 1.0,
):
    r""""""
    if not os.path.exists(idir):
        log.error(f"Benchmark directory not found: {idir}")
        return

    if odir is None:
        odir = Path("./benchmarks") / benchmark.value / modality.value

    if benchmark == BenchMarkNames.DPBENCH:

        if (
            modality == EvaluationModality.END2END
            or modality == EvaluationModality.LAYOUT
        ):
            create_dpbench_e2e_dataset(
                dpbench_dir=idir, output_dir=odir, image_scale=image_scale
            )

        elif modality == EvaluationModality.TABLEFORMER:
            create_dpbench_tableformer_dataset(
                dpbench_dir=idir, output_dir=odir, image_scale=image_scale
            )

        else:
            log.error(f"{modality} is not yet implemented for {benchmark}")

    else:
        log.error(f"{benchmark} is not yet implemented")


def evaluate(
    modality: EvaluationModality, benchmark: BenchMarkNames, idir: Path, odir: Path
):
    r""""""
    if not os.path.exists(idir):
        log.error(f"Benchmark directory not found: {idir}")

    # Save the evaluation
    save_fn = odir / f"evaluation_{benchmark.value}_{modality.value}.json"

    if modality == EvaluationModality.END2END:
        logging.error("not supported")

    elif modality == EvaluationModality.LAYOUT:
        layout_evaluator = LayoutEvaluator()
        layout_evaluation = layout_evaluator(idir, split="test")

        with open(save_fn, "w") as fd:
            json.dump(layout_evaluation.model_dump(), fd, indent=2, sort_keys=True)

    elif modality == EvaluationModality.TABLEFORMER:
        table_evaluator = TableEvaluator()
        table_evaluation = table_evaluator(idir, split="test")

        with open(save_fn, "w") as fd:
            json.dump(table_evaluation.model_dump(), fd, indent=2, sort_keys=True)

    elif modality == EvaluationModality.CODEFORMER:
        pass

    log.info("The evaluation has been saved in '%s'", save_fn)


def visualise(
    modality: EvaluationModality, benchmark: BenchMarkNames, idir: Path, odir: Path
):

    filename = odir / f"evaluation_{benchmark.value}_{modality.value}.json"

    if modality == EvaluationModality.END2END:
        pass

    elif modality == EvaluationModality.LAYOUT:
        with open(filename, "r") as fd:
            evaluation = DatasetLayoutEvaluation.parse_file(filename)

        table, headers = evaluation.to_table()
        
        logging.info("Class mAP[0.5:0.95] table: \n\n"+tabulate(table, headers=headers, tablefmt="github"))

    elif modality == EvaluationModality.TABLEFORMER:

        with open(filename, "r") as fd:
            evaluation = DatasetTableEvaluation.parse_file(filename)

        table, headers = evaluation.to_table()        
        logging.info("TEDS table: \n\n"+tabulate(table, headers=headers, tablefmt="github"))
            
        figname = odir / f"evaluation_{benchmark.value}_{modality.value}.png"
        logging.info(f"saving figure to {figname}")
        
        evaluation.save_histogram(figname)


    elif modality == EvaluationModality.CODEFORMER:
        pass


@app.command(no_args_is_help=True)
def main(
    task: Annotated[
        EvaluationTask,
        typer.Option(
            ...,  # EvaluationTask.CREATE,
            "-t",  # Short name
            "--task",  # Long name
            help="Evaluation task",
        ),
    ],
    modality: Annotated[
        EvaluationModality,
        typer.Option(
            ...,  # EvaluationModality.TABLEFORMER,
            "-m",  # Short name
            "--modality",  # Long name
            help="Evaluation modality",
        ),
    ],
    benchmark: Annotated[
        BenchMarkNames,
        typer.Option(
            ...,  # BenchMarkNames.DPBENCH,
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

    elif task == EvaluationTask.VISUALIZE:
        visualise(modality, benchmark, idir, odir)

    else:
        log.error("Unsupported command: '%s'", task.value)


if __name__ == "__main__":
    app()
