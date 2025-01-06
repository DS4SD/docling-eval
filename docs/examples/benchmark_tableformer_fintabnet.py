import logging
import os
from pathlib import Path

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.benchmarks.tableformer_huggingface_otsl.create import (
    create_fintabnet_tableformer_dataset,
)
from docling_eval.cli.main import evaluate, visualise

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():

    benchmark = BenchMarkNames.FINTABNET
    
    odir = Path(f"./benchmarks/{BenchMarkNames.FINTABNET.value}-dataset")

    odir_tab = Path(odir) / "tableformer"

    for _ in [odir, odir_tab]:
        os.makedirs(_, exist_ok=True)

    if True:
        create_fintabnet_tableformer_dataset(output_dir=odir_tab, max_items=1000, do_viz=True)

        evaluate(
            modality=EvaluationModality.TABLEFORMER,
            benchmark=benchmark,
            idir=odir_tab,
            odir=odir_tab,
        )
        visualise(
            modality=EvaluationModality.TABLEFORMER,
            benchmark=benchmark,
            idir=odir_tab,
            odir=odir_tab,
        )


if __name__ == "__main__":
    main()
