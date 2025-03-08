import logging
import os
from pathlib import Path

from docling.cli.main import OcrEngine

from docling_eval.benchmarks.constants import BenchMarkNames
from docling_eval.benchmarks.pixparse_idl_wds.create import create_pixparse_dataset
from docling_eval.benchmarks.pixparse_idl_wds.utils import Hyperscaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


def main():
    # idir = Path(f"./benchmarks/{BenchMarkNames.PIXPARSEIDL.value}-original")
    # odir = Path(f"./benchmarks/{BenchMarkNames.PIXPARSEIDL.value}-dataset")
    idir = Path("/custom-dataset/ground-truth")
    odir = Path("/custom-dataset/ground-truth")

    os.makedirs(odir, exist_ok=True)

    os.environ["AWS_ACCESS_KEY_ID"] = ""
    os.environ["AWS_SECRET_ACCESS_KEY"] = ""
    os.environ["GOOGLE_PROJECT_ID"] = ""
    os.environ["GOOGLE_PROCESSOR_ID"] = ""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

    os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"] = ""
    os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"] = ""

    log.info("Create the converted PIXPARSE dataset")

    # Process test split
    create_pixparse_dataset(
        name="pixparse",
        split="test",
        input_dir=idir,
        output_dir=odir,
        ocr_engine=OcrEngine.EASYOCR,
        reprocess=True,
    )

    create_pixparse_dataset(
        name="pixparse",
        split="test",
        input_dir=idir,
        output_dir=odir,
        hyperscaler=Hyperscaler.AWS,
        reprocess=False,
    )

    create_pixparse_dataset(
        name="pixparse",
        split="test",
        input_dir=idir,
        output_dir=odir,
        hyperscaler=Hyperscaler.GOOGLE,
        reprocess=False,
    )

    create_pixparse_dataset(
        name="pixparse",
        split="test",
        input_dir=idir,
        output_dir=odir,
        hyperscaler=Hyperscaler.AZURE,
        reprocess=False,
    )


if __name__ == "__main__":
    main()
