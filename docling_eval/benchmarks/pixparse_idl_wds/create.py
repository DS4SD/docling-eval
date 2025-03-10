import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from docling.cli.main import OcrEngine
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    PageItem,
    ProvenanceItem,
    Size,
)
from PIL import Image
from tqdm import tqdm

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.pixparse_idl_wds.hyperscaler_clients import (
    initialize_azure_document_intelligence_client,
    initialize_google_doc_ai_client,
    initialize_textract_client,
    process_with_azure,
    process_with_google,
    process_with_textract,
)
from docling_eval.benchmarks.pixparse_idl_wds.utils import (
    CustomJSONEncoder,
    Hyperscaler,
    check_service_env_vars,
    read_image_content,
    write_dataset_info,
    write_json_output,
)
from docling_eval.converters.conversion import create_image_docling_converter
from docling_eval.converters.hyperscalers import (
    convert_azure_output_to_docling,
    convert_google_output_to_docling,
    convert_textract_output_to_docling,
)


def process_with_service(
    image_file: Path,
    doc_id: str,
    input_dir: Path,
    output_dir: Path,
    service: Union[Hyperscaler, OcrEngine],
) -> Dict:
    """Process an image with a specific service and return standardized record."""
    service_name = service.value

    if not check_service_env_vars(service):
        logging.error(
            f"Cannot process with {service.name} due to missing environment variables"
        )
        raise EnvironmentError(
            f"Missing required environment variables for {service.name}"
        )

    image_content_bytes = read_image_content(image_file)
    if isinstance(service, Hyperscaler):
        clients = initialize_hyperscaler_client(service)
        output_raw_path = input_dir / doc_id / f"{service_name}_raw.json"

        if service == Hyperscaler.AWS:
            aws_result: Optional[dict] = process_with_textract(
                clients["textract"], image_content_bytes
            )
            if aws_result is not None:
                write_json_output(output_raw_path, aws_result)
                converted_result = convert_textract_output_to_docling(
                    aws_result, image_file
                )
            else:
                logging.warning("AWS Textract processing returned None result")
        elif service == Hyperscaler.GOOGLE:
            google_result: Optional[dict] = process_with_google(
                clients["google"], clients["google_processor_name"], image_content_bytes
            )
            if google_result is not None:
                write_json_output(output_raw_path, google_result)
                converted_result = convert_google_output_to_docling(
                    google_result, image_file
                )
            else:
                logging.warning("Google processing returned None result")
        elif service == Hyperscaler.AZURE:
            azure_result: Optional[dict] = process_with_azure(
                clients["azure"], image_content_bytes
            )
            if azure_result is not None:
                write_json_output(output_raw_path, azure_result)
                converted_result = convert_azure_output_to_docling(
                    azure_result, image_file
                )
            else:
                logging.warning("Azure processing returned None result")
    elif isinstance(service, OcrEngine):
        docling_ocr_doc_converter = create_image_docling_converter(
            do_ocr=True, ocr_engine=service, ocr_lang=["en"]
        )
        converted_result = docling_ocr_doc_converter.convert(image_file).document

    converted_results_dir = input_dir / doc_id / "converted_results"
    converted_results_dir.mkdir(parents=True, exist_ok=True)
    converted_path = converted_results_dir / f"{doc_id}_{service_name}_converted.json"

    with open(converted_path, "w") as f:
        json.dump(converted_result.export_to_dict(), f, indent=2)

    with open(input_dir / doc_id / "ground_truth.json", "r") as f:
        gt_data = json.load(f)

    true_doc = create_docling_document(doc_id, gt_data, image_file)
    pred_doc = converted_result  # Already in docling document format.

    record = {
        BenchMarkColumns.DOC_ID: doc_id,
        BenchMarkColumns.GROUNDTRUTH: true_doc.export_to_dict(),
        BenchMarkColumns.PREDICTION: pred_doc.export_to_dict(),
    }

    return record


def save_service_shard(
    items: List[Dict], output_dir: Path, service_name: str, shard_id: int
) -> None:
    """Save a shard of service predictions to disk."""
    shard_path = output_dir / f"{service_name}_shard_{shard_id:05d}.jsonl"

    with open(shard_path, "w") as f:
        for item in items:
            json_str = json.dumps(item, cls=CustomJSONEncoder)
            f.write(json_str + "\n")


def initialize_hyperscaler_client(hyperscaler: Hyperscaler) -> Dict[str, Any]:
    """Initialize a specific hyperscaler client."""
    clients = {}

    if hyperscaler == Hyperscaler.AWS:
        clients["textract"] = initialize_textract_client()
    elif hyperscaler == Hyperscaler.GOOGLE:
        clients["google"], clients["google_processor_name"] = (
            initialize_google_doc_ai_client()
        )
    elif hyperscaler == Hyperscaler.AZURE:
        clients["azure"] = initialize_azure_document_intelligence_client()

    return clients


def create_docling_document(
    doc_id: str, gt_data: Dict, image_file: Path
) -> DoclingDocument:
    """Create a DoclingDocument from ground truth data and image file."""
    w, h = Image.open(image_file).size

    true_doc = DoclingDocument(name=doc_id)
    true_doc.pages[1] = PageItem(size=Size(width=float(w), height=float(h)), page_no=1)

    for page_idx, page in enumerate(gt_data["pages"], 1):
        for text, bbox, _ in zip(page["text"], page["bbox"], page["score"]):
            bbox_obj = BoundingBox.from_tuple(
                (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[0] + bbox[2]),
                    float(bbox[1] + bbox[3]),
                ),
                CoordOrigin.TOPLEFT,
            )
            prov = ProvenanceItem(
                page_no=page_idx, bbox=bbox_obj, charspan=(0, len(text))
            )
            true_doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)

    return true_doc


def create_pixparse_dataset(
    name: str,
    split: str,
    input_dir: Path,
    output_dir: Path,
    do_viz: bool = False,
    max_items: int = -1,
    ocr_engine: Optional[OcrEngine] = None,
    hyperscaler: Optional[Hyperscaler] = None,
    reprocess: bool = False,
) -> None:
    """Create datasets from input files with specified processing options."""
    # TODO: Upload the dataset to Huggingface and then use this code to create the dataset
    # ds = load_dataset(name, split=split)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization directory if needed
    if do_viz:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Find all ground truth files
    ground_truth_files = list(input_dir.rglob("ground_truth.json"))
    if max_items > 0:
        ground_truth_files = ground_truth_files[:max_items]

    # Determine which services to process
    services_to_process: list[Union[Hyperscaler, OcrEngine]] = []

    if hyperscaler:
        services_to_process.append(hyperscaler)
    if ocr_engine:
        services_to_process.append(ocr_engine)

    # Default to all hyperscalers only if no specific service is provided
    if not services_to_process:
        services_to_process.extend([h for h in Hyperscaler])

    # Track records by service
    service_records: dict[str, list] = {s.value: [] for s in services_to_process}
    service_counts = {s.value: 0 for s in services_to_process}
    logging.info(services_to_process)
    for gt_file in tqdm(ground_truth_files):
        try:
            image_file = gt_file.parent / "original.tif"
            if not image_file.exists():
                logging.info(f"Warning: No image file found for {gt_file}")
                continue

            doc_id = gt_file.parent.name

            # Process with each requested service
            for service in services_to_process:
                try:
                    service_name = service.value

                    if reprocess:
                        record = process_with_service(
                            image_file, doc_id, input_dir, output_dir, service
                        )
                    else:
                        # load existing converted or raw result if available
                        converted_path = (
                            input_dir
                            / doc_id
                            / "converted_results"
                            / f"{doc_id}_{service_name}_converted.json"
                        )

                        with open(gt_file, "r") as f:
                            gt_data = json.load(f)

                        true_doc = create_docling_document(doc_id, gt_data, image_file)

                        # if no converted files exist, load and use the raw outputs
                        if converted_path.exists():
                            pred_doc = DoclingDocument.load_from_json(converted_path)
                        else:
                            logging.info(
                                f"No converted result found for {doc_id} with {service_name}, loading raw files"
                            )
                            output_raw_path = (
                                input_dir / doc_id / f"{service_name}_raw.json"
                            )
                            with open(output_raw_path, "r") as f:
                                result = json.load(f)

                            converters: Dict[
                                Hyperscaler, Callable[[Any, Path], DoclingDocument]
                            ] = {
                                Hyperscaler.AZURE: convert_azure_output_to_docling,
                                Hyperscaler.AWS: convert_textract_output_to_docling,
                                Hyperscaler.GOOGLE: convert_google_output_to_docling,
                            }

                            pred_doc = converters.get(service, lambda *args: None)(  # type: ignore
                                result, image_file
                            )
                            if pred_doc:
                                converted_results_dir = (
                                    input_dir / doc_id / "converted_results"
                                )
                                converted_results_dir.mkdir(parents=True, exist_ok=True)

                                converted_path = (
                                    converted_results_dir
                                    / f"{doc_id}_{service_name}_converted.json"
                                )

                                with open(converted_path, "w") as f:
                                    json.dump(pred_doc.export_to_dict(), f, indent=2)

                        record = {
                            BenchMarkColumns.DOC_ID: doc_id,
                            BenchMarkColumns.GROUNDTRUTH: true_doc.export_to_dict(),
                            BenchMarkColumns.PREDICTION: pred_doc.export_to_dict(),
                        }

                    service_records[service_name].append(record)
                    service_counts[service_name] += 1

                    # Save periodically
                    if service_counts[service_name] % 1000 == 0:
                        shard_id = service_counts[service_name] // 1000 - 1
                        save_service_shard(
                            service_records[service_name],
                            output_dir,
                            service_name,
                            shard_id,
                        )
                        service_records[service_name] = []

                except Exception as e:
                    logging.info(
                        f"Error processing {doc_id} with {service.value}: {str(e)}"
                    )
                    raise

            # Generate visualization if requested
            if do_viz:
                viz_path = viz_dir / f"{doc_id}_viz.html"
                # TODO: Add visualization code

        except Exception as e:
            logging.info(f"Error processing {gt_file}: {str(e)}")
            raise

    # Save any remaining records
    for service in services_to_process:
        service_name = service.value
        if service_records[service_name]:
            shard_id = service_counts[service_name] // 1000
            save_service_shard(
                service_records[service_name], output_dir, service_name, shard_id
            )

    # Write dataset info
    total_records = sum(service_counts.values())
    write_dataset_info(
        name=f"{name} Dataset",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=total_records,
    )
