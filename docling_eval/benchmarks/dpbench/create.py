import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict

from tqdm import tqdm  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from bs4 import BeautifulSoup  # type: ignore
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    ProvenanceItem,
    TableCell,
    TableData,
)
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image  # as PILImage

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.utils import (
    add_pages_to_true_doc,
    convert_html_table_into_docling_tabledata,
    save_comparison_html,
    save_comparison_html_with_clusters,
    write_datasets_info,
)
from docling_eval.docling.conversion import create_converter
from docling_eval.docling.models.tableformer.tf_model_prediction import (
    TableFormerUpdater,
)
from docling_eval.docling.utils import (
    crop_bounding_box,
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    get_binary,
    save_shard_to_disk,
)

TRUE_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    # Additional
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}

PRED_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    # Additional
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}


def update(
    doc: DoclingDocument,
    annots: Dict,
    page,
    page_image: Image.Image,
    page_width: float,
    page_height: float,
):

    label = annots["category"]

    min_x = annots["coordinates"][0]["x"]
    max_x = annots["coordinates"][0]["x"]

    min_y = annots["coordinates"][0]["y"]
    max_y = annots["coordinates"][0]["y"]

    for coor in annots["coordinates"]:
        min_x = min(min_x, coor["x"])
        max_x = max(max_x, coor["x"])

        min_y = min(min_y, coor["y"])
        max_y = max(max_y, coor["y"])

    text = annots["content"]["text"].replace("\n", " ")
    html = annots["content"]["html"]

    bbox = BoundingBox(
        l=min_x * page_width,
        r=max_x * page_width,
        b=min_y * page_height,
        t=max_y * page_height,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(text)))

    img = crop_bounding_box(page_image=page_image, page=page, bbox=bbox)

    if label == "Header":
        doc.add_text(label=DocItemLabel.PAGE_HEADER, text=text, orig=text, prov=prov)

    elif label == "Footer":
        doc.add_text(label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov)

    elif label == "Paragraph":
        doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

    elif label == "Index":

        # FIXME: ultra approximate solution
        text = annots["content"]["text"]
        rows = text.split("\n")

        num_rows = len(rows)
        num_cols = 2

        row_span = 1
        col_span = 1

        cells = []
        for row_idx, row in enumerate(rows):

            parts = row.split(" ")

            col_idx = 0
            cell = TableCell(
                row_span=row_span,
                col_span=col_span,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + row_span,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + col_span,
                text=" ".join(parts[:-1]),
            )
            cells.append(cell)

            col_idx = 1
            cell = TableCell(
                row_span=row_span,
                col_span=col_span,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + row_span,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + col_span,
                text=parts[-1],
            )
            cells.append(cell)

        table_data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)
        doc.add_table(
            data=table_data, caption=None, prov=prov, label=DocItemLabel.DOCUMENT_INDEX
        )

    elif label == "List":
        doc.add_list_item(text=text, orig=text, prov=prov)
        # doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

    elif label == "Caption":
        doc.add_text(label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov)

    elif label == "Equation":
        doc.add_text(label=DocItemLabel.FORMULA, text=text, orig=text, prov=prov)

    elif label == "Figure":
        uri = from_pil_to_base64uri(img)

        imgref = ImageRef(
            mimetype="image/png",
            dpi=72,
            size=Size(width=img.width, height=img.height),
            uri=uri,
        )

        doc.add_picture(prov=prov, image=imgref)

    elif label == "Table":

        table_data = convert_html_table_into_docling_tabledata(table_html=html)

        doc.add_table(data=table_data, caption=None, prov=prov)

    elif label == "Chart":
        uri = from_pil_to_base64uri(img)

        imgref = ImageRef(
            mimetype="image/png",
            dpi=72,
            size=Size(width=img.width, height=img.height),
            uri=uri,
        )

        doc.add_picture(prov=prov, image=imgref)

        # doc.add_picture(prov=prov)

    elif label == "Footnote":
        doc.add_text(label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov)

    elif label == "Heading1":
        doc.add_heading(text=text, orig=text, level=1, prov=prov)

    else:
        return


def create_dpbench_e2e_dataset(
    dpbench_dir: Path, output_dir: Path, image_scale: float = 1.0
):

    # Create Converter
    doc_converter = create_converter(page_image_scale=image_scale)

    # load the groundtruth
    with open(dpbench_dir / f"dataset/reference.json", "r") as fr:
        gt = json.load(fr)

    viz_dir = output_dir / "vizualisations"
    os.makedirs(viz_dir, exist_ok=True)

    records = []

    for filename, annots in tqdm(
        gt.items(),
        desc="Processing files for DP-Bench with end-to-end",
        total=len(gt),
        ncols=128,
    ):

        pdf_path = dpbench_dir / f"dataset/pdfs/{filename}"

        # Create the predicted Document
        conv_results = doc_converter.convert(source=pdf_path, raises_on_error=True)
        pred_doc = conv_results.document

        # Create the groundtruth Document
        true_doc = DoclingDocument(name=f"ground-truth {os.path.basename(pdf_path)}")
        true_doc, true_page_images = add_pages_to_true_doc(
            pdf_path=pdf_path, true_doc=true_doc, image_scale=image_scale
        )

        assert len(true_page_images) == 1, "len(true_page_images)==1"

        page_width = true_doc.pages[1].size.width
        page_height = true_doc.pages[1].size.height

        for elem in annots["elements"]:
            update(
                true_doc,
                elem,
                page=true_doc.pages[1],
                page_image=true_page_images[0],
                page_width=page_width,
                page_height=page_height,
            )

        if True:
            """
            save_comparison_html(
                filename=viz_dir / f"{os.path.basename(pdf_path)}-comp.html",
                true_doc=true_doc,
                pred_doc=pred_doc,
                page_image=true_page_images[0],
                true_labels=TRUE_HTML_EXPORT_LABELS,
                pred_labels=PRED_HTML_EXPORT_LABELS,
            )
            """

            save_comparison_html_with_clusters(
                filename=viz_dir / f"{os.path.basename(pdf_path)}-clusters.html",
                true_doc=true_doc,
                pred_doc=pred_doc,
                page_image=true_page_images[0],
                true_labels=TRUE_HTML_EXPORT_LABELS,
                pred_labels=PRED_HTML_EXPORT_LABELS,
            )

        pred_doc, pictures, page_images = extract_images(
            pred_doc,
            pictures_column=BenchMarkColumns.PICTURES.value,  # pictures_column,
            page_images_column=BenchMarkColumns.PAGE_IMAGES.value,  # page_images_column,
        )

        record = {
            BenchMarkColumns.DOCLING_VERSION: docling_version(),
            BenchMarkColumns.STATUS: str(conv_results.status),
            BenchMarkColumns.DOC_ID: str(filename),
            BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
            BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
            BenchMarkColumns.ORIGINAL: get_binary(pdf_path),
            BenchMarkColumns.MIMETYPE: "application/pdf",
            BenchMarkColumns.PAGE_IMAGES: page_images,
            BenchMarkColumns.PICTURES: pictures,
        }
        records.append(record)

    test_dir = output_dir / "test"
    os.makedirs(test_dir, exist_ok=True)

    save_shard_to_disk(items=records, dataset_path=test_dir)

    write_datasets_info(
        name="DPBench: end-to-end",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )


def create_dpbench_layout_dataset(
    dpbench_dir: Path, output_dir: Path, image_scale: float = 1.0
):
    create_dpbench_e2e_dataset(
        dpbench_dir=dpbench_dir, output_dir=output_dir, image_scale=image_scale
    )


def create_dpbench_tableformer_dataset(
    dpbench_dir: Path, output_dir: Path, image_scale: float = 1.0
):
    # Init the TableFormer model
    tf_updater = TableFormerUpdater()

    # load the groundtruth
    with open(dpbench_dir / f"dataset/reference.json", "r") as fr:
        gt = json.load(fr)

    viz_dir = output_dir / "vizualisations"
    os.makedirs(viz_dir, exist_ok=True)

    records = []

    for filename, annots in tqdm(
        gt.items(),
        desc="Processing files for DP-Bench with TableFormer",
        total=len(gt),
        ncols=128,
    ):

        pdf_path = dpbench_dir / f"dataset/pdfs/{filename}"

        # Create the groundtruth Document
        true_doc = DoclingDocument(name=f"ground-truth {os.path.basename(pdf_path)}")
        true_doc, true_page_images = add_pages_to_true_doc(
            pdf_path=pdf_path, true_doc=true_doc, image_scale=image_scale
        )

        assert len(true_page_images) == 1, "len(true_page_images)==1"

        page_width = true_doc.pages[1].size.width
        page_height = true_doc.pages[1].size.height

        for elem in annots["elements"]:
            update(
                true_doc,
                elem,
                page=true_doc.pages[1],
                page_image=true_page_images[0],
                page_width=page_width,
                page_height=page_height,
            )

        # Create the updated Document
        updated, pred_doc = tf_updater.replace_tabledata(
            pdf_path=pdf_path, true_doc=true_doc, true_page_images=true_page_images
        )

        if updated:

            if True:
                save_comparison_html(
                    filename=viz_dir / f"{os.path.basename(pdf_path)}-comp.html",
                    true_doc=true_doc,
                    pred_doc=pred_doc,
                    page_image=true_page_images[0],
                    true_labels=TRUE_HTML_EXPORT_LABELS,
                    pred_labels=PRED_HTML_EXPORT_LABELS,
                )

            record = {
                BenchMarkColumns.DOCLING_VERSION: docling_version(),
                BenchMarkColumns.STATUS: "SUCCESS",
                BenchMarkColumns.DOC_ID: str(os.path.basename(pdf_path)),
                BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
                BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
                BenchMarkColumns.ORIGINAL: get_binary(pdf_path),
                BenchMarkColumns.MIMETYPE: "application/pdf",
                BenchMarkColumns.PAGE_IMAGES: true_page_images,
                BenchMarkColumns.PICTURES: [],  # pred_pictures,
            }
            records.append(record)

    test_dir = output_dir / "test"
    os.makedirs(test_dir, exist_ok=True)

    save_shard_to_disk(items=records, dataset_path=test_dir)

    write_datasets_info(
        name="DPBench: tableformer",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )


def parse_arguments():
    """Parse arguments for DP-Bench parsing."""

    parser = argparse.ArgumentParser(
        description="Process DP-Bench benchmark from directory into HF dataset."
    )
    parser.add_argument(
        "-i",
        "--dpbench-directory",
        help="input directory with documents",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="output directory with shards",
        required=False,
        default="./benchmarks/dpbench",
    )
    parser.add_argument(
        "-s",
        "--image-scale",
        help="image-scale of the pages",
        required=False,
        default=1.0,
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="mode of dataset",
        required=False,
        choices=["end-2-end", "table", "formula", "all"],
    )
    args = parser.parse_args()

    return (
        Path(args.dpbench_directory),
        Path(args.output_directory),
        float(args.image_scale),
        args.mode,
    )


def main():

    dpbench_dir, output_dir, image_scale, mode = parse_arguments()

    # Create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    odir_e2e = Path(output_dir) / "end_to_end"
    odir_tab = Path(output_dir) / "tables"
    odir_eqn = Path(output_dir) / "formulas"

    os.makedirs(odir_e2e, exist_ok=True)
    os.makedirs(odir_tab, exist_ok=True)
    # os.makedirs(odir_eqn, exist_ok=True)

    for _ in ["test", "train"]:
        os.makedirs(odir_e2e / _, exist_ok=True)
        os.makedirs(odir_tab / _, exist_ok=True)

    if mode == "end-2-end":
        create_dpbench_e2e_dataset(
            dpbench_dir=dpbench_dir, output_dir=odir_e2e, image_scale=image_scale
        )

    elif mode == "table":
        create_dpbench_tableformer_dataset(
            dpbench_dir=dpbench_dir, output_dir=odir_tab, image_scale=image_scale
        )

    elif mode == "all":
        create_dpbench_e2e_dataset(
            dpbench_dir=dpbench_dir, output_dir=odir_e2e, image_scale=image_scale
        )

        create_dpbench_tableformer_dataset(
            dpbench_dir=dpbench_dir, output_dir=odir_tab, image_scale=image_scale
        )


if __name__ == "__main__":
    main()
