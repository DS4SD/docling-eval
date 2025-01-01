import copy
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import docling_ibm_models.tableformer.data_management.tf_predictor as tf_predictor
import numpy as np
import pytest
import torch
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    PageItem,
    ProvenanceItem,
    TableCell,
    TableData,
    TableItem,
)
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler
from docling_parse.pdf_parsers import pdf_parser_v2
from huggingface_hub import snapshot_download

# import cv2
from PIL import Image, ImageDraw

from docling_eval.docling.models.tableformer.tf_constants import tf_config
from docling_eval.docling.utils import crop_bounding_box, map_to_records

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def init_tf_model() -> dict:
    r"""
    Initialize the testing environment
    """
    config: Dict[str, Any] = tf_config

    # Download models from HF
    download_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
    save_dir = os.path.join(download_path, "model_artifacts/tableformer/fast")

    config["model"]["save_dir"] = save_dir
    return config


def get_iocr_page(parsed_page: Dict, table_bbox: Tuple[float, float, float, float]):

    height = parsed_page["sanitized"]["dimension"]["height"]
    width = parsed_page["sanitized"]["dimension"]["width"]

    records = map_to_records(parsed_page["sanitized"]["cells"])

    cnt = 0

    tokens = []
    text_lines = []
    for i, rec in enumerate(records):
        tokens.append(
            {
                "bbox": {
                    "l": rec["x0"],
                    "t": height - rec["y1"],
                    "r": rec["x1"],
                    "b": height - rec["y0"],
                },
                "text": rec["text"],
                "id": i,
            }
        )

        text_lines.append(
            {
                "bbox": [rec["x0"], height - rec["y1"], rec["x1"], height - rec["y0"]],
                "text": rec["text"],
            }
        )

        """
        if table_bbox[0]<=tokens[-1]["bbox"]["l"] and \
           table_bbox[2]>=tokens[-1]["bbox"]["r"] and \
           table_bbox[1]<=tokens[-1]["bbox"]["b"] and \
           table_bbox[3]>=tokens[-1]["bbox"]["t"]:
            cnt += 1
            print(f"text-cell [{cnt}]: ", tokens[-1]["text"], "\t", tokens[-1]["bbox"])
        """

    iocr_page = {"tokens": tokens, "height": height, "width": width}

    return iocr_page


def to_np(pil_image: Image.Image):
    # Convert to NumPy array
    np_image = np.array(pil_image)

    # Handle different formats
    if np_image.ndim == 3:  # RGB or RGBA image
        if np_image.shape[2] == 4:  # RGBA image
            # Discard alpha channel and convert to BGR
            np_image = np_image[:, :, :3]  # Keep only RGB channels

        # Convert RGB to BGR by reversing the last axis
        np_image = np_image[:, :, ::-1]

        return np_image
    else:
        raise ValueError("Unsupported image format")


def tf_predict(
    config,
    page_image: Image.Image,
    parsed_page: dict,
    table_bbox: Tuple[float, float, float, float],
    viz: bool = True,
    device: str = "cpu",
    num_threads: int = 2,
    image_scale: float = 1.0,
):
    r"""
    Test the TFPredictor
    """

    table_bboxes = [[table_bbox[0], table_bbox[1], table_bbox[2], table_bbox[3]]]

    iocr_page = get_iocr_page(parsed_page, table_bbox=table_bbox)

    iocr_page["image"] = to_np(page_image)
    iocr_page["table_bboxes"] = table_bboxes

    # Loop over the iocr_pages
    predictor = TFPredictor(config, device=device, num_threads=num_threads)

    tf_output = predictor.multi_table_predict(
        iocr_page,
        table_bboxes=table_bboxes,
        do_matching=True,
        correct_overlapping_cells=False,
        sort_row_col_indexes=True,
    )
    # print("tf-output: ", json.dumps(tf_output, indent=2))

    table_out = tf_output[0]

    do_cell_matching = True

    table_cells = []
    for element in table_out["tf_responses"]:

        tc = TableCell.model_validate(element)
        if do_cell_matching and tc.bbox is not None:
            tc.bbox = tc.bbox.scaled(1 / image_scale)
        table_cells.append(tc)

    # Retrieving cols/rows, after post processing:
    num_rows = table_out["predict_details"]["num_rows"]
    num_cols = table_out["predict_details"]["num_cols"]
    otsl_seq = table_out["predict_details"]["prediction"]["rs_seq"]

    table_data = TableData(
        num_rows=num_rows, num_cols=num_cols, table_cells=table_cells
    )

    return table_data


class TableFormerUpdater:

    def __init__(self):
        # Init the TableFormer model
        self.tf_config = init_tf_model()

    def get_page_cells(self, filename: str):

        parser = pdf_parser_v2("fatal")

        try:
            key = "key"
            parser.load_document(key=key, filename=filename)

            parsed_doc = parser.parse_pdf_from_key(key=key)

            parser.unload_document(key)
            return parsed_doc

        except Exception as exc:
            logging.error(exc)

        return None

    def replace_tabledata(
        self,
        pdf_path: Path,
        true_doc: DoclingDocument,
        true_page_images: List[Image.Image],
    ) -> Tuple[bool, DoclingDocument]:

        updated = False

        # deep copy of the true-document
        pred_doc = copy.deepcopy(true_doc)

        parsed_doc = self.get_page_cells(str(pdf_path))
        if parsed_doc == None:
            logging.error("could not parse pdf-file")
            return False, pred_doc

        # Replace the groundtruth tables with predictions from TableFormer
        for item, level in pred_doc.iterate_items():
            if isinstance(item, TableItem):
                for prov in item.prov:

                    # md = item.export_to_markdown()
                    # print("groundtruth: \n\n", md)

                    page_image = true_page_images[prov.page_no - 1]
                    # page_image.show()

                    table_image = crop_bounding_box(
                        page_image=page_image,
                        page=pred_doc.pages[prov.page_no],
                        bbox=prov.bbox,
                    )
                    table_json = item.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    )
                    # print(json.dumps(table_json, indent=2))
                    # table_image.show()

                    table_data = tf_predict(
                        config=self.tf_config,
                        page_image=page_image,
                        parsed_page=parsed_doc["pages"][prov.page_no - 1],
                        table_bbox=(prov.bbox.l, prov.bbox.b, prov.bbox.r, prov.bbox.t),
                    )

                    item.data = table_data

                    updated = True

                    # md = item.export_to_markdown()
                    # print("prediction from table-former: \n\n", md)

        return updated, pred_doc
