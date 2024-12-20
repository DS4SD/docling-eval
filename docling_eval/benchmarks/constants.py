from enum import Enum


class BenchMarkColumns(Enum):
    DOCLING_VERSION = "docling_version"

    STATUS = "status"
    DOC_ID = "document_id"

    GROUNDTRUTH = "GroundTruthDocument"
    PREDICTION = "PredictedDocument"
    ORIGINAL = "BinaryDocument"

    PAGE_IMAGES = "page_images"
    PICTURES = "pictures"

    MIMETYPE = "mimetype"


class BenchMarkNames(Enum):

    # End-to-End
    DPBENCH = "DPBench"
    OMNIDOCBENCH = "OmniDcoBench"
    WORDSCAPE = "WordScape"

    # Layout
    PUBLAYNET = "PubLayNet"
    DOCLAYNET = "DocLayNet"

    # Table
    PUB1M = "Pub1M"
    PUBTABNET = "PubTabNet"
    FINTABNET = "FinTabNet"
    WIKITABNET = "WikiTabNet"

    # Formula
    # ???

    # OCR
    # ???
