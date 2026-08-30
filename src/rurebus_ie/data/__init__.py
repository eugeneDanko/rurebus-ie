"""Dataset parsing and preprocessing utilities."""

from .brat_parser import (
    BratDocument,
    BratFormatError,
    Entity,
    Relation,
    TextSpan,
    load_brat_document,
)
from .ner_dataset import (
    RuReBusNerDataset,
    TokenizedNerExample,
    build_ner_dataset_from_manifest,
    build_ner_examples,
    load_documents_from_manifest,
)
from .ner_labels import BIO_LABELS, ID2LABEL, LABEL2ID
from .span_dataset import (
    GoldTokenSpan,
    RuReBusSpanDataset,
    TokenizedSpanExample,
    build_span_dataset_from_manifest,
    build_span_examples,
)
from .span_labels import SPAN_ID2LABEL, SPAN_LABEL2ID, SPAN_LABELS
from .span_hierarchy import (
    DEFAULT_SUPERCLASS_GROUPS,
    SpanLabelHierarchy,
    build_span_label_hierarchy,
)
from .protocol_split import build_global_test_protocol, validate_global_test_protocol
from .relation_dataset import (
    ENTITY_MARKERS,
    RelationTextExample,
    RuReBusRelationDataset,
    build_relation_dataset_from_manifest,
    build_relation_examples,
)
from .relation_labels import (
    NEGATIVE_RELATION_LABEL,
    RELATION_ID2LABEL,
    RELATION_LABEL2ID,
    RELATION_LABELS,
)

__all__ = [
    "BratDocument",
    "BratFormatError",
    "Entity",
    "Relation",
    "TextSpan",
    "load_brat_document",
    "BIO_LABELS",
    "ID2LABEL",
    "LABEL2ID",
    "RuReBusNerDataset",
    "TokenizedNerExample",
    "build_ner_dataset_from_manifest",
    "build_ner_examples",
    "load_documents_from_manifest",
    "GoldTokenSpan",
    "RuReBusSpanDataset",
    "SPAN_ID2LABEL",
    "SPAN_LABEL2ID",
    "SPAN_LABELS",
    "DEFAULT_SUPERCLASS_GROUPS",
    "SpanLabelHierarchy",
    "build_span_label_hierarchy",
    "TokenizedSpanExample",
    "build_span_dataset_from_manifest",
    "build_span_examples",
    "build_global_test_protocol",
    "validate_global_test_protocol",
    "ENTITY_MARKERS",
    "RelationTextExample",
    "RuReBusRelationDataset",
    "build_relation_dataset_from_manifest",
    "build_relation_examples",
    "NEGATIVE_RELATION_LABEL",
    "RELATION_ID2LABEL",
    "RELATION_LABEL2ID",
    "RELATION_LABELS",
]
