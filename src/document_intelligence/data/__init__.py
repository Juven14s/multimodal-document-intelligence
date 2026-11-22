"""Dataset adapters and canonical ground-truth contracts."""

from document_intelligence.data.adapters import adapt_cord_record, adapt_sroie_record
from document_intelligence.data.models import CanonicalRecord, DatasetName

__all__ = ["CanonicalRecord", "DatasetName", "adapt_cord_record", "adapt_sroie_record"]
