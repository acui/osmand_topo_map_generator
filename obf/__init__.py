"""OsmAnd binary map (.obf) parsing — map sections only.

Reference implementation: osmandapp/OsmAnd ``BinaryMapIndexReader.java``.
"""
from obf.obf_file import MapObject, ObfFile, open_obf_files

__all__ = ["MapObject", "ObfFile", "open_obf_files"]
