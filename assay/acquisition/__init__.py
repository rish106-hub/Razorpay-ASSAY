"""Immutable acquisition adapters for merchant-risk evidence sources."""

from assay.acquisition.mca import (
    McaAcquisitionCheckpoint,
    McaAcquisitionConfig,
    McaAcquisitionError,
    McaCompanyMasterDownloader,
)

__all__ = [
    "McaAcquisitionCheckpoint",
    "McaAcquisitionConfig",
    "McaAcquisitionError",
    "McaCompanyMasterDownloader",
]
