//! NIfTI (.nii / .nii.gz) I/O stubs.
//!
//! Provides the `ImageReader`/`ImageWriter` API for NIfTI files.
//! A full implementation would depend on a `nifti` crate.

use crate::image::{Image, Region};

/// NIfTI image reader. Analog to ITK's `NiftiImageIO`.
pub struct NiftiImageReader {
    pub path: std::path::PathBuf,
}

impl NiftiImageReader {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Read a 2D f32 image from a NIfTI file.
    /// Returns `Err` until a native NIfTI backend is linked.
    pub fn read_2d(&self) -> Result<Image<f32, 2>, String> {
        Err(format!("NIfTI reader not yet backed by a native implementation; path={:?}", self.path))
    }

    /// Read a 3D f32 image from a NIfTI file.
    pub fn read_3d(&self) -> Result<Image<f32, 3>, String> {
        Err(format!("NIfTI reader not yet backed by a native implementation; path={:?}", self.path))
    }
}

/// NIfTI image writer. Analog to ITK's `NiftiImageIO`.
pub struct NiftiImageWriter {
    pub path: std::path::PathBuf,
}

impl NiftiImageWriter {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Write a 2D f32 image to a NIfTI file.
    pub fn write_2d(&self, _image: &Image<f32, 2>) -> Result<(), String> {
        Err(format!("NIfTI writer not yet backed by a native implementation; path={:?}", self.path))
    }

    /// Write a 3D f32 image to a NIfTI file.
    pub fn write_3d(&self, _image: &Image<f32, 3>) -> Result<(), String> {
        Err(format!("NIfTI writer not yet backed by a native implementation; path={:?}", self.path))
    }
}
