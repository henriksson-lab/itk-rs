//! Stub I/O implementations for common medical and scientific image formats.
//!
//! Each format provides `*ImageReader` and `*ImageWriter` structs matching
//! ITK's `ImageIOBase`-derived classes. Full implementations require external
//! crates (e.g. `nifti`, `dicom`, `hdf5`, `tiff`, `image`).

use crate::image::Image;

macro_rules! format_io_stub {
    ($reader:ident, $writer:ident, $desc:literal, $ext:literal) => {
        #[doc = concat!($desc, " reader. Analog to ITK's `", $ext, "ImageIO`.")]
        pub struct $reader {
            pub path: std::path::PathBuf,
        }
        impl $reader {
            pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
            pub fn read_2d(&self) -> Result<Image<f32, 2>, String> {
                Err(format!(concat!($desc, " reader not yet implemented; path={:?}"), self.path))
            }
            pub fn read_3d(&self) -> Result<Image<f32, 3>, String> {
                Err(format!(concat!($desc, " reader not yet implemented; path={:?}"), self.path))
            }
        }
        #[doc = concat!($desc, " writer. Analog to ITK's `", $ext, "ImageIO`.")]
        pub struct $writer {
            pub path: std::path::PathBuf,
        }
        impl $writer {
            pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
            pub fn write_2d(&self, _image: &Image<f32, 2>) -> Result<(), String> {
                Err(format!(concat!($desc, " writer not yet implemented; path={:?}"), self.path))
            }
            pub fn write_3d(&self, _image: &Image<f32, 3>) -> Result<(), String> {
                Err(format!(concat!($desc, " writer not yet implemented; path={:?}"), self.path))
            }
        }
    };
}

format_io_stub!(NrrdImageReader, NrrdImageWriter, "NRRD (.nrrd/.nhdr)", "Nrrd");
format_io_stub!(DicomImageReader, DicomImageWriter, "DICOM", "GDCM");
format_io_stub!(MrcImageReader, MrcImageWriter, "MRC (cryo-EM)", "MRC");
format_io_stub!(Hdf5ImageReader, Hdf5ImageWriter, "HDF5", "HDF5");
format_io_stub!(TiffImageReader, TiffImageWriter, "TIFF", "TIFF");
format_io_stub!(JpegImageReader, JpegImageWriter, "JPEG", "JPEG");
format_io_stub!(BmpImageReader, BmpImageWriter, "BMP", "BMP");
format_io_stub!(VtkImageReader, VtkImageWriter, "VTK image", "VTK");
format_io_stub!(MincImageReader, MincImageWriter, "MINC", "MINC");
format_io_stub!(ZeissLsmReader, ZeissLsmWriter, "Zeiss LSM", "LSM");
format_io_stub!(GeImageReader, GeImageWriter, "GE (4/5/Adw)", "GE");
format_io_stub!(SiemensImageReader, SiemensImageWriter, "Siemens", "Siemens");
format_io_stub!(BrukerImageReader, BrukerImageWriter, "Bruker 2dseq", "Bruker");
format_io_stub!(PhilipsParRecReader, PhilipsParRecWriter, "Philips PAR/REC", "PhilipsRec");
format_io_stub!(StimulateImageReader, StimulateImageWriter, "Stimulate", "Stimulate");
format_io_stub!(GiplImageReader, GiplImageWriter, "GIPL", "GIPL");
format_io_stub!(RawImageReader, RawImageWriter, "RAW", "Raw");

// ---------------------------------------------------------------------------
// Mesh formats
// ---------------------------------------------------------------------------

/// VTK PolyData mesh reader.
pub struct VtkPolyDataReader { pub path: std::path::PathBuf }
impl VtkPolyDataReader {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn read(&self) -> Result<Vec<[f32; 3]>, String> {
        Err(format!("VTK PolyData reader not yet implemented; path={:?}", self.path))
    }
}

/// VTK PolyData mesh writer.
pub struct VtkPolyDataWriter { pub path: std::path::PathBuf }
impl VtkPolyDataWriter {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn write(&self, _verts: &[[f32; 3]], _faces: &[[u32; 3]]) -> Result<(), String> {
        Err(format!("VTK PolyData writer not yet implemented; path={:?}", self.path))
    }
}

/// OBJ mesh reader.
pub struct ObjMeshReader { pub path: std::path::PathBuf }
impl ObjMeshReader {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn read(&self) -> Result<Vec<[f32; 3]>, String> {
        Err(format!("OBJ reader not yet implemented; path={:?}", self.path))
    }
}

/// OBJ mesh writer.
pub struct ObjMeshWriter { pub path: std::path::PathBuf }
impl ObjMeshWriter {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn write(&self, _verts: &[[f32; 3]], _faces: &[[u32; 3]]) -> Result<(), String> {
        Err(format!("OBJ writer not yet implemented; path={:?}", self.path))
    }
}

/// OFF mesh reader.
pub struct OffMeshReader { pub path: std::path::PathBuf }
impl OffMeshReader {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn read(&self) -> Result<Vec<[f32; 3]>, String> {
        Err(format!("OFF reader not yet implemented; path={:?}", self.path))
    }
}

/// OFF mesh writer.
pub struct OffMeshWriter { pub path: std::path::PathBuf }
impl OffMeshWriter {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn write(&self, _verts: &[[f32; 3]], _faces: &[[u32; 3]]) -> Result<(), String> {
        Err(format!("OFF writer not yet implemented; path={:?}", self.path))
    }
}

/// FreeSurfer mesh reader.
pub struct FreeSurferMeshReader { pub path: std::path::PathBuf }
impl FreeSurferMeshReader {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn read(&self) -> Result<Vec<[f32; 3]>, String> {
        Err(format!("FreeSurfer reader not yet implemented; path={:?}", self.path))
    }
}

/// FreeSurfer mesh writer.
pub struct FreeSurferMeshWriter { pub path: std::path::PathBuf }
impl FreeSurferMeshWriter {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn write(&self, _verts: &[[f32; 3]], _faces: &[[u32; 3]]) -> Result<(), String> {
        Err(format!("FreeSurfer writer not yet implemented; path={:?}", self.path))
    }
}

/// GIFTI mesh reader.
pub struct GiftiMeshReader { pub path: std::path::PathBuf }
impl GiftiMeshReader {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn read(&self) -> Result<Vec<[f32; 3]>, String> {
        Err(format!("GIFTI reader not yet implemented; path={:?}", self.path))
    }
}

/// GIFTI mesh writer.
pub struct GiftiMeshWriter { pub path: std::path::PathBuf }
impl GiftiMeshWriter {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn write(&self, _verts: &[[f32; 3]], _faces: &[[u32; 3]]) -> Result<(), String> {
        Err(format!("GIFTI writer not yet implemented; path={:?}", self.path))
    }
}

// ---------------------------------------------------------------------------
// Transform I/O
// ---------------------------------------------------------------------------

/// ITK legacy transform file reader.
pub struct ItkTransformReader { pub path: std::path::PathBuf }
impl ItkTransformReader {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn read(&self) -> Result<Vec<f64>, String> {
        Err(format!("ITK transform reader not yet implemented; path={:?}", self.path))
    }
}

/// ITK legacy transform file writer.
pub struct ItkTransformWriter { pub path: std::path::PathBuf }
impl ItkTransformWriter {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn write(&self, _params: &[f64]) -> Result<(), String> {
        Err(format!("ITK transform writer not yet implemented; path={:?}", self.path))
    }
}

/// HDF5 transform file reader.
pub struct Hdf5TransformReader { pub path: std::path::PathBuf }
impl Hdf5TransformReader {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn read(&self) -> Result<Vec<f64>, String> {
        Err(format!("HDF5 transform reader not yet implemented; path={:?}", self.path))
    }
}

/// HDF5 transform file writer.
pub struct Hdf5TransformWriter { pub path: std::path::PathBuf }
impl Hdf5TransformWriter {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self { Self { path: path.into() } }
    pub fn write(&self, _params: &[f64]) -> Result<(), String> {
        Err(format!("HDF5 transform writer not yet implemented; path={:?}", self.path))
    }
}
