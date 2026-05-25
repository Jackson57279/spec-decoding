use std::{fs, path::Path};

use crate::model::{ModelError, ModelResult};

const GGUF_HEADER_BYTES: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightTensorMetadata {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorsFileMetadata {
    pub tensors: Vec<WeightTensorMetadata>,
    pub user_metadata_fields: usize,
    pub data_bytes: usize,
}

impl SafeTensorsFileMetadata {
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufFileMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub header_bytes: usize,
}

pub fn read_gguf_file_metadata(path: &Path) -> ModelResult<GgufFileMetadata> {
    let buffer =
        fs::read(path).map_err(|_| ModelError::InvalidConfig("gguf file must be readable"))?;

    parse_gguf_file_metadata(&buffer)
}

fn parse_gguf_file_metadata(buffer: &[u8]) -> ModelResult<GgufFileMetadata> {
    if buffer.len() < GGUF_HEADER_BYTES || &buffer[0..4] != b"GGUF" {
        return Err(ModelError::InvalidConfig("invalid gguf metadata"));
    }

    let version = u32::from_le_bytes(
        buffer[4..8]
            .try_into()
            .expect("gguf version field should be 4 bytes"),
    );
    if version == 0 {
        return Err(ModelError::InvalidConfig("invalid gguf metadata"));
    }

    Ok(GgufFileMetadata {
        version,
        tensor_count: u64::from_le_bytes(
            buffer[8..16]
                .try_into()
                .expect("gguf tensor count field should be 8 bytes"),
        ),
        metadata_kv_count: u64::from_le_bytes(
            buffer[16..24]
                .try_into()
                .expect("gguf metadata count field should be 8 bytes"),
        ),
        header_bytes: GGUF_HEADER_BYTES,
    })
}

#[cfg(feature = "safetensors")]
pub fn read_safetensors_file_metadata(path: &Path) -> ModelResult<SafeTensorsFileMetadata> {
    let buffer = fs::read(path)
        .map_err(|_| ModelError::InvalidConfig("safetensors file must be readable"))?;
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&buffer)
        .map_err(|_| ModelError::InvalidConfig("invalid safetensors metadata"))?;

    let mut tensors = metadata
        .tensors()
        .into_iter()
        .map(|(name, info)| WeightTensorMetadata {
            name,
            dtype: format!("{:?}", info.dtype),
            shape: info.shape.clone(),
            data_bytes: info.data_offsets.1 - info.data_offsets.0,
        })
        .collect::<Vec<_>>();
    tensors.sort_by(|left, right| left.name.cmp(&right.name));

    Ok(SafeTensorsFileMetadata {
        tensors,
        user_metadata_fields: metadata.metadata().as_ref().map_or(0, |value| value.len()),
        data_bytes: metadata.data_len(),
    })
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{create_dir_all, remove_dir_all, write},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{
        model::ModelError,
        weight_metadata::{GgufFileMetadata, read_gguf_file_metadata},
    };

    #[cfg(feature = "safetensors")]
    use crate::weight_metadata::{
        SafeTensorsFileMetadata, WeightTensorMetadata, read_safetensors_file_metadata,
    };

    struct TempWeightFile {
        root: PathBuf,
        path: PathBuf,
    }

    impl TempWeightFile {
        fn new(name: &str, file_name: &str, contents: Vec<u8>) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be valid")
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "speclative-diffusion-weight-metadata-{name}-{}-{unique}",
                std::process::id()
            ));
            create_dir_all(&root).expect("temp dir should be created");

            let path = root.join(file_name);
            write(&path, contents).expect("weight file should be written");

            Self { root, path }
        }
    }

    impl Drop for TempWeightFile {
        fn drop(&mut self) {
            let _ = remove_dir_all(&self.root);
        }
    }

    fn gguf_bytes(version: u32, tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(version.to_le_bytes());
        bytes.extend(tensor_count.to_le_bytes());
        bytes.extend(metadata_kv_count.to_le_bytes());
        bytes
    }

    #[test]
    fn reads_gguf_file_metadata() {
        let file = TempWeightFile::new("gguf-valid", "model.gguf", gguf_bytes(3, 12, 4));

        assert_eq!(
            read_gguf_file_metadata(&file.path),
            Ok(GgufFileMetadata {
                version: 3,
                tensor_count: 12,
                metadata_kv_count: 4,
                header_bytes: 24,
            })
        );
    }

    #[test]
    fn rejects_invalid_gguf_metadata() {
        let file = TempWeightFile::new("gguf-invalid", "model.gguf", b"not-gguf".to_vec());

        assert_eq!(
            read_gguf_file_metadata(&file.path),
            Err(ModelError::InvalidConfig("invalid gguf metadata"))
        );
    }

    #[cfg(feature = "safetensors")]
    fn safetensors_bytes() -> Vec<u8> {
        let header = br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut bytes = Vec::new();
        bytes.extend((header.len() as u64).to_le_bytes());
        bytes.extend(header);
        bytes.extend([0_u8; 8]);
        bytes
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn reads_safetensors_file_metadata() {
        let file = TempWeightFile::new(
            "safetensors-valid",
            "model.safetensors",
            safetensors_bytes(),
        );

        assert_eq!(
            read_safetensors_file_metadata(&file.path),
            Ok(SafeTensorsFileMetadata {
                tensors: vec![WeightTensorMetadata {
                    name: String::from("weight"),
                    dtype: String::from("F32"),
                    shape: vec![2],
                    data_bytes: 8,
                }],
                user_metadata_fields: 1,
                data_bytes: 8,
            })
        );
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn rejects_invalid_safetensors_metadata() {
        let file = TempWeightFile::new(
            "safetensors-invalid",
            "model.safetensors",
            b"not-safetensors".to_vec(),
        );

        assert_eq!(
            read_safetensors_file_metadata(&file.path),
            Err(ModelError::InvalidConfig("invalid safetensors metadata"))
        );
    }
}
