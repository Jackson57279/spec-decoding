use std::{fs, path::Path};

use crate::{
    gguf_parse::{ParsedGgufTensorInfo, parse_gguf_file},
    model::{ModelError, ModelResult},
};

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
    pub architecture: Option<String>,
    pub tensors: Vec<GgufTensorMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorMetadata {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: u32,
    pub offset: u64,
}

impl From<ParsedGgufTensorInfo> for GgufTensorMetadata {
    fn from(value: ParsedGgufTensorInfo) -> Self {
        Self {
            name: value.name,
            shape: value.shape,
            ggml_type: value.ggml_type,
            offset: value.offset,
        }
    }
}

pub fn read_gguf_file_metadata(path: &Path) -> ModelResult<GgufFileMetadata> {
    let buffer =
        fs::read(path).map_err(|_| ModelError::InvalidConfig("gguf file must be readable"))?;

    parse_gguf_file_metadata(&buffer)
}

fn parse_gguf_file_metadata(buffer: &[u8]) -> ModelResult<GgufFileMetadata> {
    let parsed = parse_gguf_file(buffer)?;
    Ok(GgufFileMetadata {
        version: parsed.version,
        tensor_count: parsed.tensor_count,
        metadata_kv_count: parsed.metadata_kv_count,
        header_bytes: parsed.header_bytes,
        architecture: parsed.architecture,
        tensors: parsed.tensors.into_iter().map(Into::into).collect(),
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
        gguf_parse::test_gguf_bytes, model::ModelError, weight_metadata::read_gguf_file_metadata,
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

    #[test]
    fn reads_gguf_file_metadata() {
        let file = TempWeightFile::new(
            "gguf-valid",
            "model.gguf",
            test_gguf_bytes(Some("llama"), "token_embd.weight", &[4, 2]),
        );

        let metadata = read_gguf_file_metadata(&file.path).expect("metadata should parse");

        assert_eq!(metadata.version, 3);
        assert_eq!(metadata.tensor_count, 1);
        assert_eq!(metadata.metadata_kv_count, 1);
        assert!(metadata.header_bytes > 24);
        assert_eq!(metadata.architecture.as_deref(), Some("llama"));
        assert_eq!(metadata.tensors[0].name, "token_embd.weight");
        assert_eq!(metadata.tensors[0].shape, vec![4, 2]);
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
