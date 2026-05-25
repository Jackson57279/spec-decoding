#[cfg(feature = "safetensors")]
use std::{fs, path::Path};

#[cfg(feature = "safetensors")]
use crate::model::{ModelError, ModelResult};

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

#[cfg(all(test, feature = "safetensors"))]
mod tests {
    use std::{
        fs::{create_dir_all, remove_dir_all, write},
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{
        model::ModelError,
        weight_metadata::{
            SafeTensorsFileMetadata, WeightTensorMetadata, read_safetensors_file_metadata,
        },
    };

    struct TempSafeTensors {
        root: PathBuf,
        path: PathBuf,
    }

    impl TempSafeTensors {
        fn new(name: &str, contents: Vec<u8>) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be valid")
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "speclative-diffusion-safetensors-{name}-{}-{unique}",
                std::process::id()
            ));
            create_dir_all(&root).expect("temp dir should be created");

            let path = root.join("model.safetensors");
            write(&path, contents).expect("safetensors file should be written");

            Self { root, path }
        }
    }

    impl Drop for TempSafeTensors {
        fn drop(&mut self) {
            let _ = remove_dir_all(&self.root);
        }
    }

    fn safetensors_bytes() -> Vec<u8> {
        let header = br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut bytes = Vec::new();
        bytes.extend((header.len() as u64).to_le_bytes());
        bytes.extend(header);
        bytes.extend([0_u8; 8]);
        bytes
    }

    #[test]
    fn reads_safetensors_file_metadata() {
        let file = TempSafeTensors::new("valid", safetensors_bytes());

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

    #[test]
    fn rejects_invalid_safetensors_metadata() {
        let file = TempSafeTensors::new("invalid", b"not-safetensors".to_vec());

        assert_eq!(
            read_safetensors_file_metadata(&file.path),
            Err(ModelError::InvalidConfig("invalid safetensors metadata"))
        );
    }
}
