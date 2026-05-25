use crate::model::{ModelError, ModelResult};

const GGUF_HEADER_BYTES: usize = 24;
const GGUF_VALUE_UINT8: u32 = 0;
const GGUF_VALUE_INT8: u32 = 1;
const GGUF_VALUE_UINT16: u32 = 2;
const GGUF_VALUE_INT16: u32 = 3;
const GGUF_VALUE_UINT32: u32 = 4;
const GGUF_VALUE_INT32: u32 = 5;
const GGUF_VALUE_FLOAT32: u32 = 6;
const GGUF_VALUE_BOOL: u32 = 7;
const GGUF_VALUE_STRING: u32 = 8;
const GGUF_VALUE_ARRAY: u32 = 9;
const GGUF_VALUE_UINT64: u32 = 10;
const GGUF_VALUE_INT64: u32 = 11;
const GGUF_VALUE_FLOAT64: u32 = 12;
const MAX_ARRAY_DEPTH: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedGgufFile {
    pub(crate) version: u32,
    pub(crate) tensor_count: u64,
    pub(crate) metadata_kv_count: u64,
    pub(crate) header_bytes: usize,
    pub(crate) architecture: Option<String>,
    pub(crate) tensors: Vec<ParsedGgufTensorInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedGgufTensorInfo {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) ggml_type: u32,
    pub(crate) offset: u64,
}

pub(crate) fn parse_gguf_file(buffer: &[u8]) -> ModelResult<ParsedGgufFile> {
    if buffer.len() < GGUF_HEADER_BYTES || &buffer[0..4] != b"GGUF" {
        return Err(invalid_gguf());
    }

    let mut cursor = 4;
    let version = read_u32(&mut cursor, buffer)?;
    if version == 0 {
        return Err(invalid_gguf());
    }

    let tensor_count = read_u64(&mut cursor, buffer)?;
    let metadata_kv_count = read_u64(&mut cursor, buffer)?;
    let architecture = read_metadata(&mut cursor, buffer, metadata_kv_count)?;
    let tensors = read_tensor_infos(&mut cursor, buffer, tensor_count)?;

    Ok(ParsedGgufFile {
        version,
        tensor_count,
        metadata_kv_count,
        header_bytes: cursor,
        architecture,
        tensors,
    })
}

fn read_metadata(
    cursor: &mut usize,
    buffer: &[u8],
    count: u64,
) -> ModelResult<Option<String>> {
    let mut architecture = None;
    for _ in 0..count {
        let key = read_string(cursor, buffer)?;
        let value_type = read_u32(cursor, buffer)?;
        if key == "general.architecture" && value_type == GGUF_VALUE_STRING {
            architecture = Some(read_string(cursor, buffer)?);
        } else {
            skip_value(cursor, buffer, value_type, 0)?;
        }
    }
    Ok(architecture)
}

fn read_tensor_infos(
    cursor: &mut usize,
    buffer: &[u8],
    count: u64,
) -> ModelResult<Vec<ParsedGgufTensorInfo>> {
    let count = usize::try_from(count).map_err(|_| invalid_gguf())?;
    let mut tensors = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_string(cursor, buffer)?;
        let dimension_count = usize::try_from(read_u32(cursor, buffer)?).map_err(|_| invalid_gguf())?;
        let mut shape = Vec::with_capacity(dimension_count);
        for _ in 0..dimension_count {
            shape.push(usize::try_from(read_u64(cursor, buffer)?).map_err(|_| invalid_gguf())?);
        }
        let ggml_type = read_u32(cursor, buffer)?;
        let offset = read_u64(cursor, buffer)?;
        tensors.push(ParsedGgufTensorInfo {
            name,
            shape,
            ggml_type,
            offset,
        });
    }
    Ok(tensors)
}

fn skip_value(
    cursor: &mut usize,
    buffer: &[u8],
    value_type: u32,
    depth: usize,
) -> ModelResult<()> {
    match value_type {
        GGUF_VALUE_UINT8 | GGUF_VALUE_INT8 | GGUF_VALUE_BOOL => skip_bytes(cursor, buffer, 1),
        GGUF_VALUE_UINT16 | GGUF_VALUE_INT16 => skip_bytes(cursor, buffer, 2),
        GGUF_VALUE_UINT32 | GGUF_VALUE_INT32 | GGUF_VALUE_FLOAT32 => {
            skip_bytes(cursor, buffer, 4)
        }
        GGUF_VALUE_UINT64 | GGUF_VALUE_INT64 | GGUF_VALUE_FLOAT64 => {
            skip_bytes(cursor, buffer, 8)
        }
        GGUF_VALUE_STRING => read_string(cursor, buffer).map(|_| ()),
        GGUF_VALUE_ARRAY => {
            if depth >= MAX_ARRAY_DEPTH {
                return Err(invalid_gguf());
            }
            let item_type = read_u32(cursor, buffer)?;
            let item_count = read_u64(cursor, buffer)?;
            for _ in 0..item_count {
                skip_value(cursor, buffer, item_type, depth + 1)?;
            }
            Ok(())
        }
        _ => Err(invalid_gguf()),
    }
}

fn read_string(cursor: &mut usize, buffer: &[u8]) -> ModelResult<String> {
    let length = usize::try_from(read_u64(cursor, buffer)?).map_err(|_| invalid_gguf())?;
    let bytes = read_bytes(cursor, buffer, length)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| invalid_gguf())
}

fn read_u32(cursor: &mut usize, buffer: &[u8]) -> ModelResult<u32> {
    let bytes = read_bytes(cursor, buffer, 4)?;
    Ok(u32::from_le_bytes(
        bytes.try_into().expect("u32 read should be 4 bytes"),
    ))
}

fn read_u64(cursor: &mut usize, buffer: &[u8]) -> ModelResult<u64> {
    let bytes = read_bytes(cursor, buffer, 8)?;
    Ok(u64::from_le_bytes(
        bytes.try_into().expect("u64 read should be 8 bytes"),
    ))
}

fn read_bytes<'a>(cursor: &mut usize, buffer: &'a [u8], length: usize) -> ModelResult<&'a [u8]> {
    let end = cursor.checked_add(length).ok_or_else(invalid_gguf)?;
    if end > buffer.len() {
        return Err(invalid_gguf());
    }
    let bytes = &buffer[*cursor..end];
    *cursor = end;
    Ok(bytes)
}

fn skip_bytes(cursor: &mut usize, buffer: &[u8], length: usize) -> ModelResult<()> {
    read_bytes(cursor, buffer, length).map(|_| ())
}

fn invalid_gguf() -> ModelError {
    ModelError::InvalidConfig("invalid gguf metadata")
}

#[cfg(test)]
pub(crate) fn test_gguf_bytes(
    architecture: Option<&str>,
    tensor_name: &str,
    shape: &[u64],
) -> Vec<u8> {
    let metadata_kv_count = u64::from(architecture.is_some());
    let mut bytes = Vec::new();
    bytes.extend(b"GGUF");
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(1_u64.to_le_bytes());
    bytes.extend(metadata_kv_count.to_le_bytes());

    if let Some(architecture) = architecture {
        write_string(&mut bytes, "general.architecture");
        bytes.extend(GGUF_VALUE_STRING.to_le_bytes());
        write_string(&mut bytes, architecture);
    }

    write_string(&mut bytes, tensor_name);
    bytes.extend((shape.len() as u32).to_le_bytes());
    for dimension in shape {
        bytes.extend(dimension.to_le_bytes());
    }
    bytes.extend(0_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes
}

#[cfg(test)]
pub(crate) fn test_gguf_empty_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(b"GGUF");
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes
}

#[cfg(test)]
fn write_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend((value.len() as u64).to_le_bytes());
    bytes.extend(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use crate::{
        gguf_parse::{parse_gguf_file, test_gguf_bytes},
        model::ModelError,
    };

    #[test]
    fn parses_gguf_architecture_and_tensor_info() {
        let parsed = parse_gguf_file(&test_gguf_bytes(
            Some("llama"),
            "token_embd.weight",
            &[4, 2],
        ))
        .expect("gguf metadata should parse");

        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.tensor_count, 1);
        assert_eq!(parsed.metadata_kv_count, 1);
        assert_eq!(parsed.architecture.as_deref(), Some("llama"));
        assert_eq!(parsed.tensors[0].name, "token_embd.weight");
        assert_eq!(parsed.tensors[0].shape, vec![4, 2]);
        assert!(parsed.header_bytes > 24);
    }

    #[test]
    fn rejects_truncated_gguf_metadata() {
        let mut bytes = test_gguf_bytes(Some("llama"), "token_embd.weight", &[4, 2]);
        bytes.truncate(bytes.len() - 3);

        assert_eq!(
            parse_gguf_file(&bytes),
            Err(ModelError::InvalidConfig("invalid gguf metadata"))
        );
    }

    #[test]
    fn rejects_invalid_gguf_magic() {
        assert_eq!(
            parse_gguf_file(b"not-gguf"),
            Err(ModelError::InvalidConfig("invalid gguf metadata"))
        );
    }
}
