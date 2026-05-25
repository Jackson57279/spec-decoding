use crate::model::{ModelError, ModelResult};

const GGUF_HEADER_BYTES: usize = 24;
const DEFAULT_GGUF_ALIGNMENT: usize = 32;
#[cfg(test)]
const MAX_TEST_SIMPLE_TENSOR_BYTES: usize = 1024 * 1024;
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
    pub(crate) alignment: usize,
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
    let metadata = read_metadata(&mut cursor, buffer, metadata_kv_count)?;
    let tensors = read_tensor_infos(&mut cursor, buffer, tensor_count)?;
    validate_tensor_data_bounds(cursor, buffer, metadata.alignment, &tensors)?;

    Ok(ParsedGgufFile {
        version,
        tensor_count,
        metadata_kv_count,
        header_bytes: cursor,
        architecture: metadata.architecture,
        alignment: metadata.alignment,
        tensors,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedGgufMetadata {
    architecture: Option<String>,
    alignment: usize,
}

fn read_metadata(cursor: &mut usize, buffer: &[u8], count: u64) -> ModelResult<ParsedGgufMetadata> {
    let mut architecture = None;
    let mut alignment = DEFAULT_GGUF_ALIGNMENT;
    for _ in 0..count {
        let key = read_string(cursor, buffer)?;
        let value_type = read_u32(cursor, buffer)?;
        if key == "general.architecture" && value_type == GGUF_VALUE_STRING {
            architecture = Some(read_string(cursor, buffer)?);
        } else if key == "general.alignment" && value_type == GGUF_VALUE_UINT32 {
            alignment = usize::try_from(read_u32(cursor, buffer)?).map_err(|_| invalid_gguf())?;
        } else {
            skip_value(cursor, buffer, value_type, 0)?;
        }
    }

    if alignment == 0 {
        return Err(invalid_gguf());
    }

    Ok(ParsedGgufMetadata {
        architecture,
        alignment,
    })
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
        let dimension_count =
            usize::try_from(read_u32(cursor, buffer)?).map_err(|_| invalid_gguf())?;
        let mut shape = Vec::with_capacity(dimension_count);
        for _ in 0..dimension_count {
            shape.push(usize::try_from(read_u64(cursor, buffer)?).map_err(|_| invalid_gguf())?);
        }
        let ggml_type = read_u32(cursor, buffer)?;
        if !is_known_ggml_type(ggml_type) {
            return Err(invalid_gguf());
        }
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

fn validate_tensor_data_bounds(
    tensor_info_end: usize,
    buffer: &[u8],
    alignment: usize,
    tensors: &[ParsedGgufTensorInfo],
) -> ModelResult<()> {
    if tensors.is_empty() {
        return Ok(());
    }

    let data_start = align_up(tensor_info_end, alignment)?;
    if data_start > buffer.len() {
        return Err(invalid_gguf());
    }

    let data_len = buffer.len() - data_start;
    for tensor in tensors {
        let offset = usize::try_from(tensor.offset).map_err(|_| invalid_gguf())?;
        if offset % alignment != 0 || offset > data_len {
            return Err(invalid_gguf());
        }

        if let Some(data_bytes) = simple_tensor_data_bytes(tensor)? {
            let end = offset.checked_add(data_bytes).ok_or_else(invalid_gguf)?;
            if end > data_len {
                return Err(invalid_gguf());
            }
        }
    }

    Ok(())
}

fn align_up(value: usize, alignment: usize) -> ModelResult<usize> {
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }

    value
        .checked_add(alignment - remainder)
        .ok_or_else(invalid_gguf)
}

fn simple_tensor_data_bytes(tensor: &ParsedGgufTensorInfo) -> ModelResult<Option<usize>> {
    let Some(element_bytes) = simple_ggml_element_bytes(tensor.ggml_type) else {
        return Ok(None);
    };
    let element_count = tensor
        .shape
        .iter()
        .try_fold(1_usize, |product, dimension| {
            if *dimension == 0 {
                return Err(invalid_gguf());
            }
            product.checked_mul(*dimension).ok_or_else(invalid_gguf)
        })?;

    element_count
        .checked_mul(element_bytes)
        .map(Some)
        .ok_or_else(invalid_gguf)
}

fn simple_ggml_element_bytes(ggml_type: u32) -> Option<usize> {
    match ggml_type {
        0 => Some(4),
        1 | 25 | 30 => Some(2),
        24 => Some(1),
        26 => Some(4),
        27 | 28 => Some(8),
        _ => None,
    }
}

fn is_known_ggml_type(ggml_type: u32) -> bool {
    matches!(
        ggml_type,
        0 | 1
            | 2
            | 3
            | 6
            | 7
            | 8
            | 9
            | 10
            | 11
            | 12
            | 13
            | 14
            | 15
            | 16
            | 17
            | 18
            | 19
            | 20
            | 21
            | 22
            | 23
            | 24
            | 25
            | 26
            | 27
            | 28
            | 29
            | 30
            | 34
            | 35
            | 39
    )
}

fn skip_value(cursor: &mut usize, buffer: &[u8], value_type: u32, depth: usize) -> ModelResult<()> {
    match value_type {
        GGUF_VALUE_UINT8 | GGUF_VALUE_INT8 | GGUF_VALUE_BOOL => skip_bytes(cursor, buffer, 1),
        GGUF_VALUE_UINT16 | GGUF_VALUE_INT16 => skip_bytes(cursor, buffer, 2),
        GGUF_VALUE_UINT32 | GGUF_VALUE_INT32 | GGUF_VALUE_FLOAT32 => skip_bytes(cursor, buffer, 4),
        GGUF_VALUE_UINT64 | GGUF_VALUE_INT64 | GGUF_VALUE_FLOAT64 => skip_bytes(cursor, buffer, 8),
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
    let data_bytes = f32_tensor_data_bytes(shape);
    let ggml_type = if data_bytes <= MAX_TEST_SIMPLE_TENSOR_BYTES {
        0
    } else {
        2
    };
    test_gguf_bytes_with_type(architecture, tensor_name, shape, ggml_type)
}

#[cfg(test)]
fn test_gguf_bytes_with_type(
    architecture: Option<&str>,
    tensor_name: &str,
    shape: &[u64],
    ggml_type: u32,
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
    bytes.extend(ggml_type.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    pad_to_alignment(&mut bytes, DEFAULT_GGUF_ALIGNMENT);
    if simple_ggml_element_bytes(ggml_type).is_some() {
        bytes.extend(vec![0_u8; f32_tensor_data_bytes(shape)]);
    }
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
fn pad_to_alignment(bytes: &mut Vec<u8>, alignment: usize) {
    let padding = (alignment - (bytes.len() % alignment)) % alignment;
    bytes.extend(vec![0_u8; padding]);
}

#[cfg(test)]
fn f32_tensor_data_bytes(shape: &[u64]) -> usize {
    shape
        .iter()
        .try_fold(1_usize, |product, dimension| {
            product.checked_mul(usize::try_from(*dimension).expect("test shape should fit usize"))
        })
        .expect("test tensor size should fit usize")
        * 4
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
        assert_eq!(parsed.alignment, 32);
        assert_eq!(parsed.tensors[0].name, "token_embd.weight");
        assert_eq!(parsed.tensors[0].shape, vec![4, 2]);
        assert_eq!(parsed.tensors[0].ggml_type, 0);
        assert_eq!(parsed.tensors[0].offset, 0);
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
    fn rejects_unknown_ggml_tensor_type() {
        let bytes =
            super::test_gguf_bytes_with_type(Some("llama"), "token_embd.weight", &[4, 2], u32::MAX);

        assert_eq!(
            parse_gguf_file(&bytes),
            Err(ModelError::InvalidConfig("invalid gguf metadata"))
        );
    }

    #[test]
    fn rejects_tensor_data_outside_file() {
        let mut bytes = test_gguf_bytes(Some("llama"), "token_embd.weight", &[4, 2]);
        bytes.truncate(bytes.len() - 1);

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
