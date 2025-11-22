//! Model serialization and I/O utilities

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Save a model to a file using bincode
pub fn save_model<T: Serialize>(model: &T, path: impl AsRef<Path>) -> Result<()> {
    let encoded = bincode::serialize(model)
        .map_err(|e| crate::error::RustyAIError::SerializationError(e.to_string()))?;
    let mut file = File::create(path.as_ref())?;
    file.write_all(&encoded)?;
    Ok(())
}

/// Load a model from a file using bincode
pub fn load_model<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<T> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let model = bincode::deserialize(&buffer)
        .map_err(|e| crate::error::RustyAIError::SerializationError(e.to_string()))?;
    Ok(model)
}
