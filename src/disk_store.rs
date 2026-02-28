use crate::Storage;
use bytemuck::cast_slice;
use memmap2::MmapOptions;
use std::fs::File;
use std::path::Path; // Imports our trait from lib.rs

pub struct MmapVectorStore {
    mmap: memmap2::Mmap,
    dimensions: usize,
    vector_count: usize,
}

impl MmapVectorStore {
    pub fn open<P: AsRef<Path>>(path: P, dimensions: usize) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;

        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let bytes_per_f32 = 4;
        let bytes_per_vector = dimensions * bytes_per_f32;

        if mmap.len() % bytes_per_vector != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File size is not a clean multiple of the vector dimensions.",
            ));
        }

        let vector_count = mmap.len() / bytes_per_vector;

        Ok(MmapVectorStore {
            mmap,
            dimensions,
            vector_count,
        })
    }
}

impl Storage for MmapVectorStore {
    fn get_vector(&self, id: usize) -> &[f32] {
        let start_byte = id * self.dimensions * 4;
        let end_byte = start_byte + (self.dimensions * 4);

        let byte_slice = &self.mmap[start_byte..end_byte];

        cast_slice(byte_slice)
    }

    fn len(&self) -> usize {
        self.vector_count
    }
}
