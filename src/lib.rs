use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
pub struct Vector {
    pub id: usize,
    pub data: Vec<f32>,
}

impl Vector {
    pub fn l2_squared_distance(&self, other: &Vector) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    }

    pub fn cosine_similarity(&self, other: &Vector) -> f32 {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for (a, b) in self.data.iter().zip(other.data.iter()) {
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    }
}

pub struct VectorStore {
    vectors: Vec<Vector>,
    dimensions: usize,
}

impl VectorStore {
    pub fn new(dimensions: usize) -> Self {
        VectorStore {
            vectors: Vec::new(),
            dimensions,
        }
    }

    pub fn add_vector(&mut self, vector: Vector) -> Result<(), String> {
        if vector.data.len() != self.dimensions {
            return Err(format!(
                "Vector dimensions mismatch. Expected {}, got {}",
                self.dimensions,
                vector.data.len()
            ));
        }
        self.vectors.push(vector);
        Ok(())
    }

    pub fn search_knn(&self, query: &Vector, k: usize) -> Vec<(usize, f32)> {
        let mut heap: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();

        for v in &self.vectors {
            let dist = query.l2_squared_distance(v);

            if heap.len() < k {
                heap.push((OrderedFloat(dist), v.id));
            } else if let Some(max_dist) = heap.peek() {
                if dist < max_dist.0.into_inner() {
                    heap.pop();
                    heap.push((OrderedFloat(dist), v.id));
                }
            }
        }

        let mut results: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|(dist, id)| (id, dist.into_inner()))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_squared_distance() {
        let v1 = Vector {
            id: 1,
            data: vec![1.0, 2.0, 3.0],
        };
        let v2 = Vector {
            id: 2,
            data: vec![4.0, 5.0, 6.0],
        };

        // Math: (1-4)^2 + (2-5)^2 + (3-6)^2 = (-3)^2 + (-3)^2 + (-3)^2 = 9 + 9 + 9 = 27
        assert_eq!(v1.l2_squared_distance(&v2), 27.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = Vector {
            id: 1,
            data: vec![1.0, 0.0],
        };
        let v2 = Vector {
            id: 2,
            data: vec![0.0, 1.0],
        };

        // Orthogonal vectors (90 degrees apart) should have 0.0 similarity
        assert_eq!(v1.cosine_similarity(&v2), 0.0);

        let v3 = Vector {
            id: 3,
            data: vec![2.0, 0.0],
        };

        // Parallel vectors (same direction, different magnitude) should have 1.0 similarity
        assert_eq!(v1.cosine_similarity(&v3), 1.0);
    }

    #[test]
    fn test_search_knn() {
        let mut store = VectorStore::new(2); // 2-dimensional space

        store
            .add_vector(Vector {
                id: 1,
                data: vec![0.0, 0.0],
            })
            .unwrap();
        store
            .add_vector(Vector {
                id: 2,
                data: vec![1.0, 1.0],
            })
            .unwrap();
        store
            .add_vector(Vector {
                id: 3,
                data: vec![5.0, 5.0],
            })
            .unwrap();
        store
            .add_vector(Vector {
                id: 4,
                data: vec![2.0, 2.0],
            })
            .unwrap();

        let query = Vector {
            id: 99,
            data: vec![0.0, 0.0],
        }; // Query at the origin

        // We want the 2 closest vectors
        let results = store.search_knn(&query, 2);

        assert_eq!(results.len(), 2);

        // The absolute closest should be id 1 (distance 0)
        assert_eq!(results[0].0, 1);
        assert_eq!(results[0].1, 0.0);

        // The second closest should be id 2 (distance 1^2 + 1^2 = 2)
        assert_eq!(results[1].0, 2);
        assert_eq!(results[1].1, 2.0);
    }
}
