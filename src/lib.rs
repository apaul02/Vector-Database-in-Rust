use core::f64;
use ordered_float::OrderedFloat;
use rand::{Rng, RngExt};
use std::cmp::Reverse;
use std::{
    collections::{BinaryHeap, HashSet},
    vec,
};

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

#[derive(Debug, Clone)]
pub struct HnswNode {
    pub vector_id: usize,
    pub connections: Vec<Vec<usize>>,
}

impl HnswNode {
    pub fn new(vector_id: usize, max_layer: usize) -> Self {
        HnswNode {
            vector_id,
            connections: vec![Vec::new(); max_layer + 1],
        }
    }
}

pub struct HnswIndex {
    pub nodes: Vec<HnswNode>,
    pub entry_point: Option<usize>,
    pub max_layer: usize,

    pub m: usize,
    pub m_max_0: usize,
    pub m_l: f64,
    pub ef_construction: usize,
}

impl HnswIndex {
    pub fn new(m: usize) -> Self {
        let m_l = 1.0 / (m as f64).ln();

        HnswIndex {
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            m,
            m_max_0: m * 2,
            m_l,
            ef_construction,
        }
    }

    pub fn generate_random_layer(&self) -> usize {
        let mut rng = rand::rng();
        let r: f64 = rng.random_range(f64::EPSILON..=1.0);

        let layer = -r.ln() * self.m_l;
        layer.floor() as usize
    }

    pub fn search_layer(
        &self,
        query: &Vector,
        entry_point_id: usize,
        layer: usize,
        store: &VectorStore,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        use ordered_float::OrderedFloat;
        use std::cmp::Reverse;
        use std::collections::{BinaryHeap, HashSet};

        let mut visited = HashSet::new();
        visited.insert(entry_point_id);

        let ep_dist = query.l2_squared_distance(&store.vectors[entry_point_id]);

        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();
        candidates.push(Reverse((OrderedFloat(ep_dist), entry_point_id)));

        let mut top_results: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        top_results.push((OrderedFloat(ep_dist), entry_point_id));

        while let Some(Reverse((OrderedFloat(cand_dist), cand_id))) = candidates.pop() {
            let furthest_res_dist = top_results.peek().unwrap().0.into_inner();
            if cand_dist > furthest_res_dist && top_results.len() == ef {
                break;
            }

            for &neighbor_id in &self.nodes[cand_id].connections[layer] {
                if visited.insert(neighbor_id) {
                    let neighbor_vec = &store.vectors[neighbor_id];
                    let dist = query.l2_squared_distance(neighbor_vec);

                    let furthest_res_dist = top_results.peek().unwrap().0.into_inner();

                    if top_results.len() < ef || dist < furthest_res_dist {
                        candidates.push(Reverse((OrderedFloat(dist), neighbor_id)));
                        top_results.push((OrderedFloat(dist), neighbor_id));

                        if top_results.len() > ef {
                            top_results.pop();
                        }
                    }
                }
            }
        }

        let mut results: Vec<(usize, f32)> = top_results
            .into_iter()
            .map(|(dist, id)| (id, dist.into_inner()))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        results
    }

    pub fn insert(&mut self, vector_id: usize, store: &VectorStore) {
        let new_node_level = self.generate_random_layer();
        let mut new_node = HnswNode::new(vector_id, new_node_level);

        if self.entry_point.is_none() {
            self.entry_point = Some(vector_id);
            self.max_layer = new_node_level;
            self.nodes.push(new_node);
            return;
        }

        let mut current_entry = self.entry_point.unwrap();
        let query_vec = &store.vectors[vector_id];

        let mut current_level = self.max_layer;

        while current_level > new_node_level {
            let closest_nodes =
                self.search_layer(query_vec, current_entry, current_level, store, 1);
            current_entry = closest_nodes[0].0;

            if current_level == 0 {
                break;
            }
            current_level -= 1;
        }

        let start_level = std::cmp::min(new_node_level, self.max_layer);
        for layer in (0..=start_level).rev() {
            let m_max = if layer == 0 { self.m_max_0 } else { self.m };

            let closest_nodes =
                self.search_layer(query_vec, current_entry, layer, store, self.ef_construction);

            let neighbors_to_add = std::cmp::min(closest_nodes.len(), m_max);

            for i in 0..neighbors_to_add {
                let (closest_id, _) = closest_nodes[i];
                new_node.connections[layer].push(closest_id);

                let existing_node_connections = &mut self.nodes[closest_id].connections[layer];
                existing_node_connections.push(vector_id);

                if existing_node_connections.len() > m_max {
                    let existing_vec = &store.vectors[closest_id];

                    let mut distances: Vec<(usize, f32)> = existing_node_connections
                        .iter()
                        .map(|&conn_id| {
                            let dist = existing_vec.l2_squared_distance(&store.vectors[conn_id]);
                            (conn_id, dist)
                        })
                        .collect();

                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    existing_node_connections.clear();
                    for (conn_id, _) in distances.into_iter().take(m_max) {
                        existing_node_connections.push(conn_id);
                    }
                }
            }

            current_entry = closest_nodes[0].0;
        }

        self.nodes.push(new_node);

        if new_node_level > self.max_layer {
            self.max_layer = new_node_level;
            self.entry_point = Some(vector_id);
        }
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
