// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Tests for multiple indexes per column functionality

use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lancedb::index::{Index, IndexBuilder};
use lancedb::index::vector::{IvfFlatIndexBuilder, IvfPqIndexBuilder};
use lancedb::{connect, DistanceType, Result};
use tempfile::TempDir;

#[tokio::test]
async fn test_multiple_vector_indexes_per_column() -> Result<()> {
    let tmp_dir = TempDir::new().unwrap();
    let db = connect(tmp_dir.path()).execute().await?;

    // Create test data with vector column
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
    ]));

    let vectors = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ];

    let vector_array = FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
        vectors.into_iter().map(Some),
        4,
    );

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(vector_array),
        ],
    )?;

    let table = db.create_table("test_table", Box::new([batch].into_iter().map(Ok))).execute().await?;

    // Create multiple indexes on the same vector column with different distance metrics
    
    // Index 1: IVF_FLAT with L2 distance
    table
        .create_index(&["vector"], Index::IvfFlat(IvfFlatIndexBuilder {
            distance_type: DistanceType::L2,
            num_partitions: Some(2),
            sample_rate: 256,
            max_iterations: 50,
        }))
        .name("vector_l2_idx")
        .execute()
        .await?;

    // Index 2: IVF_FLAT with Cosine distance
    table
        .create_index(&["vector"], Index::IvfFlat(IvfFlatIndexBuilder {
            distance_type: DistanceType::Cosine,
            num_partitions: Some(2),
            sample_rate: 256,
            max_iterations: 50,
        }))
        .name("vector_cosine_idx")
        .execute()
        .await?;

    // Index 3: IVF_PQ with L2 distance
    table
        .create_index(&["vector"], Index::IvfPq(IvfPqIndexBuilder {
            distance_type: DistanceType::L2,
            num_partitions: Some(2),
            num_sub_vectors: Some(2),
            num_bits: Some(8),
            sample_rate: 256,
            max_iterations: 50,
        }))
        .name("vector_pq_l2_idx")
        .execute()
        .await?;

    // Verify all indexes exist
    let indices = table.list_indices().await?;
    assert_eq!(indices.len(), 3);

    let index_names: Vec<String> = indices.iter().map(|idx| idx.name.clone()).collect();
    assert!(index_names.contains(&"vector_l2_idx".to_string()));
    assert!(index_names.contains(&"vector_cosine_idx".to_string()));
    assert!(index_names.contains(&"vector_pq_l2_idx".to_string()));

    // Verify all indexes are on the same column
    for index in &indices {
        assert_eq!(index.columns, vec!["vector"]);
    }

    // Test that we can drop individual indexes
    table.drop_index("vector_cosine_idx").await?;
    let indices_after_drop = table.list_indices().await?;
    assert_eq!(indices_after_drop.len(), 2);

    let remaining_names: Vec<String> = indices_after_drop.iter().map(|idx| idx.name.clone()).collect();
    assert!(remaining_names.contains(&"vector_l2_idx".to_string()));
    assert!(remaining_names.contains(&"vector_pq_l2_idx".to_string()));
    assert!(!remaining_names.contains(&"vector_cosine_idx".to_string()));

    Ok(())
}

#[tokio::test]
async fn test_multiple_scalar_indexes_per_column() -> Result<()> {
    let tmp_dir = TempDir::new().unwrap();
    let db = connect(tmp_dir.path()).execute().await?;

    // Create test data with scalar column
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["A", "B", "A", "C", "B"])),
        ],
    )?;

    let table = db.create_table("scalar_test_table", Box::new([batch].into_iter().map(Ok))).execute().await?;

    // Create multiple indexes on the same scalar column
    
    // Index 1: BTree index
    table
        .create_index(&["category"], Index::BTree(Default::default()))
        .name("category_btree_idx")
        .execute()
        .await?;

    // Index 2: Bitmap index  
    table
        .create_index(&["category"], Index::Bitmap(Default::default()))
        .name("category_bitmap_idx")
        .execute()
        .await?;

    // Verify both indexes exist
    let indices = table.list_indices().await?;
    assert_eq!(indices.len(), 2);

    let index_names: Vec<String> = indices.iter().map(|idx| idx.name.clone()).collect();
    assert!(index_names.contains(&"category_btree_idx".to_string()));
    assert!(index_names.contains(&"category_bitmap_idx".to_string()));

    // Verify both indexes are on the same column
    for index in &indices {
        assert_eq!(index.columns, vec!["category"]);
    }

    Ok(())
}

#[tokio::test]
async fn test_index_name_conflicts() -> Result<()> {
    let tmp_dir = TempDir::new().unwrap();
    let db = connect(tmp_dir.path()).execute().await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )?;

    let table = db.create_table("conflict_test_table", Box::new([batch].into_iter().map(Ok))).execute().await?;

    // Create first index
    table
        .create_index(&["id"], Index::BTree(Default::default()))
        .name("my_index")
        .execute()
        .await?;

    // Try to create second index with same name and replace=false
    let result = table
        .create_index(&["id"], Index::Bitmap(Default::default()))
        .name("my_index")
        .replace(false)
        .execute()
        .await;

    // Should fail due to name conflict
    assert!(result.is_err());

    // Should succeed with replace=true (default)
    table
        .create_index(&["id"], Index::Bitmap(Default::default()))
        .name("my_index")
        .execute()
        .await?;

    let indices = table.list_indices().await?;
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].name, "my_index");
    assert_eq!(indices[0].index_type.to_string(), "BITMAP");

    Ok(())
}

#[tokio::test]
async fn test_default_index_naming() -> Result<()> {
    let tmp_dir = TempDir::new().unwrap();
    let db = connect(tmp_dir.path()).execute().await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
    ]));

    let vector_array = FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
        vec![vec![1.0, 2.0, 3.0, 4.0]].into_iter().map(Some),
        4,
    );

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(vector_array),
        ],
    )?;

    let table = db.create_table("naming_test_table", Box::new([batch].into_iter().map(Ok))).execute().await?;

    // Create indexes without custom names to test default naming
    table
        .create_index(&["vector"], Index::IvfFlat(IvfFlatIndexBuilder {
            distance_type: DistanceType::L2,
            num_partitions: Some(1),
            sample_rate: 256,
            max_iterations: 50,
        }))
        .execute()
        .await?;

    table
        .create_index(&["vector"], Index::IvfFlat(IvfFlatIndexBuilder {
            distance_type: DistanceType::Cosine,
            num_partitions: Some(1),
            sample_rate: 256,
            max_iterations: 50,
        }))
        .execute()
        .await?;

    table
        .create_index(&["id"], Index::BTree(Default::default()))
        .execute()
        .await?;

    let indices = table.list_indices().await?;
    assert_eq!(indices.len(), 3);

    let index_names: Vec<String> = indices.iter().map(|idx| idx.name.clone()).collect();
    
    // Check that default names include distance type for vector indexes
    assert!(index_names.iter().any(|name| name.contains("ivf_flat") && name.contains("l2")));
    assert!(index_names.iter().any(|name| name.contains("ivf_flat") && name.contains("cosine")));
    assert!(index_names.iter().any(|name| name == "id_idx"));

    Ok(())
}
