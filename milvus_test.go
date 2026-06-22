package milvus

import (
	"testing"

	"github.com/milvus-io/milvus/client/v2/entity"
)

func TestGenerateSparseVectors(t *testing.T) {
	vectors, err := GenerateSparseVectors(3, 128, 10, 7)
	if err != nil {
		t.Fatalf("GenerateSparseVectors returned error: %v", err)
	}
	if len(vectors) != 3 {
		t.Fatalf("len(vectors) = %d, want 3", len(vectors))
	}
	for i, vector := range vectors {
		if vector.FieldType() != entity.FieldTypeSparseVector {
			t.Fatalf("vector %d type = %v, want sparse", i, vector.FieldType())
		}
		if vector.Len() != 10 {
			t.Fatalf("vector %d nnz = %d, want 10", i, vector.Len())
		}
		if vector.Dim() > 128 {
			t.Fatalf("vector %d dim = %d, want <= 128", i, vector.Dim())
		}
	}
}

func TestGenerateSparseVectorsRejectsInvalidNNZ(t *testing.T) {
	_, err := GenerateSparseVectors(1, 4, 5, 1)
	if err == nil {
		t.Fatal("GenerateSparseVectors with nnz > dimension succeeded")
	}
}

func TestBuildGeneratedSparseColumn(t *testing.T) {
	col, err := buildGeneratedColumn(2, map[string]interface{}{
		"name":      "sparse",
		"type":      "sparse_float_vector",
		"generator": "random_sparse_vector",
		"dimension": 64,
		"nnz":       8,
		"seed":      3,
	})
	if err != nil {
		t.Fatalf("buildGeneratedColumn returned error: %v", err)
	}
	if col.Type() != entity.FieldTypeSparseVector {
		t.Fatalf("column type = %v, want sparse", col.Type())
	}
	if col.Len() != 2 {
		t.Fatalf("column len = %d, want 2", col.Len())
	}
}

func TestBuildSparseColumnFromObjects(t *testing.T) {
	col, err := buildColumn("sparse", "sparse_float_vector", 0, []interface{}{
		map[string]interface{}{
			"indices": []interface{}{float64(1), float64(9)},
			"values":  []interface{}{float64(0.2), float64(0.8)},
		},
		map[string]interface{}{
			"3": float64(0.4),
			"7": float64(0.6),
		},
	})
	if err != nil {
		t.Fatalf("buildColumn returned error: %v", err)
	}
	if col.Type() != entity.FieldTypeSparseVector {
		t.Fatalf("column type = %v, want sparse", col.Type())
	}
	if col.Len() != 2 {
		t.Fatalf("column len = %d, want 2", col.Len())
	}
}

func TestRandomStringGenerator(t *testing.T) {
	values, err := generateStringValues(2, map[string]interface{}{
		"generator": "random_string",
		"prefix":    "tenant-",
		"length":    6,
		"seed":      5,
	}, "random_string")
	if err != nil {
		t.Fatalf("generateStringValues returned error: %v", err)
	}
	for _, value := range values {
		if len(value) != len("tenant-")+6 {
			t.Fatalf("value %q length = %d, want %d", value, len(value), len("tenant-")+6)
		}
	}
}
