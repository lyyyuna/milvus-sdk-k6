package milvus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	milvusclient "github.com/milvus-io/milvus/client/v2/milvusclient"
	"go.k6.io/k6/js/modules"
)

func init() {
	modules.Register("k6/x/milvus", new(Module))
}

type Module struct{}

func (m *Module) NewModuleInstance(_ modules.VU) modules.Instance {
	factory := &ClientFactory{}
	return &moduleInstance{exports: modules.Exports{
		Default: factory,
		Named: map[string]interface{}{
			"connect":               factory.Connect,
			"generateFloatVectors":  GenerateFloatVectors,
			"generateSparseVectors": GenerateSparseVectors,
		},
	}}
}

type moduleInstance struct {
	exports modules.Exports
}

func (i *moduleInstance) Exports() modules.Exports {
	return i.exports
}

type ClientFactory struct{}

func (f *ClientFactory) Connect(input map[string]interface{}) (*Client, error) {
	address := getString(input, "address", "")
	if address == "" {
		address = getString(input, "uri", "")
	}
	if address == "" {
		return nil, errors.New("address is required")
	}

	timeout := getDuration(input, "timeoutMs", 30*time.Second)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address:       address,
		Username:      getString(input, "username", ""),
		Password:      getString(input, "password", ""),
		DBName:        getString(input, "dbName", ""),
		APIKey:        getString(input, "apiKey", ""),
		EnableTLSAuth: getBool(input, "tls", false),
	})
	if err != nil {
		return nil, err
	}
	return &Client{client: cli, defaultTimeout: timeout}, nil
}

func GenerateFloatVectors(count int, dim int, seed int64) [][]float32 {
	r := rand.New(rand.NewSource(seed))
	vectors := make([][]float32, count)
	for i := range vectors {
		row := make([]float32, dim)
		for j := range row {
			row[j] = r.Float32()
		}
		vectors[i] = row
	}
	return vectors
}

func GenerateSparseVectors(count int, dim int, nnz int, seed int64) ([]entity.SparseEmbedding, error) {
	if dim <= 0 {
		return nil, errors.New("sparse vector dimension must be > 0")
	}
	if nnz <= 0 {
		return nil, errors.New("sparse vector nnz must be > 0")
	}
	if nnz > dim {
		return nil, fmt.Errorf("sparse vector nnz %d must be <= dimension %d", nnz, dim)
	}

	r := rand.New(rand.NewSource(seed))
	vectors := make([]entity.SparseEmbedding, count)
	for i := range vectors {
		seen := make(map[uint32]struct{}, nnz)
		positions := make([]uint32, 0, nnz)
		values := make([]float32, nnz)
		for len(positions) < nnz {
			pos := uint32(r.Intn(dim))
			if _, ok := seen[pos]; ok {
				continue
			}
			seen[pos] = struct{}{}
			positions = append(positions, pos)
			values[len(positions)-1] = r.Float32()
		}
		vector, err := entity.NewSliceSparseEmbedding(positions, values)
		if err != nil {
			return nil, err
		}
		vectors[i] = vector
	}
	return vectors, nil
}

type Client struct {
	client         *milvusclient.Client
	defaultTimeout time.Duration
}

func (c *Client) Close() error {
	ctx, cancel := context.WithTimeout(context.Background(), c.defaultTimeout)
	defer cancel()
	return c.client.Close(ctx)
}

func (c *Client) UseDatabase(input map[string]interface{}) error {
	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.UseDatabase(ctx, milvusclient.NewUseDatabaseOption(requiredString(input, "dbName")))
}

func (c *Client) ListDatabases(input map[string]interface{}) ([]map[string]interface{}, error) {
	ctx, cancel := c.context(input)
	defer cancel()
	dbs, err := c.client.ListDatabase(ctx, milvusclient.NewListDatabaseOption())
	if err != nil {
		return nil, err
	}
	out := make([]map[string]interface{}, 0, len(dbs))
	for _, db := range dbs {
		out = append(out, map[string]interface{}{"name": db})
	}
	return out, nil
}

func (c *Client) CreateDatabase(input map[string]interface{}) error {
	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.CreateDatabase(ctx, milvusclient.NewCreateDatabaseOption(requiredString(input, "dbName")))
}

func (c *Client) DropDatabase(input map[string]interface{}) error {
	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.DropDatabase(ctx, milvusclient.NewDropDatabaseOption(requiredString(input, "dbName")))
}

func (c *Client) CheckHealth(input map[string]interface{}) (map[string]interface{}, error) {
	version, err := c.GetVersion(input)
	return map[string]interface{}{"isHealthy": err == nil, "version": version}, err
}

func (c *Client) GetVersion(input map[string]interface{}) (string, error) {
	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.GetServerVersion(ctx, milvusclient.NewGetServerVersionOption())
}

func (c *Client) NewCollection(input map[string]interface{}) error {
	name := requiredString(input, "name")
	opt := milvusclient.SimpleCreateCollectionOptions(name, int64(requiredInt(input, "dimension"))).
		WithPKFieldName(getString(input, "primaryField", "id")).
		WithVectorFieldName(getString(input, "vectorField", "vector")).
		WithMetricType(parseMetric(getString(input, "metricType", "COSINE"))).
		WithAutoID(getBool(input, "autoID", false)).
		WithDynamicSchema(getBool(input, "enableDynamic", true)).
		WithShardNum(int32(getInt(input, "shardsNum", 1)))
	if strings.EqualFold(getString(input, "primaryFieldType", "int64"), "varchar") {
		opt.WithVarcharPK(true, getInt(input, "primaryMaxLength", 256))
	}

	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.CreateCollection(ctx, opt)
}

func (c *Client) CreateCollection(input map[string]interface{}) error {
	name := requiredString(input, "name")
	schema := entity.NewSchema().
		WithName(name).
		WithAutoID(getBool(input, "autoID", false)).
		WithDynamicFieldEnabled(getBool(input, "enableDynamic", false))

	rawFields, ok := input["fields"].([]interface{})
	if !ok || len(rawFields) == 0 {
		return errors.New("fields must be a non-empty array")
	}
	for _, raw := range rawFields {
		fieldMap, ok := raw.(map[string]interface{})
		if !ok {
			return fmt.Errorf("field must be an object, got %T", raw)
		}
		field, err := buildField(fieldMap)
		if err != nil {
			return err
		}
		schema.WithField(field)
	}

	opt := milvusclient.NewCreateCollectionOption(name, schema).WithShardNum(int32(getInt(input, "shardsNum", 1)))
	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.CreateCollection(ctx, opt)
}

func (c *Client) HasCollection(input map[string]interface{}) (bool, error) {
	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.HasCollection(ctx, milvusclient.NewHasCollectionOption(requiredString(input, "name")))
}

func (c *Client) ListCollections(input map[string]interface{}) ([]map[string]interface{}, error) {
	ctx, cancel := c.context(input)
	defer cancel()
	names, err := c.client.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		return nil, err
	}
	out := make([]map[string]interface{}, 0, len(names))
	for _, name := range names {
		out = append(out, map[string]interface{}{"name": name})
	}
	return out, nil
}

func (c *Client) DescribeCollection(input map[string]interface{}) (map[string]interface{}, error) {
	ctx, cancel := c.context(input)
	defer cancel()
	coll, err := c.client.DescribeCollection(ctx, milvusclient.NewDescribeCollectionOption(requiredString(input, "name")))
	if err != nil {
		return nil, err
	}
	fields := make([]map[string]interface{}, 0, len(coll.Schema.Fields))
	for _, field := range coll.Schema.Fields {
		fields = append(fields, map[string]interface{}{
			"name":       field.Name,
			"type":       field.DataType.Name(),
			"primaryKey": field.PrimaryKey,
			"autoID":     field.AutoID,
			"typeParams": field.TypeParams,
		})
	}
	return map[string]interface{}{"id": coll.ID, "name": coll.Name, "fields": fields}, nil
}

func (c *Client) DropCollection(input map[string]interface{}) error {
	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.DropCollection(ctx, milvusclient.NewDropCollectionOption(requiredString(input, "name")))
}

func (c *Client) CreatePartition(input map[string]interface{}) error {
	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.CreatePartition(ctx, milvusclient.NewCreatePartitionOption(requiredString(input, "collection"), requiredString(input, "partition")))
}

func (c *Client) DropPartition(input map[string]interface{}) error {
	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.DropPartition(ctx, milvusclient.NewDropPartitionOption(requiredString(input, "collection"), requiredString(input, "partition")))
}

func (c *Client) LoadCollection(input map[string]interface{}) error {
	ctx, cancel := c.context(input)
	defer cancel()
	task, err := c.client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(requiredString(input, "name")))
	if err != nil || getBool(input, "async", false) {
		return err
	}
	return task.Await(ctx)
}

func (c *Client) ReleaseCollection(input map[string]interface{}) error {
	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.ReleaseCollection(ctx, milvusclient.NewReleaseCollectionOption(requiredString(input, "name")))
}

func (c *Client) GetLoadingProgress(input map[string]interface{}) (int64, error) {
	ctx, cancel := c.context(input)
	defer cancel()
	state, err := c.client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(requiredString(input, "name"), getStringSlice(input, "partitions")...))
	if err != nil {
		return 0, err
	}
	return state.Progress, nil
}

func (c *Client) GetLoadState(input map[string]interface{}) (string, error) {
	ctx, cancel := c.context(input)
	defer cancel()
	state, err := c.client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(requiredString(input, "name"), getStringSlice(input, "partitions")...))
	return fmt.Sprint(state.State), err
}

func (c *Client) CreateIndex(input map[string]interface{}) error {
	params := stringMap(getMap(input, "params"))
	params["index_type"] = getString(input, "indexType", "AUTOINDEX")
	params["metric_type"] = getString(input, "metricType", "COSINE")
	idx := index.NewGenericIndex(getString(input, "indexName", ""), params)
	ctx, cancel := c.context(input)
	defer cancel()
	task, err := c.client.CreateIndex(ctx, milvusclient.NewCreateIndexOption(requiredString(input, "collection"), getString(input, "field", "vector"), idx))
	if err != nil || getBool(input, "async", false) {
		return err
	}
	return task.Await(ctx)
}

func (c *Client) DropIndex(input map[string]interface{}) error {
	ctx, cancel := c.context(input)
	defer cancel()
	return c.client.DropIndex(ctx, milvusclient.NewDropIndexOption(requiredString(input, "collection"), getString(input, "indexName", "")))
}

func (c *Client) Flush(input map[string]interface{}) error {
	ctx, cancel := c.context(input)
	defer cancel()
	task, err := c.client.Flush(ctx, milvusclient.NewFlushOption(requiredString(input, "collection")))
	if err != nil || getBool(input, "async", false) {
		return err
	}
	return task.Await(ctx)
}

func (c *Client) Insert(input map[string]interface{}) (map[string]interface{}, error) {
	columns, err := buildColumns(input)
	if err != nil {
		return nil, err
	}
	opt := milvusclient.NewColumnBasedInsertOption(requiredString(input, "collection"), columns...)
	if partition := getString(input, "partition", ""); partition != "" {
		opt.WithPartition(partition)
	}
	ctx, cancel := c.context(input)
	defer cancel()
	result, err := c.client.Insert(ctx, opt)
	if err != nil {
		return nil, err
	}
	out := map[string]interface{}{"count": result.InsertCount}
	if getBool(input, "returnIDs", false) {
		out["ids"] = columnToValues(result.IDs)
	}
	return out, nil
}

func (c *Client) InsertGenerated(input map[string]interface{}) (map[string]interface{}, error) {
	columns, err := buildGeneratedColumns(input)
	if err != nil {
		return nil, err
	}
	opt := milvusclient.NewColumnBasedInsertOption(requiredString(input, "collection"), columns...)
	if partition := getString(input, "partition", ""); partition != "" {
		opt.WithPartition(partition)
	}
	ctx, cancel := c.context(input)
	defer cancel()
	result, err := c.client.Insert(ctx, opt)
	if err != nil {
		return nil, err
	}
	out := map[string]interface{}{"count": result.InsertCount}
	if getBool(input, "returnIDs", false) {
		out["ids"] = columnToValues(result.IDs)
	}
	return out, nil
}

func (c *Client) Upsert(input map[string]interface{}) (map[string]interface{}, error) {
	columns, err := buildColumns(input)
	if err != nil {
		return nil, err
	}
	opt := milvusclient.NewColumnBasedInsertOption(requiredString(input, "collection"), columns...)
	if partition := getString(input, "partition", ""); partition != "" {
		opt.WithPartition(partition)
	}
	ctx, cancel := c.context(input)
	defer cancel()
	result, err := c.client.Upsert(ctx, opt)
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{"ids": columnToValues(result.IDs), "count": result.UpsertCount}, nil
}

func (c *Client) Search(input map[string]interface{}) ([]map[string]interface{}, error) {
	vectors, err := getVectors(input)
	if err != nil {
		return nil, err
	}
	opt := milvusclient.NewSearchOption(requiredString(input, "collection"), requiredInt(input, "topK"), vectors).
		WithANNSField(getString(input, "vectorField", "vector")).
		WithFilter(getString(input, "expr", "")).
		WithOutputFields(getStringSlice(input, "outputFields")...).
		WithPartitions(getStringSlice(input, "partitions")...).
		WithIgnoreGrowing(getBool(input, "ignoreGrowing", false))
	if offset := getInt(input, "offset", 0); offset > 0 {
		opt.WithOffset(offset)
	}
	for key, value := range stringMap(getMap(input, "params")) {
		opt.WithSearchParam(key, value)
	}
	if nprobe := getInt(input, "nprobe", 0); nprobe > 0 {
		opt.WithSearchParam("nprobe", strconv.Itoa(nprobe))
	}
	if ef := getInt(input, "ef", 0); ef > 0 {
		opt.WithSearchParam("ef", strconv.Itoa(ef))
	}

	ctx, cancel := c.context(input)
	defer cancel()
	results, err := c.client.Search(ctx, opt)
	if err != nil {
		return nil, err
	}
	return searchResultsToMaps(results), nil
}

func (c *Client) Query(input map[string]interface{}) (map[string]interface{}, error) {
	opt := milvusclient.NewQueryOption(requiredString(input, "collection")).
		WithFilter(requiredString(input, "expr")).
		WithOutputFields(getStringSlice(input, "outputFields")...).
		WithPartitions(getStringSlice(input, "partitions")...)
	if limit := getInt(input, "limit", 0); limit > 0 {
		opt.WithLimit(limit)
	}
	if offset := getInt(input, "offset", 0); offset > 0 {
		opt.WithOffset(offset)
	}

	ctx, cancel := c.context(input)
	defer cancel()
	rs, err := c.client.Query(ctx, opt)
	if err != nil {
		return nil, err
	}
	return resultSetToMap(rs), nil
}

func (c *Client) Delete(input map[string]interface{}) (map[string]interface{}, error) {
	opt := milvusclient.NewDeleteOption(requiredString(input, "collection")).WithExpr(requiredString(input, "expr"))
	if partition := getString(input, "partition", ""); partition != "" {
		opt.WithPartition(partition)
	}
	ctx, cancel := c.context(input)
	defer cancel()
	result, err := c.client.Delete(ctx, opt)
	return map[string]interface{}{"count": result.DeleteCount}, err
}

func (c *Client) DeleteByPks(input map[string]interface{}) (map[string]interface{}, error) {
	opt := milvusclient.NewDeleteOption(requiredString(input, "collection"))
	field := getString(input, "primaryField", "id")
	if strings.EqualFold(getString(input, "primaryFieldType", "int64"), "varchar") {
		values, err := toStringSlice(input["ids"])
		if err != nil {
			return nil, err
		}
		opt.WithStringIDs(field, values)
	} else {
		values, err := toInt64Slice(input["ids"])
		if err != nil {
			return nil, err
		}
		opt.WithInt64IDs(field, values)
	}
	if partition := getString(input, "partition", ""); partition != "" {
		opt.WithPartition(partition)
	}
	ctx, cancel := c.context(input)
	defer cancel()
	result, err := c.client.Delete(ctx, opt)
	return map[string]interface{}{"count": result.DeleteCount}, err
}

func (c *Client) context(input map[string]interface{}) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), getDuration(input, "timeoutMs", c.defaultTimeout))
}

func buildField(input map[string]interface{}) (*entity.Field, error) {
	ft, err := parseFieldType(getString(input, "type", ""))
	if err != nil {
		return nil, err
	}
	field := entity.NewField().
		WithName(requiredString(input, "name")).
		WithDataType(ft).
		WithIsPrimaryKey(getBool(input, "primaryKey", false)).
		WithIsAutoID(getBool(input, "autoID", false)).
		WithIsDynamic(getBool(input, "dynamic", false)).
		WithIsPartitionKey(getBool(input, "partitionKey", false)).
		WithIsClusteringKey(getBool(input, "clusteringKey", false)).
		WithNullable(getBool(input, "nullable", false))
	if dim := getInt(input, "dimension", 0); dim > 0 {
		field.WithDim(int64(dim))
	}
	if maxLength := getInt(input, "maxLength", 0); maxLength > 0 {
		field.WithMaxLength(int64(maxLength))
	}
	return field, nil
}

func buildColumns(input map[string]interface{}) ([]column.Column, error) {
	rawColumns, ok := input["columns"].(map[string]interface{})
	if !ok || len(rawColumns) == 0 {
		return nil, errors.New("columns must be a non-empty object")
	}
	types := stringMap(getMap(input, "types"))
	dims := intMap(getMap(input, "dimensions"))

	columns := make([]column.Column, 0, len(rawColumns))
	for name, value := range rawColumns {
		fieldType := strings.ToLower(types[name])
		if fieldType == "" {
			fieldType = inferColumnType(value)
		}
		col, err := buildColumn(name, fieldType, dims[name], value)
		if err != nil {
			return nil, fmt.Errorf("column %s: %w", name, err)
		}
		columns = append(columns, col)
	}
	return columns, nil
}

func buildGeneratedColumns(input map[string]interface{}) ([]column.Column, error) {
	count := requiredInt(input, "count")
	rawSpecs, ok := input["columns"].([]interface{})
	if !ok || len(rawSpecs) == 0 {
		rawSpecs = legacyGeneratedColumnSpecs(input)
	}

	columns := make([]column.Column, 0, len(rawSpecs))
	for _, raw := range rawSpecs {
		spec, ok := raw.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("generated column spec must be an object, got %T", raw)
		}
		col, err := buildGeneratedColumn(count, spec)
		if err != nil {
			return nil, err
		}
		columns = append(columns, col)
	}
	return columns, nil
}

func legacyGeneratedColumnSpecs(input map[string]interface{}) []interface{} {
	return []interface{}{
		map[string]interface{}{
			"name":      getString(input, "primaryField", "id"),
			"type":      "int64",
			"generator": "sequence",
			"start":     getInt(input, "startID", 0),
		},
		map[string]interface{}{
			"name":      getString(input, "vectorField", "vector"),
			"type":      "float_vector",
			"generator": "random_vector",
			"dimension": requiredInt(input, "dimension"),
			"seed":      getInt(input, "seed", 1),
		},
	}
}

func buildGeneratedColumn(count int, spec map[string]interface{}) (column.Column, error) {
	name := requiredString(spec, "name")
	fieldType := strings.ToLower(getString(spec, "type", ""))
	generator := strings.ToLower(getString(spec, "generator", ""))
	if generator == "" {
		return nil, fmt.Errorf("generated column %s: generator is required", name)
	}

	switch fieldType {
	case "int64", "long":
		values, err := generateInt64Values(count, spec, generator)
		return column.NewColumnInt64(name, values), err
	case "int32", "int":
		values, err := generateInt32Values(count, spec, generator)
		return column.NewColumnInt32(name, values), err
	case "float", "float32":
		values, err := generateFloat32Values(count, spec, generator)
		return column.NewColumnFloat(name, values), err
	case "double", "float64":
		values, err := generateFloat64Values(count, spec, generator)
		return column.NewColumnDouble(name, values), err
	case "varchar", "string":
		values, err := generateStringValues(count, spec, generator)
		return column.NewColumnVarChar(name, values), err
	case "json":
		values, err := generateJSONValues(count, spec, generator)
		return column.NewColumnJSONBytes(name, values), err
	case "floatvector", "float_vector", "vector":
		values, err := generateFloatVectorValues(count, spec, generator)
		if err != nil {
			return nil, err
		}
		return column.NewColumnFloatVector(name, requiredInt(spec, "dimension"), values), nil
	case "sparsefloatvector", "sparse_float_vector", "sparse_vector", "sparse":
		values, err := generateSparseVectorValues(count, spec, generator)
		if err != nil {
			return nil, err
		}
		return column.NewColumnSparseVectors(name, values), nil
	default:
		return nil, fmt.Errorf("generated column %s: unsupported type %q", name, fieldType)
	}
}

func generateInt64Values(count int, spec map[string]interface{}, generator string) ([]int64, error) {
	switch generator {
	case "sequence":
		start, err := getInt64(spec, "start", 0)
		if err != nil {
			return nil, err
		}
		step, err := getInt64(spec, "step", 1)
		if err != nil {
			return nil, err
		}
		values := make([]int64, count)
		for i := range values {
			values[i] = start + int64(i)*step
		}
		return values, nil
	case "constant":
		value, err := getInt64(spec, "value", 0)
		if err != nil {
			return nil, err
		}
		values := make([]int64, count)
		for i := range values {
			values[i] = value
		}
		return values, nil
	case "random_int":
		minValue, err := getInt64(spec, "min", 0)
		if err != nil {
			return nil, err
		}
		maxValue, err := getInt64(spec, "max", 1000000)
		if err != nil {
			return nil, err
		}
		if maxValue < minValue {
			return nil, errors.New("random_int max must be >= min")
		}
		r := rand.New(rand.NewSource(int64(getInt(spec, "seed", 1))))
		values := make([]int64, count)
		span := maxValue - minValue + 1
		for i := range values {
			values[i] = minValue + r.Int63n(span)
		}
		return values, nil
	default:
		return nil, fmt.Errorf("unsupported int64 generator %q", generator)
	}
}

func generateInt32Values(count int, spec map[string]interface{}, generator string) ([]int32, error) {
	values64, err := generateInt64Values(count, spec, generator)
	if err != nil {
		return nil, err
	}
	values := make([]int32, count)
	for i, value := range values64 {
		if value < math.MinInt32 || value > math.MaxInt32 {
			return nil, fmt.Errorf("int32 generated value out of range at row %d: %d", i, value)
		}
		values[i] = int32(value)
	}
	return values, nil
}

func generateFloat32Values(count int, spec map[string]interface{}, generator string) ([]float32, error) {
	values64, err := generateFloat64Values(count, spec, generator)
	if err != nil {
		return nil, err
	}
	values := make([]float32, count)
	for i, value := range values64 {
		values[i] = float32(value)
	}
	return values, nil
}

func generateFloat64Values(count int, spec map[string]interface{}, generator string) ([]float64, error) {
	switch generator {
	case "sequence":
		start := toFloat64(getValue(spec, "start", 0))
		step := toFloat64(getValue(spec, "step", 1))
		values := make([]float64, count)
		for i := range values {
			values[i] = start + float64(i)*step
		}
		return values, nil
	case "constant":
		value := toFloat64(getValue(spec, "value", 0))
		values := make([]float64, count)
		for i := range values {
			values[i] = value
		}
		return values, nil
	case "random_float":
		minValue := toFloat64(getValue(spec, "min", 0))
		maxValue := toFloat64(getValue(spec, "max", 1))
		if maxValue < minValue {
			return nil, errors.New("random_float max must be >= min")
		}
		r := rand.New(rand.NewSource(int64(getInt(spec, "seed", 1))))
		values := make([]float64, count)
		for i := range values {
			values[i] = minValue + r.Float64()*(maxValue-minValue)
		}
		return values, nil
	default:
		return nil, fmt.Errorf("unsupported float generator %q", generator)
	}
}

func generateStringValues(count int, spec map[string]interface{}, generator string) ([]string, error) {
	switch generator {
	case "constant":
		value := getString(spec, "value", "")
		values := make([]string, count)
		for i := range values {
			values[i] = value
		}
		return values, nil
	case "sequence":
		prefix := getString(spec, "prefix", "")
		start := getInt(spec, "start", 0)
		step := getInt(spec, "step", 1)
		values := make([]string, count)
		for i := range values {
			values[i] = fmt.Sprintf("%s%d", prefix, start+i*step)
		}
		return values, nil
	case "random_string", "random_text":
		prefix := getString(spec, "prefix", "")
		length := getInt(spec, "length", 16)
		if length < 0 {
			return nil, errors.New("random_string length must be >= 0")
		}
		charset := getString(spec, "charset", "abcdefghijklmnopqrstuvwxyz0123456789")
		if charset == "" && length > 0 {
			return nil, errors.New("random_string charset must not be empty when length > 0")
		}
		r := rand.New(rand.NewSource(int64(getInt(spec, "seed", 1))))
		values := make([]string, count)
		for i := range values {
			var b strings.Builder
			b.Grow(len(prefix) + length)
			b.WriteString(prefix)
			for j := 0; j < length; j++ {
				b.WriteByte(charset[r.Intn(len(charset))])
			}
			values[i] = b.String()
		}
		return values, nil
	default:
		return nil, fmt.Errorf("unsupported string generator %q", generator)
	}
}

func generateJSONValues(count int, spec map[string]interface{}, generator string) ([][]byte, error) {
	switch generator {
	case "constant":
		value := getValue(spec, "value", map[string]interface{}{})
		values := make([][]byte, count)
		for i := range values {
			b, err := json.Marshal(value)
			if err != nil {
				return nil, err
			}
			values[i] = b
		}
		return values, nil
	case "random_json":
		seed := int64(getInt(spec, "seed", 1))
		r := rand.New(rand.NewSource(seed))
		values := make([][]byte, count)
		for i := range values {
			b, err := json.Marshal(map[string]interface{}{
				"seq":    i,
				"bucket": getString(spec, "bucket", "k6"),
				"value":  r.Float64(),
			})
			if err != nil {
				return nil, err
			}
			values[i] = b
		}
		return values, nil
	default:
		return nil, fmt.Errorf("unsupported json generator %q", generator)
	}
}

func generateFloatVectorValues(count int, spec map[string]interface{}, generator string) ([][]float32, error) {
	switch generator {
	case "random_vector", "random":
		return GenerateFloatVectors(count, requiredInt(spec, "dimension"), int64(getInt(spec, "seed", 1))), nil
	case "constant":
		rawVector, ok := spec["value"]
		if !ok {
			return nil, errors.New("constant vector generator requires value")
		}
		vector, err := toFloat32Slice(rawVector)
		if err != nil {
			return nil, err
		}
		dim := requiredInt(spec, "dimension")
		if len(vector) != dim {
			return nil, fmt.Errorf("constant vector length %d does not match dimension %d", len(vector), dim)
		}
		values := make([][]float32, count)
		for i := range values {
			row := make([]float32, len(vector))
			copy(row, vector)
			values[i] = row
		}
		return values, nil
	default:
		return nil, fmt.Errorf("unsupported float_vector generator %q", generator)
	}
}

func generateSparseVectorValues(count int, spec map[string]interface{}, generator string) ([]entity.SparseEmbedding, error) {
	switch generator {
	case "random_sparse_vector", "random_sparse", "random":
		return GenerateSparseVectors(count, requiredInt(spec, "dimension"), requiredInt(spec, "nnz"), int64(getInt(spec, "seed", 1)))
	case "constant":
		rawVector, ok := spec["value"]
		if !ok {
			return nil, errors.New("constant sparse vector generator requires value")
		}
		vector, err := toSparseVector(rawVector)
		if err != nil {
			return nil, err
		}
		values := make([]entity.SparseEmbedding, count)
		for i := range values {
			values[i] = vector
		}
		return values, nil
	default:
		return nil, fmt.Errorf("unsupported sparse_float_vector generator %q", generator)
	}
}

func buildColumn(name string, fieldType string, dim int, value interface{}) (column.Column, error) {
	switch strings.ToLower(fieldType) {
	case "int64", "long":
		values, err := toInt64Slice(value)
		return column.NewColumnInt64(name, values), err
	case "int32", "int":
		values, err := toInt32Slice(value)
		return column.NewColumnInt32(name, values), err
	case "float", "float32":
		values, err := toFloat32Slice(value)
		return column.NewColumnFloat(name, values), err
	case "double", "float64":
		values, err := toFloat64Slice(value)
		return column.NewColumnDouble(name, values), err
	case "varchar", "string":
		values, err := toStringSlice(value)
		return column.NewColumnVarChar(name, values), err
	case "json":
		values, err := toJSONBytes(value)
		return column.NewColumnJSONBytes(name, values), err
	case "floatvector", "float_vector", "vector":
		values, err := toFloatVectors(value)
		if err != nil {
			return nil, err
		}
		if dim == 0 && len(values) > 0 {
			dim = len(values[0])
		}
		return column.NewColumnFloatVector(name, dim, values), nil
	case "sparsefloatvector", "sparse_float_vector", "sparse_vector", "sparse":
		values, err := toSparseVectors(value)
		if err != nil {
			return nil, err
		}
		return column.NewColumnSparseVectors(name, values), nil
	default:
		return nil, fmt.Errorf("unsupported column type %q", fieldType)
	}
}

func getVectors(input map[string]interface{}) ([]entity.Vector, error) {
	vectorType := strings.ToLower(getString(input, "vectorType", getString(input, "type", "")))
	if isSparseVectorType(vectorType) || input["sparseVectors"] != nil {
		var values []entity.SparseEmbedding
		if input["sparseVectors"] != nil {
			var err error
			values, err = toSparseVectors(input["sparseVectors"])
			if err != nil {
				return nil, err
			}
		} else {
			var err error
			values, err = GenerateSparseVectors(getInt(input, "nq", 1), requiredInt(input, "dimension"), requiredInt(input, "nnz"), int64(getInt(input, "seed", 1)))
			if err != nil {
				return nil, err
			}
		}
		vectors := make([]entity.Vector, 0, len(values))
		for _, value := range values {
			vectors = append(vectors, value)
		}
		return vectors, nil
	}

	var values [][]float32
	if input["vectors"] != nil {
		var err error
		values, err = toFloatVectors(input["vectors"])
		if err != nil {
			return nil, err
		}
	} else {
		values = GenerateFloatVectors(getInt(input, "nq", 1), requiredInt(input, "dimension"), int64(getInt(input, "seed", 1)))
	}
	vectors := make([]entity.Vector, 0, len(values))
	for _, value := range values {
		vectors = append(vectors, entity.FloatVector(value))
	}
	return vectors, nil
}

func searchResultsToMaps(results []milvusclient.ResultSet) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(results))
	for _, result := range results {
		item := map[string]interface{}{
			"count":  result.ResultCount,
			"ids":    columnToValues(result.IDs),
			"scores": result.Scores,
			"fields": dataSetToPlainMap(result.Fields),
		}
		if result.Err != nil {
			item["error"] = result.Err.Error()
		}
		out = append(out, item)
	}
	return out
}

func resultSetToMap(rs milvusclient.ResultSet) map[string]interface{} {
	return map[string]interface{}{"count": rs.Len(), "fields": dataSetToPlainMap(rs.Fields)}
}

func dataSetToPlainMap(ds milvusclient.DataSet) map[string]interface{} {
	out := make(map[string]interface{}, len(ds))
	for _, col := range ds {
		out[col.Name()] = columnToValues(col)
	}
	return out
}

func columnToValues(col column.Column) []interface{} {
	if col == nil {
		return nil
	}
	values := make([]interface{}, 0, col.Len())
	for i := 0; i < col.Len(); i++ {
		value, err := col.Get(i)
		if err != nil {
			values = append(values, nil)
			continue
		}
		values = append(values, value)
	}
	return values
}

func parseMetric(value string) entity.MetricType {
	switch strings.ToUpper(value) {
	case "L2":
		return entity.L2
	case "IP":
		return entity.IP
	default:
		return entity.COSINE
	}
}

func parseFieldType(value string) (entity.FieldType, error) {
	switch strings.ToLower(value) {
	case "bool":
		return entity.FieldTypeBool, nil
	case "int8":
		return entity.FieldTypeInt8, nil
	case "int16":
		return entity.FieldTypeInt16, nil
	case "int32", "int":
		return entity.FieldTypeInt32, nil
	case "int64", "long":
		return entity.FieldTypeInt64, nil
	case "float", "float32":
		return entity.FieldTypeFloat, nil
	case "double", "float64":
		return entity.FieldTypeDouble, nil
	case "varchar", "string":
		return entity.FieldTypeVarChar, nil
	case "json":
		return entity.FieldTypeJSON, nil
	case "floatvector", "float_vector", "vector":
		return entity.FieldTypeFloatVector, nil
	case "sparsefloatvector", "sparse_float_vector", "sparse_vector", "sparse":
		return entity.FieldTypeSparseVector, nil
	default:
		return entity.FieldTypeNone, fmt.Errorf("unsupported field type %q", value)
	}
}

func inferColumnType(value interface{}) string {
	if rows, ok := value.([]interface{}); ok && len(rows) > 0 {
		if _, ok := rows[0].([]interface{}); ok {
			return "floatVector"
		}
		if row, ok := rows[0].(map[string]interface{}); ok && looksLikeSparseVector(row) {
			return "sparseFloatVector"
		}
	}
	return "varchar"
}

func isSparseVectorType(value string) bool {
	switch strings.ToLower(value) {
	case "sparsefloatvector", "sparse_float_vector", "sparse_vector", "sparse":
		return true
	default:
		return false
	}
}

func looksLikeSparseVector(row map[string]interface{}) bool {
	if _, ok := row["values"]; ok {
		if _, hasIndices := row["indices"]; hasIndices {
			return true
		}
		if _, hasPositions := row["positions"]; hasPositions {
			return true
		}
	}
	if len(row) == 0 {
		return false
	}
	for key := range row {
		if _, err := strconv.ParseUint(key, 10, 32); err != nil {
			return false
		}
	}
	return true
}

func int64Range(start int64, count int) []int64 {
	values := make([]int64, count)
	for i := range values {
		values[i] = start + int64(i)
	}
	return values
}

func requiredString(input map[string]interface{}, key string) string {
	value := getString(input, key, "")
	if value == "" {
		panic(fmt.Sprintf("%s is required", key))
	}
	return value
}

func requiredInt(input map[string]interface{}, key string) int {
	value := getInt(input, key, 0)
	if value == 0 {
		panic(fmt.Sprintf("%s is required", key))
	}
	return value
}

func getMap(input map[string]interface{}, key string) map[string]interface{} {
	if value, ok := input[key].(map[string]interface{}); ok {
		return value
	}
	return nil
}

func getValue(input map[string]interface{}, key string, fallback interface{}) interface{} {
	value, ok := input[key]
	if !ok || value == nil {
		return fallback
	}
	return value
}

func getString(input map[string]interface{}, key string, fallback string) string {
	value, ok := input[key]
	if !ok || value == nil {
		return fallback
	}
	if v, ok := value.(string); ok {
		return v
	}
	return fmt.Sprint(value)
}

func getBool(input map[string]interface{}, key string, fallback bool) bool {
	value, ok := input[key]
	if !ok || value == nil {
		return fallback
	}
	switch v := value.(type) {
	case bool:
		return v
	case string:
		parsed, err := strconv.ParseBool(v)
		if err == nil {
			return parsed
		}
	}
	return fallback
}

func getInt(input map[string]interface{}, key string, fallback int) int {
	value, ok := input[key]
	if !ok || value == nil {
		return fallback
	}
	switch v := value.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	case string:
		parsed, err := strconv.Atoi(v)
		if err == nil {
			return parsed
		}
	}
	return fallback
}

func getInt64(input map[string]interface{}, key string, fallback int64) (int64, error) {
	value, ok := input[key]
	if !ok || value == nil {
		return fallback, nil
	}
	switch v := value.(type) {
	case int:
		return int64(v), nil
	case int64:
		return v, nil
	case float64:
		if math.Trunc(v) != v {
			return 0, fmt.Errorf("%s must be an integer, got %v", key, v)
		}
		if v < float64(math.MinInt64) || v > float64(math.MaxInt64) {
			return 0, fmt.Errorf("%s out of int64 range: %v", key, v)
		}
		return int64(v), nil
	case string:
		parsed, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return 0, fmt.Errorf("%s must be int64 string: %w", key, err)
		}
		return parsed, nil
	default:
		return 0, fmt.Errorf("%s must be int64-compatible, got %T", key, value)
	}
}

func getDuration(input map[string]interface{}, key string, fallback time.Duration) time.Duration {
	ms := getInt(input, key, 0)
	if ms <= 0 {
		return fallback
	}
	return time.Duration(ms) * time.Millisecond
}

func getStringSlice(input map[string]interface{}, key string) []string {
	raw, ok := input[key]
	if !ok || raw == nil {
		return nil
	}
	values, err := toStringSlice(raw)
	if err != nil {
		return nil
	}
	return values
}

func stringMap(input map[string]interface{}) map[string]string {
	out := make(map[string]string, len(input))
	for k, v := range input {
		out[k] = fmt.Sprint(v)
	}
	return out
}

func intMap(input map[string]interface{}) map[string]int {
	out := make(map[string]int, len(input))
	for k, v := range input {
		out[k] = int(toFloat64(v))
	}
	return out
}

func toInt64Slice(value interface{}) ([]int64, error) {
	raw, ok := value.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected array, got %T", value)
	}
	out := make([]int64, len(raw))
	for i, item := range raw {
		out[i] = int64(toFloat64(item))
	}
	return out, nil
}

func toInt32Slice(value interface{}) ([]int32, error) {
	raw, ok := value.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected array, got %T", value)
	}
	out := make([]int32, len(raw))
	for i, item := range raw {
		out[i] = int32(toFloat64(item))
	}
	return out, nil
}

func toUint32Slice(value interface{}) ([]uint32, error) {
	raw, ok := value.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected array, got %T", value)
	}
	out := make([]uint32, len(raw))
	for i, item := range raw {
		parsed, err := toUint32(item)
		if err != nil {
			return nil, fmt.Errorf("index %d: %w", i, err)
		}
		out[i] = parsed
	}
	return out, nil
}

func toUint32(value interface{}) (uint32, error) {
	switch v := value.(type) {
	case int:
		if v < 0 {
			return 0, fmt.Errorf("must be >= 0, got %d", v)
		}
		return uint32(v), nil
	case int64:
		if v < 0 || v > math.MaxUint32 {
			return 0, fmt.Errorf("out of uint32 range: %d", v)
		}
		return uint32(v), nil
	case float64:
		if math.Trunc(v) != v {
			return 0, fmt.Errorf("must be an integer, got %v", v)
		}
		if v < 0 || v > float64(math.MaxUint32) {
			return 0, fmt.Errorf("out of uint32 range: %v", v)
		}
		return uint32(v), nil
	case string:
		parsed, err := strconv.ParseUint(v, 10, 32)
		if err != nil {
			return 0, fmt.Errorf("must be uint32 string: %w", err)
		}
		return uint32(parsed), nil
	default:
		return 0, fmt.Errorf("must be uint32-compatible, got %T", value)
	}
}

func toFloat32Slice(value interface{}) ([]float32, error) {
	raw, ok := value.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected array, got %T", value)
	}
	out := make([]float32, len(raw))
	for i, item := range raw {
		out[i] = float32(toFloat64(item))
	}
	return out, nil
}

func toFloat64Slice(value interface{}) ([]float64, error) {
	raw, ok := value.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected array, got %T", value)
	}
	out := make([]float64, len(raw))
	for i, item := range raw {
		out[i] = toFloat64(item)
	}
	return out, nil
}

func toStringSlice(value interface{}) ([]string, error) {
	raw, ok := value.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected array, got %T", value)
	}
	out := make([]string, len(raw))
	for i, item := range raw {
		out[i] = fmt.Sprint(item)
	}
	return out, nil
}

func toFloatVectors(value interface{}) ([][]float32, error) {
	switch v := value.(type) {
	case [][]float32:
		return v, nil
	case []interface{}:
		out := make([][]float32, len(v))
		for i, row := range v {
			converted, err := toFloat32Slice(row)
			if err != nil {
				return nil, err
			}
			out[i] = converted
		}
		return out, nil
	default:
		return nil, fmt.Errorf("expected vector array, got %T", value)
	}
}

func toSparseVectors(value interface{}) ([]entity.SparseEmbedding, error) {
	switch v := value.(type) {
	case []entity.SparseEmbedding:
		return v, nil
	case []interface{}:
		out := make([]entity.SparseEmbedding, len(v))
		for i, row := range v {
			converted, err := toSparseVector(row)
			if err != nil {
				return nil, fmt.Errorf("sparse vector row %d: %w", i, err)
			}
			out[i] = converted
		}
		return out, nil
	default:
		return nil, fmt.Errorf("expected sparse vector array, got %T", value)
	}
}

func toSparseVector(value interface{}) (entity.SparseEmbedding, error) {
	switch v := value.(type) {
	case entity.SparseEmbedding:
		return v, nil
	case map[string]interface{}:
		if rawValues, ok := v["values"]; ok {
			rawPositions, ok := v["indices"]
			if !ok {
				rawPositions = v["positions"]
			}
			if rawPositions == nil {
				return nil, errors.New("sparse vector object requires indices or positions")
			}
			positions, err := toUint32Slice(rawPositions)
			if err != nil {
				return nil, err
			}
			values, err := toFloat32Slice(rawValues)
			if err != nil {
				return nil, err
			}
			return newSparseEmbedding(positions, values)
		}
		positions := make([]uint32, 0, len(v))
		values := make([]float32, 0, len(v))
		for key, rawValue := range v {
			pos, err := strconv.ParseUint(key, 10, 32)
			if err != nil {
				return nil, fmt.Errorf("sparse vector map key %q must be uint32 index", key)
			}
			positions = append(positions, uint32(pos))
			values = append(values, float32(toFloat64(rawValue)))
		}
		return newSparseEmbedding(positions, values)
	case []interface{}:
		positions := make([]uint32, 0, len(v))
		values := make([]float32, 0, len(v))
		for i, rawPair := range v {
			pair, ok := rawPair.([]interface{})
			if !ok || len(pair) != 2 {
				return nil, fmt.Errorf("sparse vector pair %d must be [index, value]", i)
			}
			pos, err := toUint32(pair[0])
			if err != nil {
				return nil, fmt.Errorf("sparse vector pair %d index: %w", i, err)
			}
			positions = append(positions, pos)
			values = append(values, float32(toFloat64(pair[1])))
		}
		return newSparseEmbedding(positions, values)
	default:
		return nil, fmt.Errorf("expected sparse vector row, got %T", value)
	}
}

func newSparseEmbedding(positions []uint32, values []float32) (entity.SparseEmbedding, error) {
	if len(positions) == 0 {
		return nil, errors.New("sparse vector must contain at least one non-zero value")
	}
	if len(positions) != len(values) {
		return nil, fmt.Errorf("sparse vector positions length %d does not match values length %d", len(positions), len(values))
	}
	return entity.NewSliceSparseEmbedding(positions, values)
}

func toJSONBytes(value interface{}) ([][]byte, error) {
	raw, ok := value.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected array, got %T", value)
	}
	out := make([][]byte, len(raw))
	for i, item := range raw {
		if v, ok := item.(string); ok {
			out[i] = []byte(v)
			continue
		}
		b, err := json.Marshal(item)
		if err != nil {
			return nil, err
		}
		out[i] = b
	}
	return out, nil
}

func toFloat64(value interface{}) float64 {
	switch v := value.(type) {
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case float32:
		return float64(v)
	case float64:
		return v
	case string:
		parsed, _ := strconv.ParseFloat(v, 64)
		return parsed
	default:
		return 0
	}
}
