# xk6 Milvus SDK

`k6/x/milvus` is an xk6 JavaScript extension backed by the Milvus Go SDK
(`github.com/milvus-io/milvus/client/v2`).
It is intended for Milvus load tests where the hot path should stay in Go
instead of moving large vector payloads through JavaScript.

Current pins:

- k6: `v1.8.0`
- Milvus Go client: `github.com/milvus-io/milvus/client/v2 v2.6.5`

## Build

```bash
go install go.k6.io/xk6/cmd/xk6@latest
xk6 build --with github.com/lyyyuna/milvus-sdk-k6=.
```

## Quick Start

```bash
MILVUS_ADDR=localhost:19530 ./k6 run examples/search.js
```

## API

```js
import milvus from 'k6/x/milvus';

const c = milvus.connect({
  address: 'localhost:19530',
  username: '',
  password: '',
  dbName: '',
  tls: false,
  timeoutMs: 30000,
});
```

Common methods:

- `close()`
- `checkHealth({})`, `getVersion({})`
- `useDatabase({ dbName })`, `listDatabases({})`, `createDatabase({ dbName })`, `dropDatabase({ dbName })`
- `newCollection({ name, dimension, primaryField, vectorField, metricType, autoID, enableDynamic })`
- `createCollection({ name, shardsNum, autoID, enableDynamic, fields })`
- `hasCollection({ name })`, `listCollections({})`, `describeCollection({ name })`, `dropCollection({ name })`
- `createPartition({ collection, partition })`, `dropPartition({ collection, partition })`
- `createIndex({ collection, field, indexType, params, async })`, `dropIndex({ collection, field })`
- `loadCollection({ name, async })`, `releaseCollection({ name })`, `getLoadingProgress({ name })`, `getLoadState({ name })`
- `insert({ collection, partition, columns, types, dimensions })`
- `insertGenerated({ collection, count, columns })`
- `insertGenerated({ collection, count, dimension, startID, seed, primaryField, vectorField })` legacy shortcut
- `upsert({ collection, partition, columns, types, dimensions })`
- `search({ collection, vectors, dimension, nq, seed, topK, vectorField, metricType, outputFields, expr, indexType, nprobe, ef })`
- `query({ collection, expr, outputFields, limit, offset })`
- `delete({ collection, expr })`, `deleteByPks({ collection, primaryField, primaryFieldType, ids })`
- `flush({ collection, async })`

For high-throughput tests, prefer `insertGenerated()` and generated search vectors
(`dimension` + `nq` + `seed`) so vector data is created inside Go.

Schema-driven generated insert:

```js
c.insertGenerated({
  collection: 'test',
  count: 1000,
  returnIDs: false,
  columns: [
    { name: 'id', type: 'int64', generator: 'sequence', start: '1000000' },
    { name: 'tenant', type: 'varchar', generator: 'constant', value: 'k6' },
    { name: 'age', type: 'int32', generator: 'random_int', min: 1, max: 100, seed: 2 },
    { name: 'score', type: 'float', generator: 'random_float', min: 0, max: 1, seed: 3 },
    { name: 'meta', type: 'json', generator: 'random_json', bucket: 'k6', seed: 4 },
    { name: 'vector', type: 'float_vector', dimension: 768, generator: 'random_vector', seed: 5 },
  ],
});
```

Supported generated types: `int64`, `int32`, `float`, `double`, `varchar`, `json`,
and `float_vector`. Supported generators: `sequence`, `constant`, `random_int`,
`random_float`, `random_json`, and `random_vector`.

`insertGenerated()` returns `{ count }` by default. Set `returnIDs: true` only when
you need generated primary keys back in JS; for load tests this avoids copying large
ID arrays over the Go/JS boundary.

For large integer ID ranges, pass `start`, `step`, `min`, `max`, or `value` as strings
so Go parses them as `int64` without relying on JavaScript number precision.

`checkHealth()` currently verifies connectivity by calling server version because
the current `client/v2` SDK does not expose a public `CheckHealth` wrapper.
