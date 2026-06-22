import { check } from 'k6';
import milvus from 'k6/x/milvus';

export const options = {
  vus: Number(__ENV.VUS || 1),
  iterations: Number(__ENV.ITERATIONS || 1),
};

const address = __ENV.MILVUS_ADDR || '127.0.0.1:19530';
const collection = __ENV.MILVUS_COLLECTION || `k6_sparse_${Date.now()}`;
const dim = Number(__ENV.DIM || 100000);
const nnz = Number(__ENV.NNZ || 200);

export function setup() {
  const c = milvus.connect({ address, timeoutMs: 60000 });
  if (c.hasCollection({ name: collection })) {
    c.dropCollection({ name: collection });
  }
  c.createCollection({
    name: collection,
    autoID: false,
    enableDynamic: false,
    fields: [
      { name: 'id', type: 'int64', primaryKey: true },
      { name: 'tenant', type: 'varchar', maxLength: 64 },
      { name: 'sparse', type: 'sparse_float_vector' },
    ],
  });
  c.insertGenerated({
    collection,
    count: Number(__ENV.ROWS || 1000),
    returnIDs: false,
    columns: [
      { name: 'id', type: 'int64', generator: 'sequence', start: '1' },
      { name: 'tenant', type: 'varchar', generator: 'random_string', prefix: 'k6-', length: 12, seed: 2 },
      { name: 'sparse', type: 'sparse_float_vector', generator: 'random_sparse_vector', dimension: dim, nnz, seed: 5 },
    ],
  });
  c.flush({ collection, timeoutMs: 60000 });
  c.createIndex({
    collection,
    field: 'sparse',
    indexType: 'SPARSE_INVERTED_INDEX',
    metricType: 'IP',
    timeoutMs: 60000,
  });
  c.loadCollection({ name: collection, timeoutMs: 60000 });
  c.close();
  return { collection };
}

export default function (data) {
  const c = milvus.connect({ address, timeoutMs: 60000 });
  const hits = c.search({
    collection: data.collection,
    vectorType: 'sparse_float_vector',
    vectorField: 'sparse',
    metricType: 'IP',
    dimension: dim,
    nnz,
    nq: 1,
    topK: 10,
    seed: __VU * 100000 + __ITER,
    outputFields: ['id', 'tenant'],
  });
  check(hits, {
    'sparse search returns one result set': (r) => r.length === 1,
  });
  c.close();
}

export function teardown(data) {
  const c = milvus.connect({ address, timeoutMs: 60000 });
  c.releaseCollection({ name: data.collection });
  c.dropCollection({ name: data.collection });
  c.close();
}
