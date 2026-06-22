import { check } from 'k6';
import milvus from 'k6/x/milvus';

export const options = {
  vus: 1,
  iterations: 1,
};

const address = __ENV.MILVUS_ADDR || '127.0.0.1:19530';
const collection = __ENV.MILVUS_COLLECTION || `k6_generated_${Date.now()}`;

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
      { name: 'score', type: 'float' },
      { name: 'vector', type: 'float_vector', dimension: 32 },
    ],
  });
  c.insertGenerated({
    collection,
    count: 200,
    columns: [
      { name: 'id', type: 'int64', generator: 'sequence', start: 1 },
      { name: 'tenant', type: 'varchar', generator: 'constant', value: 'k6' },
      { name: 'score', type: 'float', generator: 'random_float', min: 0, max: 1, seed: 3 },
      { name: 'vector', type: 'float_vector', dimension: 32, generator: 'random_vector', seed: 5 },
    ],
  });
  c.flush({ collection, timeoutMs: 60000 });
  c.createIndex({
    collection,
    field: 'vector',
    indexType: 'AUTOINDEX',
    metricType: 'COSINE',
    timeoutMs: 60000,
  });
  c.loadCollection({ name: collection, timeoutMs: 60000 });
  c.close();
  return { collection };
}

export default function (data) {
  const c = milvus.connect({ address, timeoutMs: 60000 });
  const rows = c.query({
    collection: data.collection,
    expr: 'tenant == "k6" and id in [1, 2, 3]',
    outputFields: ['id', 'tenant', 'score'],
  });
  check(rows, {
    'generated schema query returns rows': (r) => r.count === 3,
  });
  c.close();
}

export function teardown(data) {
  const c = milvus.connect({ address, timeoutMs: 60000 });
  c.releaseCollection({ name: data.collection });
  c.dropCollection({ name: data.collection });
  c.close();
}
