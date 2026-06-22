import { check } from 'k6';
import milvus from 'k6/x/milvus';

export const options = {
  vus: 1,
  iterations: 1,
};

const address = __ENV.MILVUS_ADDR;
const collection = __ENV.MILVUS_COLLECTION || `k6_smoke_${Date.now()}`;
const dim = Number(__ENV.DIM || 16);

export function setup() {
  const c = milvus.connect({ address, timeoutMs: 60000 });
  if (c.hasCollection({ name: collection })) {
    c.dropCollection({ name: collection });
  }
  c.newCollection({
    name: collection,
    dimension: dim,
    primaryField: 'id',
    vectorField: 'vector',
    metricType: 'COSINE',
    autoID: false,
    enableDynamic: true,
  });
  c.insertGenerated({
    collection,
    count: 100,
    dimension: dim,
    startID: 1,
    seed: 7,
    primaryField: 'id',
    vectorField: 'vector',
  });
  c.flush({ collection, timeoutMs: 60000 });
  c.loadCollection({ name: collection, timeoutMs: 60000 });
  c.close();
  return { collection };
}

export default function (data) {
  const c = milvus.connect({ address, timeoutMs: 60000 });
  const hits = c.search({
    collection: data.collection,
    dimension: dim,
    nq: 1,
    topK: 5,
    vectorField: 'vector',
    metricType: 'COSINE',
    outputFields: ['id'],
    seed: 11,
    timeoutMs: 60000,
  });
  check(hits, {
    'search returns one result set': (r) => r.length === 1,
    'search returns hits': (r) => r[0].count > 0,
  });

  const rows = c.query({
    collection: data.collection,
    expr: 'id in [1, 2, 3]',
    outputFields: ['id'],
    timeoutMs: 60000,
  });
  check(rows, {
    'query returns rows': (r) => r.count === 3,
  });
  c.close();
}

export function teardown(data) {
  const c = milvus.connect({ address, timeoutMs: 60000 });
  c.releaseCollection({ name: data.collection });
  c.dropCollection({ name: data.collection });
  c.close();
}
