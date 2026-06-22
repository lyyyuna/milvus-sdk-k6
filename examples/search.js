import milvus from 'k6/x/milvus';

export const options = {
  vus: 4,
  duration: '30s',
  thresholds: {
    checks: ['rate>0.99'],
  },
};

const address = __ENV.MILVUS_ADDR || 'localhost:19530';
const collection = __ENV.MILVUS_COLLECTION || 'k6_milvus_demo';
const dim = Number(__ENV.DIM || 128);

export function setup() {
  const c = milvus.connect({ address, timeoutMs: 30000 });

  if (!c.hasCollection({ name: collection })) {
    c.newCollection({
      name: collection,
      dimension: dim,
      primaryField: 'id',
      vectorField: 'vector',
      metricType: 'IP',
      enableDynamic: true,
    });
    c.insertGenerated({
      collection,
      count: 10000,
      dimension: dim,
      startID: 1,
      seed: 42,
    });
    c.flush({ collection });
    c.loadCollection({ name: collection });
  }

  c.close();
}

export default function () {
  const c = milvus.connect({ address, timeoutMs: 30000 });
  c.search({
    collection,
    dimension: dim,
    nq: 1,
    topK: 10,
    vectorField: 'vector',
    metricType: 'IP',
    outputFields: ['id'],
    seed: __VU * 100000 + __ITER,
  });
  c.close();
}
