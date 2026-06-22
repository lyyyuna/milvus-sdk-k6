import { check } from 'k6';
import milvus from 'k6/x/milvus';

export const options = {
  vus: 1,
  iterations: 1,
};

export default function () {
  const c = milvus.connect({
    address: __ENV.MILVUS_ADDR,
    timeoutMs: 60000,
  });
  const version = c.getVersion({});
  check(version, {
    'server version is returned': (v) => typeof v === 'string' && v.length > 0,
  });
  c.close();
}
