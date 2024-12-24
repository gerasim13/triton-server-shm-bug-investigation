# Steps to reproduce
### 1. Run container
```bash
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ./models:/models \
--shm-size=8589934592 \
--name=tritonserver \
nvcr.io/nvidia/tritonserver:24.12-py3 tritonserver \
--model-repository=/models \
--model-control-mode=explicit \
--repository-poll-secs=0 \
--cuda-memory-pool-byte-size=0:134217728 \
--pinned-memory-pool-byte-size=4294967296 \
--backend-config=python,shm-default-byte-size=1073741824 \
--cache-config=local,size=4294967296
```
### 2. Run inference, the server will create a new shared memory region during each iteration.
```bash
for i in $(seq 1 10);
do
  curl -X POST --location 'localhost:8000/v2/repository/models/dummy/load';
  curl -X POST --location 'localhost:8000/v2/repository/models/bls/load';
  curl --location 'localhost:8000/v2/models/bls/generate?dummy=[0-10]' --header 'Content-Type: application/json' --data '{"id": "dummy", "dummy": "0000"}' > /dev/null 2>&1 &;
  curl -X POST --location 'localhost:8000/v2/repository/models/dummy/unload';
  curl -X POST --location 'localhost:8000/v2/repository/models/bls/unload';
  docker exec tritonserver ls /dev/shm;
done
```