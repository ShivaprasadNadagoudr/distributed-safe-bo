# distributed-safe-bo
Distributed Safe Bayesian Optimization

### To run program
1. Start head node :
```bash
ray start --head --port=8888 --resources='{"resource2": 1}'
```
2. Connect worker-nodes to head node : 
```bash
ray start --address='172.20.46.18:8888' --redis-password='5241590000000000' --resources='{"resource1": 1}'
```
3. Run the python file : 
```bash
python hyperspaces.py
```

### dev-time commands
1. To connect worker-nodes to head node : 
```bash
bash setup-scripts/start-ray-clients.sh
``` 