# Kubernetes Local Setup (minikube)

## Prerequisites

- [minikube](https://minikube.sigs.k8s.io/docs/start/)
- [Helm](https://helm.sh/docs/intro/install/)
- Docker

## Deploy

```bash
# Start minikube
minikube start --cpus=4 --memory=8192

# Build image inside minikube's Docker daemon
eval $(minikube docker-env)
docker build -t agent-bench:latest -f docker/Dockerfile .

# Deploy with dev values
helm install agent-bench k8s/helm/agent-bench/ \
  -f k8s/helm/agent-bench/values-dev.yaml \
  --set provider.selfhosted.modalEndpoint=$MODAL_VLLM_URL

# Verify
kubectl get pods
kubectl port-forward svc/agent-bench 8080:8000

# Test
curl http://localhost:8080/health
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I define a path parameter in FastAPI?"}'
```

## Teardown

```bash
helm uninstall agent-bench
minikube stop
```
