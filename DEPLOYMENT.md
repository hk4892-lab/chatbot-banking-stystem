# Deployment Guide

## Production Deployment Checklist

### Pre-Deployment

- [ ] Review and customize `.env` for production
- [ ] Build FAISS index from full KB: `make index`
- [ ] Run full test suite: `make test`
- [ ] Load test API endpoints
- [ ] Review audit log configuration
- [ ] Set up monitoring (Prometheus + Grafana)

### Environment Variables (Production)

```bash
# Models (consider hosting on local model server)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
SLM_MODEL=microsoft/phi-3-mini-4k-instruct

# Security
CORS_ORIGINS=https://yourdomain.com
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60

# Performance
DEFAULT_K=5
DEFAULT_TEMPERATURE=0.3

# Logging
LOG_LEVEL=INFO
AUDIT_LOG_PATH=/var/log/chatbot/audit.jsonl
```

### Docker Deployment

#### Single Container

```bash
# Build image
docker build -t banking-chatbot:v1.0 .

# Run
docker run -d \
  --name banking-chatbot \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /path/to/index:/app/index \
  -v /path/to/logs:/app/logs \
  -e CORS_ORIGINS=https://yourdomain.com \
  banking-chatbot:v1.0
```

#### Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Scale (if using load balancer)
docker-compose up -d --scale api=3
```

### Kubernetes Deployment

#### 1. Create ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chatbot-config
data:
  EMBEDDING_MODEL: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  SLM_MODEL: "microsoft/phi-3-mini-4k-instruct"
  DEFAULT_K: "5"
  DEFAULT_TEMPERATURE: "0.3"
```

#### 2. Create Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: banking-chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: banking-chatbot
  template:
    metadata:
      labels:
        app: banking-chatbot
    spec:
      containers:
      - name: api
        image: banking-chatbot:v1.0
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: chatbot-config
        volumeMounts:
        - name: index
          mountPath: /app/index
        - name: logs
          mountPath: /app/logs
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: index
        persistentVolumeClaim:
          claimName: chatbot-index-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: chatbot-logs-pvc
```

#### 3. Create Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: banking-chatbot
spec:
  type: LoadBalancer
  selector:
    app: banking-chatbot
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

### Cloud Deployment

#### AWS (ECS + Fargate)

1. Push image to ECR
2. Create ECS task definition
3. Create ECS service with load balancer
4. Configure CloudWatch for logs
5. Set up Auto Scaling

#### Google Cloud (Cloud Run)

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/banking-chatbot

# Deploy
gcloud run deploy banking-chatbot \
  --image gcr.io/PROJECT_ID/banking-chatbot \
  --platform managed \
  --memory 8Gi \
  --cpu 4 \
  --max-instances 10 \
  --port 8000 \
  --allow-unauthenticated
```

#### Azure (Container Instances)

```bash
# Create resource group
az group create --name chatbot-rg --location eastus

# Deploy container
az container create \
  --resource-group chatbot-rg \
  --name banking-chatbot \
  --image youracr.azurecr.io/banking-chatbot:v1.0 \
  --cpu 4 \
  --memory 8 \
  --dns-name-label banking-chatbot \
  --ports 8000
```

### Monitoring Setup

#### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'banking-chatbot'
    static_configs:
      - targets: ['chatbot-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

#### Grafana Dashboard

Import these panels:
- Request rate by endpoint
- Latency percentiles (P50, P95, P99)
- Refusal rate over time
- Error rate
- Model memory usage

### Backup & Recovery

#### Index Backup

```bash
# Backup index
tar -czf index-backup-$(date +%Y%m%d).tar.gz index/

# Restore
tar -xzf index-backup-20250115.tar.gz
```

#### Logs Rotation

```bash
# Add to crontab
0 0 * * * find /var/log/chatbot -name "audit-*.jsonl" -mtime +30 -delete
```

### Scaling Considerations

#### Horizontal Scaling
- API is stateless, scales easily
- Each instance loads own models
- Shared FAISS index (read-only)

#### Vertical Scaling
- More CPU → faster inference
- More RAM → larger batch processing
- GPU → 5-10x faster generation

#### Model Optimization
- Use quantized models (int8/int4)
- Consider model distillation
- Cache embeddings for common queries

### Security Hardening

1. **Network**:
   - Use HTTPS (TLS 1.3)
   - Restrict CORS origins
   - Enable rate limiting
   - Use API gateway

2. **Authentication** (if needed):
   - Add JWT middleware
   - API key validation
   - OAuth2 integration

3. **Secrets**:
   - Use secret manager (AWS Secrets Manager, etc.)
   - Never commit `.env`
   - Rotate credentials regularly

4. **Logging**:
   - PII scrubbing enforced
   - Log to secure storage
   - Monitor for anomalies

### Health Checks

```bash
# Liveness
curl -f http://api:8000/health || exit 1

# Readiness
curl -f http://api:8000/config || exit 1
```

### Disaster Recovery

1. **Backup Strategy**:
   - Daily index backups
   - Weekly full system backups
   - Logs retained 90 days

2. **Recovery Time**:
   - RTO: < 1 hour
   - RPO: < 24 hours

3. **Recovery Steps**:
   - Restore index from backup
   - Redeploy containers
   - Verify health endpoints
   - Run smoke tests

### Performance Tuning

#### CPU Optimization

```python
# Enable PyTorch optimizations
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(2)
```

#### Memory Management

```python
# Clear CUDA cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

#### Connection Pooling

```python
# In production, use connection pooling
app.add_middleware(
    DBSessionMiddleware,
    db_url="postgresql://...",
    pool_size=20
)
```

### Cost Optimization

1. **Use spot instances** for non-critical workloads
2. **Auto-scale down** during off-peak hours
3. **Cache frequent queries** (Redis)
4. **Use smaller models** if accuracy allows
5. **Quantize models** to int8 (2-4x cost reduction)

---

**Production Checklist**: Complete all items before go-live ✅
