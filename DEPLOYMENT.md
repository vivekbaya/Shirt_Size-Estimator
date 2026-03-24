# Installation & Deployment Guide

## Quick Start (Development)

### Prerequisites
- Python 3.8+
- Node.js 14+
- MongoDB 4.4+
- Webcam

### 1. Install MongoDB

**Ubuntu/Debian:**
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod
```

**macOS:**
```bash
brew tap mongodb/brew
brew install mongodb-community@6.0
brew services start mongodb-community@6.0
```

**Windows:**
Download installer from: https://www.mongodb.com/try/download/community

### 2. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy template
cp .env.template .env

# Edit configuration (optional)
nano .env
```

### 4. Start Backend

```bash
cd backend
python main.py
```

Backend will run on: http://localhost:8000

### 5. Start Frontend

```bash
# In a new terminal
cd frontend
npm install
npm start
```

Frontend will open at: http://localhost:3000

## Production Deployment

### Docker Deployment

**1. Build Images**
```bash
docker-compose build
```

**2. Start Services**
```bash
docker-compose up -d
```

**3. Check Status**
```bash
docker-compose ps
docker-compose logs -f
```

**4. Stop Services**
```bash
docker-compose down
```

### Manual Production Deployment

**1. Install System Dependencies**
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv nginx certbot
```

**2. Setup Application**
```bash
# Create app directory
sudo mkdir -p /opt/shirt-size-estimator
sudo chown $USER:$USER /opt/shirt-size-estimator
cd /opt/shirt-size-estimator

# Clone or copy application files
# ... (copy your files here)

# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Create Systemd Service**

Create `/etc/systemd/system/shirt-size-api.service`:
```ini
[Unit]
Description=Shirt Size Estimation API
After=network.target mongodb.service

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/shirt-size-estimator
Environment="PATH=/opt/shirt-size-estimator/venv/bin"
ExecStart=/opt/shirt-size-estimator/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

**4. Enable and Start Service**
```bash
sudo systemctl daemon-reload
sudo systemctl enable shirt-size-api
sudo systemctl start shirt-size-api
sudo systemctl status shirt-size-api
```

**5. Setup Nginx Reverse Proxy**

Create `/etc/nginx/sites-available/shirt-size-estimator`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

**6. Enable Site**
```bash
sudo ln -s /etc/nginx/sites-available/shirt-size-estimator /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**7. Setup SSL (Optional)**
```bash
sudo certbot --nginx -d your-domain.com
```

**8. Build Frontend for Production**
```bash
cd frontend
npm run build

# Serve with nginx or copy to static directory
sudo cp -r build/* /var/www/html/
```

### Cloud Deployment (AWS Example)

**1. EC2 Instance Setup**
- Instance type: t3.medium or better
- AMI: Ubuntu 22.04 LTS
- Storage: 20GB minimum
- Security Group: Allow ports 22, 80, 443, 8000

**2. Install Dependencies**
```bash
ssh ubuntu@your-instance-ip
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
```

**3. Deploy with Docker**
```bash
# Copy files to instance
scp -r shirt-size-cv-system ubuntu@your-instance-ip:~

# SSH and start
ssh ubuntu@your-instance-ip
cd shirt-size-cv-system
sudo docker-compose up -d
```

**4. Setup Load Balancer (Optional)**
- Create Application Load Balancer
- Configure target group pointing to port 8000
- Setup health checks: /health endpoint

### Performance Tuning

**1. Gunicorn for Production**

Install:
```bash
pip install gunicorn
```

Update service:
```ini
ExecStart=/opt/shirt-size-estimator/venv/bin/gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**2. MongoDB Optimization**

```bash
# Connect to MongoDB
mongo

# Create indexes
use shirt_size_db
db.size_predictions.createIndex({ "session_id": 1 })
db.size_predictions.createIndex({ "timestamp": -1 })
db.size_predictions.createIndex({ "session_id": 1, "timestamp": -1 })

# Enable authentication (production)
db.createUser({
  user: "shirt_size_user",
  pwd: "strong_password_here",
  roles: [{ role: "readWrite", db: "shirt_size_db" }]
})
```

Update .env:
```
MONGODB_URL=mongodb://shirt_size_user:strong_password_here@localhost:27017/shirt_size_db
```

**3. Redis for Session Caching (Optional)**

Install Redis:
```bash
sudo apt-get install redis-server
```

Update application to use Redis for session state.

### Monitoring

**1. Setup Logging**

Create `/var/log/shirt-size-estimator/` directory:
```bash
sudo mkdir -p /var/log/shirt-size-estimator
sudo chown www-data:www-data /var/log/shirt-size-estimator
```

Update logging config in settings.py

**2. Prometheus Metrics (Optional)**

Install:
```bash
pip install prometheus-client
```

Add to main.py:
```python
from prometheus_client import Counter, Histogram, make_asgi_app

request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

**3. Health Monitoring**

Use tools like:
- UptimeRobot for uptime monitoring
- New Relic or DataDog for APM
- CloudWatch for AWS deployments

### Backup Strategy

**1. MongoDB Backup**

Daily backup script:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
mongodump --db shirt_size_db --out /backup/mongodb-$DATE
find /backup -mtime +7 -delete  # Keep 7 days
```

Add to crontab:
```bash
0 2 * * * /usr/local/bin/mongodb-backup.sh
```

**2. Application Backup**
```bash
tar -czf app-backup-$DATE.tar.gz /opt/shirt-size-estimator
```

### Security Checklist

- [ ] Enable MongoDB authentication
- [ ] Use SSL/TLS certificates
- [ ] Configure firewall (ufw or security groups)
- [ ] Regular security updates
- [ ] Rate limiting on API endpoints
- [ ] Input validation and sanitization
- [ ] CORS properly configured
- [ ] Secrets in environment variables, not code
- [ ] Regular backup testing
- [ ] Monitor logs for suspicious activity

### Scaling Considerations

**Horizontal Scaling:**
- Use load balancer (AWS ALB, nginx)
- Shared MongoDB instance
- Session affinity for WebSocket connections

**Vertical Scaling:**
- Increase EC2 instance size
- Add GPU for faster inference
- More RAM for larger models

**Database Scaling:**
- MongoDB replica set
- Sharding for large datasets
- Read replicas for analytics

## Troubleshooting

### Service Won't Start
```bash
# Check logs
sudo journalctl -u shirt-size-api -n 50
sudo journalctl -u mongod -n 50

# Check port availability
sudo netstat -tulpn | grep 8000

# Check file permissions
ls -la /opt/shirt-size-estimator
```

### High CPU Usage
- Enable frame skipping
- Reduce resolution
- Use GPU if available
- Check for memory leaks

### Database Connection Issues
```bash
# Test MongoDB connection
mongo --eval "db.stats()"

# Check MongoDB status
sudo systemctl status mongod

# View MongoDB logs
sudo tail -f /var/log/mongodb/mongod.log
```

## Support

For issues or questions:
- Check documentation: README.md, TECHNICAL_DOCS.md
- Review logs: /var/log/shirt-size-estimator/
- GitHub Issues: (your-repo-url)
