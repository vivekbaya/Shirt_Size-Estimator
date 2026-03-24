# 🚀 QUICK START GUIDE

## Get Running in 5 Minutes

### Prerequisites Check
```bash
# Check Python version (need 3.8+)
python3 --version

# Check Node.js (need 14+)
node --version

# Check MongoDB
mongod --version
```

### Step 1: Start MongoDB
```bash
# Ubuntu/Debian
sudo systemctl start mongodb

# macOS
brew services start mongodb-community

# Windows
net start MongoDB
```

### Step 2: Backend Setup
```bash
# Navigate to project
cd shirt-size-cv-system

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Start backend server
cd backend
python main.py
```

Backend is now running on: **http://localhost:8000**

### Step 3: Frontend Setup (New Terminal)
```bash
# Navigate to frontend
cd shirt-size-cv-system/frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend opens automatically at: **http://localhost:3000**

### Step 4: Use the Application

1. **Grant Camera Access**: Click "Allow" when prompted
2. **Click "Start Camera"**: Begin video streaming
3. **Stand in Frame**: Face camera, stand 6-8 feet away
4. **View Results**: Real-time size appears on right panel

### Common Issues

**MongoDB won't start?**
```bash
sudo systemctl status mongodb
sudo systemctl restart mongodb
```

**Camera not working?**
- Grant browser camera permissions
- Close other apps using camera
- Try different browser

**Dependencies failing?**
```bash
# Update pip
pip install --upgrade pip

# Try with sudo (Linux)
sudo pip install -r requirements.txt
```

### Alternative: Docker (Easiest)

```bash
# One command to rule them all
docker-compose up -d

# View at http://localhost:3000
```

### Test Without Camera

```bash
# Use example script with test image
python examples/run_estimation.py path/to/photo.jpg
```

### Next Steps

- Read **README.md** for detailed features
- See **TECHNICAL_DOCS.md** for API reference
- Check **DEPLOYMENT.md** for production setup

### Support

Having issues? Check:
1. All prerequisites installed
2. Ports 8000, 3000, 27017 available
3. Camera permissions granted
4. Logs in terminal for error messages

**Enjoy your real-time shirt size estimator! 👕**
