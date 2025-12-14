#!/bin/bash
# ============================================================================
# Oracle 23ai RAG Chatbot - Setup Script
# ============================================================================

set -e

echo "=========================================="
echo "Oracle 23ai RAG Chatbot - Setup"
echo "=========================================="
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3.11 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.11 or higher required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Create virtual environment
echo
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Warning: venv already exists. Skipping creation.${NC}"
else
    python3.11 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create directories
echo
echo "Creating data directories..."
mkdir -p data/unprocessed
mkdir -p data/processed
touch data/unprocessed/.gitkeep
touch data/processed/.gitkeep
echo -e "${GREEN}✓ Directories created${NC}"

# Setup environment file
echo
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo -e "${YELLOW}⚠ Please edit .env file with your credentials${NC}"
else
    echo -e "${YELLOW}Warning: .env already exists. Skipping creation.${NC}"
fi

# Verify configuration
echo
echo "Verifying configuration..."

if [ -f "config.yaml" ]; then
    echo -e "${GREEN}✓ config.yaml found${NC}"
else
    echo -e "${RED}✗ config.yaml not found${NC}"
    exit 1
fi

if [ -f ".env" ]; then
    echo -e "${GREEN}✓ .env file found${NC}"
else
    echo -e "${RED}✗ .env file not found${NC}"
    exit 1
fi

# Test configuration loading
echo
echo "Testing configuration loading..."
python3 -c "from config_loader import get_config; get_config()" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Configuration loaded successfully${NC}"
else
    echo -e "${RED}✗ Configuration loading failed${NC}"
    echo -e "${YELLOW}Please check .env file and config.yaml${NC}"
    exit 1
fi

# Summary
echo
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo
echo "Next steps:"
echo "1. Edit .env file with your credentials:"
echo "   vi .env"
echo
echo "2. Update config.yaml if needed:"
echo "   vi config.yaml"
echo
echo "3. Create database tables:"
echo "   sqlplus ADMIN/password@dsn @create_tables.sql"
echo
echo "4. Place documents in data/unprocessed/"
echo
echo "5. Process documents:"
echo "   python process_documents.py"
echo
echo "6. Run application:"
echo "   streamlit run app.py"
echo
echo "For more information, see README.md"
echo

deactivate
