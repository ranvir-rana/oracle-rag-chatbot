# Oracle 23ai RAG Chatbot

Enterprise-grade Retrieval-Augmented Generation (RAG) chatbot powered by Oracle 23ai Vector Database and OCI Generative AI.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Oracle 23ai](https://img.shields.io/badge/Oracle-23ai-red.svg)](https://www.oracle.com/database/23ai/)
[![OCI GenAI](https://img.shields.io/badge/OCI-GenAI-orange.svg)](https://www.oracle.com/artificial-intelligence/generative-ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)

## 🎯 Features

- **🔍 Intelligent Document Q&A** - RAG-based answers from your documents
- **🌐 Multilingual Support** - Works in English, Arabic, and 100+ languages
- **📊 Vector Search** - Oracle 23ai with HNSW indexing
- **🤖 Multiple LLMs** - Cohere Command R+, Llama 3, and more
- **⚙️ Template-Based Config** - YAML + environment variables
- **🔒 Enterprise Security** - Wallet-based DB connections, no hardcoded secrets
- **📈 Production Ready** - Comprehensive logging, error handling, type hints

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Git Commands](#git-commands-to-push-to-github)
- [License](#license)

## 🔧 Prerequisites

### Required
- **Python 3.11+** - [Download](https://www.python.org/downloads/)
- **Oracle 23ai Database** - Autonomous or On-Premise
- **OCI Account** - With Generative AI access
- **Oracle Wallet** - For Autonomous Database connections

### OCI Resources Needed
- Compartment OCID
- OCI GenAI service access (us-chicago-1 region recommended)
- API Key configured in `~/.oci/config`

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/oracle-rag-chatbot.git
cd oracle-rag-chatbot
```

### 2. Run Setup Script

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

This will:
- Create Python virtual environment
- Install all dependencies
- Create data directories
- Generate `.env` file from template

### 3. Configure Environment Variables

Edit `.env` file with your credentials:

```bash
vi .env
```

Add the following:

```bash
# Database Credentials
DB_PASSWORD=your_actual_db_password
WALLET_PASSWORD=your_actual_wallet_password

# OCI Configuration
OCI_COMPARTMENT_OCID=ocid1.compartment.oc1..your_compartment_ocid

# Optional: API Keys (only if using reranker)
COHERE_API_KEY=your_cohere_api_key
```

### 4. Update Configuration

Edit `config.yaml` with your specific settings:

```bash
vi config.yaml
```

**Required changes:**

```yaml
database:
  user: "ADMIN"                          # Your DB username
  dsn: "your_database_high"              # Your database DSN
  wallet:
    location: "/path/to/your/wallet"     # Path to Oracle wallet
    config_dir: "/path/to/your/wallet"   # Same as location

oci:
  compartment_ocid: "${OCI_COMPARTMENT_OCID}"  # Leave as-is (uses .env)
  endpoint: "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
```

### 5. Create Database Tables

```bash
sqlplus ADMIN/password@your_dsn @create_tables.sql
```

Or use SQL Developer / OCI Console to run `create_tables.sql`

### 6. Run Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run app.py
```

**Access the application:**
- Local: http://localhost:8501
- Network: http://YOUR_IP:8501

## ⚙️ Configuration

### Environment Variables (`.env`)

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `DB_PASSWORD` | Database password | Yes | `MySecurePass123` |
| `WALLET_PASSWORD` | Oracle wallet password | Yes | `WalletPass123` |
| `OCI_COMPARTMENT_OCID` | OCI Compartment ID | Yes | `ocid1.compartment.oc1..aaa...` |
| `COHERE_API_KEY` | Cohere API key (for reranker) | No | `abc123...` |

### Main Configuration (`config.yaml`)

#### Database Settings

```yaml
database:
  user: "ADMIN"
  password: "${DB_PASSWORD}"              # Uses .env variable
  dsn: "mydb_high"                        # From tnsnames.ora
  wallet:
    location: "/home/opc/wallet"
    password: "${WALLET_PASSWORD}"
    config_dir: "/home/opc/wallet"
```

#### Model Settings

```yaml
models:
  # Embedding Model
  embedding:
    type: "OCI"
    model_name: "cohere.embed-multilingual-v3.0"
    tokenizer: "Cohere/Cohere-embed-multilingual-v3.0"
    bits: 64
  
  # Generation Model
  generation:
    type: "OCI"
    default_model: "cohere.command-r-plus-08-2024"
    available_models:
      - "cohere.command-r-plus-08-2024"
      - "cohere.command-r-16k"
      - "meta.llama-3-70b-instruct"
```

**Available Models:**
- `cohere.command-r-plus-08-2024` (Recommended - Best quality)
- `cohere.command-r-16k` (Fast, 16K context)
- `meta.llama-3-70b-instruct` (Alternative)

#### RAG Settings

```yaml
rag:
  # Chunking
  chunking:
    enabled: true
    max_chunk_size: 1000        # Characters per chunk
    chunk_overlap: 100          # Overlap between chunks
  
  # Retrieval
  retrieval:
    top_k: 3                    # Documents to retrieve
    top_n: 3                    # After reranking (if enabled)
    similarity_threshold: 0.35  # Min similarity (0-1)
    enable_approximate: false   # HNSW indexing
  
  # Generation
  generation:
    max_tokens: 600             # Max response length
    temperature: 0.1            # Creativity (0-1)
    stream: false               # Streaming responses
```

**Tuning Tips:**
- **Better answers:** Increase `top_k` to 5-10
- **Faster queries:** Enable `enable_approximate: true`
- **More creative:** Increase `temperature` to 0.5-0.7
- **Longer answers:** Increase `max_tokens` to 1000+

## 📖 Usage

### Upload Documents

1. Click sidebar in web interface
2. Upload PDF, TXT, or CSV files
3. Wait for processing to complete
4. Start asking questions

**Supported formats:** PDF, TXT, CSV

### Ask Questions

**English:**
```
What is the net profit for Qatar National Bank in 2024?
```

**Arabic:**
```
ما هو صافي ربح بنك قطر الوطني في 2024؟
```

**Responses include:**
- Direct answer from documents
- Source references (file name, page number)
- Similarity scores

### Command Line Document Processing

```bash
# Place files in data/unprocessed/
cp your_document.pdf data/unprocessed/

# Process documents
python process_documents.py

# Files move to data/processed/ after processing
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Streamlit UI (app.py)           │
├─────────────────────────────────────────┤
│   Chat Engine (chat_engine.py)          │
│   ├── LLM (OCI GenAI)                   │
│   ├── Embeddings (Cohere Multi)         │
│   └── Memory (Token Buffer)             │
├─────────────────────────────────────────┤
│   Vector Store (oracle_vectorstore.py)  │
│   └── Oracle 23ai Database              │
├─────────────────────────────────────────┤
│   Document Processor                     │
│   ├── PDF Reader (pypdf)                │
│   ├── Chunker (SentenceSplitter)        │
│   └── Embedder (OCI GenAI)              │
├─────────────────────────────────────────┤
│   Database Manager (database.py)        │
│   └── Connection Pool (oracledb)        │
├─────────────────────────────────────────┤
│   Configuration (config_loader.py)      │
│   ├── YAML Parser                       │
│   └── Env Resolver                      │
└─────────────────────────────────────────┘
```

## 🐛 Troubleshooting

### Setup Issues

**Problem:** `Python 3.11 or higher required`

**Solution:** Update `setup.sh` to use correct Python:
```bash
python3.11 -m venv venv
```

**Problem:** `Database connection failed`

**Solution:** 
1. Verify wallet location in `config.yaml`
2. Test connection: `tnsping your_dsn`
3. Check credentials in `.env`

**Problem:** Documents not processing

**Solution:**
1. Check file format (PDF, TXT, CSV only)
2. Verify OCI GenAI access
3. Check logs: `tail -f app.log`

**Problem:** Poor answer quality

**Solution:**
1. Increase `top_k` in `config.yaml`
2. Lower `similarity_threshold`
3. Try different model

## 📁 Project Structure

```
oracle-rag-chatbot/
├── app.py                    # Streamlit UI
├── chat_engine.py            # RAG engine
├── config_loader.py          # Configuration management
├── database.py               # Database operations
├── oci_utils.py              # OCI utilities
├── oracle_vectorstore.py     # Vector store integration
├── process_documents.py      # Document processing
├── config.yaml               # Main configuration
├── .env.template             # Environment template
├── .env                      # Your credentials (not in git)
├── create_tables.sql         # Database schema
├── requirements.txt          # Python dependencies
├── setup.sh                  # Setup script
├── README.md                 # This file
└── .gitignore               # Git exclusions
```

## 📝 Git Commands to Push to GitHub

### First Time Setup

```bash
# 1. Initialize git (if not already done)
git init

# 2. Add all files
git add .

# 3. Commit
git commit -m "Initial commit: Oracle 23ai RAG Chatbot"

# 4. Create repository on GitHub (via web interface)
# Go to: https://github.com/new
# Repository name: oracle-rag-chatbot
# Description: Enterprise RAG chatbot with Oracle 23ai and OCI GenAI
# Public/Private: Your choice
# DO NOT initialize with README (we already have one)

# 5. Link local repo to GitHub
git remote add origin https://github.com/YOUR_USERNAME/oracle-rag-chatbot.git

# 6. Push to GitHub
git branch -M main
git push -u origin main
```

### Subsequent Updates

```bash
# 1. Check what changed
git status

# 2. Add changes
git add .

# 3. Commit with descriptive message
git commit -m "Update: describe your changes here"

# 4. Push to GitHub
git push
```

### Important: Verify Before Push

**Check `.gitignore` is working:**

```bash
# See what will be committed
git status

# Should NOT see these files:
# ❌ .env
# ❌ wallet/
# ❌ *.log
# ❌ data/
# ❌ venv/
# ❌ __pycache__/

# If you see them, they're not being ignored!
```

**If secrets were accidentally added:**

```bash
# Remove from git (keeps local copy)
git rm --cached .env
git rm --cached -r wallet/

# Commit the removal
git commit -m "Remove sensitive files"

# Push
git push
```

### Recommended: Add GitHub Topics

After pushing, add these topics to your repo (on GitHub web interface):

- `oracle`
- `oracle-database`
- `rag`
- `chatbot`
- `vector-database`
- `generative-ai`
- `llm`
- `streamlit`
- `python`

## 🔒 Security Best Practices

1. **Never commit `.env` file** - Contains secrets
2. **Use wallet auto-open** - No password in code
3. **Rotate API keys regularly**
4. **Restrict file permissions:**
   ```bash
   chmod 600 .env
   chmod 600 wallet/*
   ```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Oracle 23ai Vector Database
- OCI Generative AI Services
- LlamaIndex Framework
- Streamlit

---

**Built with ❤️ using Oracle 23ai, OCI GenAI, and LlamaIndex**

For questions or issues, please open a GitHub issue.
