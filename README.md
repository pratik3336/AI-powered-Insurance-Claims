# ClaimGuard

**AI-Powered Insurance Claims Processing & Fraud Detection System**

I built ClaimGuard as a comprehensive solution for automating insurance claim reviews using artificial intelligence. This system combines computer vision, machine learning, and intelligent workflow automation to streamline the entire claims process from submission to approval or denial.

---

## What This System Does

ClaimGuard helps insurance claim adjusters review and process claims efficiently by:

- **Analyzing damage photos** using AI vision to estimate repair costs
- **Detecting potential fraud** through multi-model analysis and pattern matching
- **Automating document generation** for approvals and denials
- **Providing intelligent recommendations** based on real-time data analysis

The system handles the complete workflow: from selecting a claim to review, analyzing damage evidence, calculating fraud risk, and generating professional approval letters or denial emails.

---

## Key Features

### Intelligent Damage Assessment
I integrated both OpenAI's Vision API and a custom YOLOv8 model to analyze vehicle damage photos. The system:
- Identifies specific damaged parts (bumper, hood, doors, etc.)
- Estimates repair costs per component
- Provides detailed breakdowns of parts, labor, and materials
- Uses historical repair data (RAG) for accurate cost estimation

### Real-Time Fraud Detection
The fraud detection engine performs multi-layered analysis:
- Compares claimed amounts against actual damage estimates
- Checks for suspicious patterns (over-claiming, under-claiming, missing evidence)
- Validates policy coverage ratios
- Analyzes historical claim patterns using XGBoost ML models
- Auto-generates detailed denial reasons when fraud is detected

### Streamlined Workflow
I designed a step-by-step interface that mirrors how adjusters actually work:
1. **Select Claim** - Click any claim from the queue to auto-load details
2. **Analyze Damage** - Process photos to get detailed cost breakdowns
3. **Calculate Fraud Risk** - Real-time assessment with clear recommendations
4. **Make Decision** - Generate approval letters or rejection emails

### Professional Document Generation
The system automatically creates:
- **Approval Letters** with settlement breakdowns and payment timelines
- **Rejection Emails** with specific fraud indicators and appeal information

---

## Technology Stack

**Frontend:**
- Gradio for the interactive web interface
- Real-time image gallery and data visualization

**Backend & AI:**
- FastAPI for API services
- OpenAI Vision API (gpt-4o-mini) for damage analysis
- Custom YOLOv8 Nano model for fast, local damage detection
- XGBoost for fraud classification
- RAG system with historical repair cost database

**Data Storage:**
- PostgreSQL for claims, policies, and fraud scores (15,420 records)
- File system for damage evidence images (20,551 images from Roboflow dataset)

---

## Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL database
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ClaimGuard.git
cd ClaimGuard

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Initialize database
python scripts/init_db.py

# Load sample data
python scripts/ingest_kaggle_data.py
python scripts/assign_roboflow_images.py
```

### Running the Application

```bash
# Start the dashboard
python app/ui/claim_review_app.py

# Open your browser to
http://localhost:7860
```

---

## Project Structure

```
ClaimGuard/
├── app/
│   ├── core/              # Configuration and database connections
│   ├── models/            # SQLAlchemy database models
│   ├── services/          # Business logic
│   │   └── ai/           # AI services (vision, fraud, RAG)
│   ├── ml/                # Machine learning models
│   └── ui/                # Gradio web interface
├── data/                  # Datasets (gitignored)
│   └── raw/
│       └── roboflow_damage/  # 312 vehicle damage images
├── scripts/               # Setup and utility scripts
├── requirements.txt       # Python dependencies
└── README.md
```

---

## How It Works

### 1. Damage Analysis with RAG

When analyzing damage photos, I don't just rely on AI vision alone. I built a RAG (Retrieval-Augmented Generation) system that:
- Retrieves historical repair costs from a knowledge base
- Matches detected damage types to real repair scenarios
- Provides accurate cost estimates based on actual data
- Includes repair timelines and recommendations

### 2. Intelligent Fraud Detection

The fraud detection goes beyond pre-calculated scores. I implemented real-time checks that:
- Compare what the claimant says the damage costs versus what the AI actually finds
- Flag suspicious patterns like claiming near maximum coverage
- Check evidence quality (missing photos, suspicious image counts)
- Auto-generate detailed denial reasons with specific fraud indicators

### 3. One-Click Workflow

I designed the interface to be as efficient as possible:
- Click any claim in the table to instantly load all details
- All images display immediately in a gallery
- Action buttons are clearly labeled with what they do
- Results update in real-time as you progress through steps

---

## Key Datasets

**Insurance Claims Data:**
- Source: Kaggle Insurance Fraud Detection Dataset
- Records: 15,420 real insurance claims with fraud labels
- Features: 33 attributes including claim amount, incident details, policy info

**Vehicle Damage Images:**
- Source: Roboflow Car Damage Dataset
- Images: 312 annotated vehicle damage photos
- Categories: 6 damage types (dent, scratch, broken glass, front/rear/side damage)
- Split: Training, validation, and test sets

---

## Configuration

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=your-api-key-here
DATABASE_URL=postgresql://user:password@localhost:5432/claimguard

# Optional
OPENAI_VISION_MODEL=gpt-4o-mini  # For cost optimization
MAX_CLAIMS_PER_DAY=1000          # Rate limiting
```

---

## Development

I structured the code to be maintainable and extensible:

- **Models** (`app/models/`) - Database schemas using SQLAlchemy
- **Services** (`app/services/`) - Business logic separated from UI
- **Scripts** (`scripts/`) - One-time setup and data loading utilities

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black app/
isort app/
```

---

## Cost Optimization

I designed this to run affordably:

- **OpenAI Vision**: ~$0.0006 per image (gpt-4o-mini)
- **Custom YOLO**: Free local inference (~50ms per image)
- **Caching**: All AI responses cached to avoid repeat API calls
- **Estimated monthly cost**: $20-30 for moderate usage

To further reduce costs:
- Use Custom YOLO instead of OpenAI Vision where possible
- Implement aggressive caching strategies
- Batch process claims during off-peak hours

---

## Future Improvements

Areas I'm planning to enhance:
- Real-time video damage analysis
- Integration with Neo4j for fraud network detection
- Mobile app for field adjusters
- Multi-language support for global deployment
- Advanced analytics dashboard with fraud trend visualization


---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- **Kaggle** for the insurance fraud dataset
- **Roboflow** for the vehicle damage image dataset
- **OpenAI** for the Vision API
- **Ultralytics** for YOLOv8

---

Built with Python, FastAPI, and Gradio.
