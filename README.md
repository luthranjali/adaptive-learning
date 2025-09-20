# Adaptive Learning Assessment System

An AI-driven adaptive learning assessment system that uses machine learning to provide personalized educational experiences and course level recommendations.

## Features

- **Adaptive Question Selection**: Uses Thompson Sampling to dynamically select questions based on student performance
- **ML-Powered Level Recommendations**: XGBoost-based model for course level recommendations
- **Knowledge Gap Detection**: Clustering and anomaly detection to identify learning gaps
- **Performance Analysis**: Ensemble models for comprehensive performance analysis
- **Real-time Adaptation**: Dynamic difficulty adjustment based on student responses
- **Bayesian Knowledge Tracing**: Tracks knowledge state progression over time
- **Item Response Theory**: Psychometric analysis for question difficulty calibration

## Technology Stack

- **Backend**: FastAPI (Python 3.9+)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Cache**: Redis
- **ML Libraries**: scikit-learn, XGBoost, pandas, numpy
- **Authentication**: JWT with bcrypt password hashing
- **Background Tasks**: Celery
- **Containerization**: Docker & Docker Compose

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd adaptive-learning
   ```

2. **Start the services**
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Manual Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Set up the database**
   ```bash
   # Start PostgreSQL and Redis
   # Update database URL in .env
   
   # Run migrations
   alembic upgrade head
   ```

4. **Start the application**
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login user
- `GET /api/v1/auth/me` - Get current user info
- `PUT /api/v1/auth/me` - Update user profile

### Assessments
- `POST /api/v1/assessments/` - Create assessment (instructor/admin)
- `GET /api/v1/assessments/` - Get available assessments
- `POST /api/v1/assessments/start` - Start assessment session
- `GET /api/v1/assessments/sessions/{id}/next-question` - Get next question
- `POST /api/v1/assessments/sessions/{id}/answer` - Submit answer
- `POST /api/v1/assessments/sessions/{id}/complete` - Complete assessment
- `GET /api/v1/assessments/sessions/{id}/results` - Get assessment results

## Machine Learning Models

### 1. Item Response Theory (IRT) Model
- **Purpose**: Question difficulty calibration
- **Algorithm**: 3-Parameter Logistic Model
- **Features**: Student ability, question difficulty, discrimination, guessing

### 2. Bayesian Knowledge Tracing (BKT)
- **Purpose**: Knowledge state estimation
- **Algorithm**: Hidden Markov Model for learning
- **Features**: Skill mastery tracking, learning progression

### 3. Level Recommender
- **Purpose**: Course level recommendations
- **Algorithm**: XGBoost ensemble
- **Features**: Performance metrics, learning patterns, knowledge state

### 4. Question Selector
- **Purpose**: Adaptive question selection
- **Algorithm**: Thompson Sampling (Multi-Armed Bandit)
- **Features**: Student ability, question difficulty, exploration vs exploitation

### 5. Gap Detector
- **Purpose**: Knowledge gap identification
- **Algorithm**: K-Means clustering + Isolation Forest
- **Features**: Performance patterns, skill mastery, error analysis

### 6. Performance Analyzer
- **Purpose**: Comprehensive performance analysis
- **Algorithm**: Ensemble (Random Forest + Gradient Boosting + Linear Regression)
- **Features**: Multi-dimensional performance metrics

## Configuration

Key configuration options in `app/config.py`:

```python
# Database
DATABASE_URL = "postgresql://user:password@localhost/adaptive_learning"

# ML Models
MODEL_STORAGE_PATH = "data/models"
TRAINING_DATA_PATH = "data/raw"

# Assessment Settings
MAX_QUESTIONS_PER_ASSESSMENT = 50
ASSESSMENT_TIME_LIMIT_MINUTES = 60

# IRT Parameters
IRT_DIFFICULTY_RANGE = (-3.0, 3.0)
IRT_DISCRIMINATION_RANGE = (0.1, 2.0)
```

## Data Models

### Core Entities
- **Users**: Students, instructors, admins
- **Assessments**: Test templates and configurations
- **Questions**: Question bank with IRT parameters
- **Sessions**: Individual assessment attempts
- **Responses**: Individual question responses
- **Recommendations**: ML-generated suggestions

### ML Data
- **Performance Patterns**: Extracted learning features
- **Model Versions**: ML model tracking
- **Prediction Logs**: Model prediction history

## Development

### Project Structure
```
app/
├── api/                 # API endpoints
├── models/              # Database models
├── schemas/             # Pydantic schemas
├── services/            # Business logic
├── ml/                  # Machine learning
│   ├── models/          # ML model implementations
│   ├── features/        # Feature engineering
│   └── training/        # Model training
├── config.py           # Configuration
├── database.py         # Database setup
└── main.py            # FastAPI app
```

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black app/

# Lint code
flake8 app/

# Type checking
mypy app/
```

## Deployment

### Production Considerations
- Use environment variables for sensitive configuration
- Set up proper database backups
- Configure Redis persistence
- Use a production WSGI server (Gunicorn)
- Set up monitoring and logging
- Implement rate limiting
- Use HTTPS

### Environment Variables
```bash
DATABASE_URL=postgresql://user:password@host:port/dbname
REDIS_URL=redis://host:port
SECRET_KEY=your-secret-key
DEBUG=false
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please open an issue in the repository or contact the development team.
