"""
Configuration management using Pydantic Settings
Loads configuration from environment variables (.env file)
"""

from typing import Literal, List
from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    Budget-optimized for student projects (~$20-30/month)
    """

    # ========================================================================
    # APPLICATION
    # ========================================================================
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    API_PORT: int = 8000
    GRADIO_PORT: int = 7860

    # ========================================================================
    # DATABASE (Local PostgreSQL)
    # ========================================================================
    DATABASE_URL: str = "postgresql://claimguard_user:claimguard_pass@localhost:5432/claimguard"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10

    # ========================================================================
    # REDIS CACHE
    # ========================================================================
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 86400  # 24 hours
    ENABLE_LLM_CACHE: bool = True
    CACHE_HIT_RATE_TARGET: float = 0.90

    # ========================================================================
    # NEO4J GRAPH DATABASE
    # ========================================================================
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "claimguard123"

    # ========================================================================
    # QDRANT VECTOR DATABASE
    # ========================================================================
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "policies"
    EMBEDDING_DIMENSION: int = 1536  # OpenAI text-embedding-3-small

    # ========================================================================
    # MINIO (S3-Compatible Storage)
    # ========================================================================
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "claimguard"
    MINIO_SECRET_KEY: str = "claimguard123"
    MINIO_BUCKET: str = "claimguard-uploads"
    MINIO_USE_SSL: bool = False

    # ========================================================================
    # AI SERVICES (Budget-Optimized)
    # ========================================================================
    # OpenAI - Using cheaper models to stay under budget
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"  # 30x cheaper than GPT-4
    OPENAI_VISION_MODEL: str = "gpt-4o-mini"  # Cheapest vision model
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_MAX_TOKENS: int = 1000
    OPENAI_TEMPERATURE: float = 0.7

    # Optional: Anthropic Claude
    ANTHROPIC_API_KEY: str | None = None

    # Optional: Local Ollama (FREE alternative)
    OLLAMA_BASE_URL: str | None = None
    OLLAMA_MODEL: str = "llama2"

    # ========================================================================
    # SECURITY
    # ========================================================================
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    BCRYPT_ROUNDS: int = 12

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:7860",
        "http://localhost:8000"
    ]

    # ========================================================================
    # RATE LIMITING (Cost Control)
    # ========================================================================
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_CLAIMS_PER_DAY: int = 1000  # Limit for budget control

    # ========================================================================
    # ML MODELS
    # ========================================================================
    FRAUD_MODEL_PATH: str = "app/ml/models/fraud_detector_v1.pkl"
    FRAUD_MODEL_THRESHOLD: float = 0.5
    DAMAGE_MODEL_PATH: str = "app/ml/models/damage_classifier_v1.pkl"

    # ========================================================================
    # DATASET PATHS
    # ========================================================================
    KAGGLE_FRAUD_DATA_PATH: str = "data/raw/kaggle_fraud/"
    VEHIDE_IMAGES_PATH: str = "data/raw/vehide_damage/"
    ROBOFLOW_IMAGES_PATH: str = "data/raw/roboflow_damage/"
    POLICY_DOCUMENTS_PATH: str = "data/policies/"

    # ========================================================================
    # WORKFLOW SETTINGS
    # ========================================================================
    INSTANT_APPROVAL_THRESHOLD: float = 5000.0
    DETAILED_REVIEW_THRESHOLD: float = 25000.0
    FRAUD_SCORE_THRESHOLD: float = 0.5

    # ========================================================================
    # MONITORING
    # ========================================================================
    ENABLE_METRICS: bool = True
    PROMETHEUS_PORT: int = 9090

    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Budget tracking helper
class BudgetTracker:
    """
    Track API costs to stay under $50/month budget
    """

    # Pricing (as of 2024)
    GPT_35_TURBO_INPUT_COST = 0.0010  # per 1K tokens
    GPT_35_TURBO_OUTPUT_COST = 0.0020  # per 1K tokens
    GPT_4O_MINI_INPUT_COST = 0.000150  # per 1K tokens
    GPT_4O_MINI_OUTPUT_COST = 0.000600  # per 1K tokens
    EMBEDDING_SMALL_COST = 0.00002  # per 1K tokens

    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-3.5-turbo") -> float:
        """Estimate cost for an API call"""
        if model == "gpt-3.5-turbo":
            cost = (
                (input_tokens / 1000) * BudgetTracker.GPT_35_TURBO_INPUT_COST +
                (output_tokens / 1000) * BudgetTracker.GPT_35_TURBO_OUTPUT_COST
            )
        elif model == "gpt-4o-mini":
            cost = (
                (input_tokens / 1000) * BudgetTracker.GPT_4O_MINI_INPUT_COST +
                (output_tokens / 1000) * BudgetTracker.GPT_4O_MINI_OUTPUT_COST
            )
        else:
            cost = 0.0
        return cost

    @staticmethod
    def log_usage(tokens: int, cost: float, operation: str):
        """Log API usage for budget tracking"""
        print(f"[BUDGET] {operation}: {tokens} tokens, ${cost:.4f}")


# Export settings
__all__ = ["settings", "Settings", "BudgetTracker"]
