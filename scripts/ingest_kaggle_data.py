"""
Load Kaggle vehicle fraud dataset into PostgreSQL
Converts raw CSV data into structured claims, policies, and fraud scores
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.core.database import get_db, init_db
from app.models.user import User, Role, RoleType
from app.models.claim import Claim, ClaimType, ClaimStatus, ClaimPriority
from app.models.policy import Policy
from app.models.fraud_score import FraudScore


def load_csv_data():
    """Load and clean the Kaggle fraud dataset"""
    csv_path = Path("data/raw/kaggle_fraud/fraud_oracle.csv")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        print("Run: python scripts/download_datasets.py first")
        sys.exit(1)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean column names (remove BOM if present)
    df.columns = df.columns.str.replace('\ufeff', '')

    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Convert fraud target to integer (0 or 1)
    df['FraudFound_P'] = df['FraudFound_P'].astype(int)

    fraud_count = df['FraudFound_P'].sum()
    print(f"\nFraud distribution: {fraud_count} fraudulent ({fraud_count/len(df)*100:.1f}%)")

    return df


def create_seed_users(session: Session):
    """Create system users for testing"""

    # Check if users already exist
    if session.query(User).count() > 0:
        print("Users already exist, skipping creation")
        return

    # Create roles
    admin_role = Role(name=RoleType.ADMIN, description="System administrator")
    adjuster_role = Role(name=RoleType.ADJUSTER, description="Claims adjuster")
    investigator_role = Role(name=RoleType.INVESTIGATOR, description="Fraud investigator")
    viewer_role = Role(name=RoleType.VIEWER, description="Read-only viewer")

    session.add_all([admin_role, adjuster_role, investigator_role, viewer_role])
    session.commit()

    # Create admin user
    admin = User(
        email="admin@claimguard.com",
        hashed_password="$2b$12$hashed_password_here",  # In production, hash properly
        full_name="System Administrator",
        is_active=True,
        is_verified=True
    )
    admin.roles.append(admin_role)

    # Create adjuster users
    adjusters = []
    for i in range(1, 6):
        adjuster = User(
            email=f"adjuster{i}@claimguard.com",
            hashed_password="$2b$12$hashed_password_here",
            full_name=f"Claims Adjuster {i}",
            is_active=True,
            is_verified=True
        )
        adjuster.roles.append(adjuster_role)
        adjusters.append(adjuster)

    # Create investigator
    investigator = User(
        email="investigator@claimguard.com",
        hashed_password="$2b$12$hashed_password_here",
        full_name="Fraud Investigator",
        is_active=True,
        is_verified=True
    )
    investigator.roles.append(investigator_role)

    session.add_all([admin, *adjusters, investigator])
    session.commit()

    print(f"Created {len(adjusters) + 2} users")


def generate_policy_from_row(row, policy_number: int) -> Policy:
    """Create a policy object from CSV row"""

    # All policies in this dataset are auto insurance
    policy_type = 'auto'

    # Parse vehicle price
    price_str = row.get('VehiclePrice', '20000 to 29000')
    if 'more than 69000' in str(price_str):
        coverage_limit = 100000.0
    elif 'to' in str(price_str):
        coverage_limit = float(price_str.split('to')[1].strip()) * 1.5
    else:
        coverage_limit = 50000.0

    # Parse deductible
    deductible = float(row.get('Deductible', 500))

    # Generate policy dates based on Days_Policy_Accident
    days_policy = row.get('Days_Policy_Accident', '15 to 30')
    if 'more than 30' in str(days_policy):
        days_active = random.randint(31, 365)
    elif 'to' in str(days_policy):
        parts = days_policy.split('to')
        days_active = random.randint(int(parts[0].strip()), int(parts[1].strip()))
    else:
        days_active = 30

    effective_date = datetime.now() - timedelta(days=days_active + 365)
    expiration_date = effective_date + timedelta(days=365)

    return Policy(
        policy_number=f"POL-{row.get('Year', 2024)}-{policy_number:06d}",
        policyholder_name=f"Policyholder {policy_number}",
        policy_type=policy_type,
        coverage_limit=coverage_limit,
        deductible=deductible,
        effective_date=effective_date,
        expiration_date=expiration_date,
        is_active=True,
        coverage_details={
            'base_policy': row.get('BasePolicy', 'Liability'),
            'vehicle_make': row.get('Make', 'Unknown'),
            'vehicle_category': row.get('VehicleCategory', 'Sedan'),
            'age_of_vehicle': row.get('AgeOfVehicle', '3 years'),
        }
    )


def generate_claim_from_row(row, policy_id, adjuster_id, claim_number: int) -> Claim:
    """Create a claim object from CSV row"""

    # Parse days since policy start
    days_policy = row.get('Days_Policy_Accident', '15 to 30')
    if 'more than 30' in str(days_policy):
        days_since_policy = random.randint(31, 180)
    elif 'to' in str(days_policy):
        parts = days_policy.split('to')
        days_since_policy = random.randint(int(parts[0].strip()), int(parts[1].strip()))
    else:
        days_since_policy = 30

    incident_date = datetime.now() - timedelta(days=random.randint(1, 90))

    # Estimate damage based on vehicle price and deductible
    deductible = float(row.get('Deductible', 500))
    price_str = row.get('VehiclePrice', '20000 to 29000')

    if 'more than 69000' in str(price_str):
        avg_price = 80000
    elif 'to' in str(price_str):
        parts = price_str.split('to')
        avg_price = (int(parts[0].strip()) + int(parts[1].strip())) / 2
    else:
        avg_price = 25000

    # Estimate damage as 5-30% of vehicle value
    estimated_damage = avg_price * random.uniform(0.05, 0.30)

    # Determine status based on fraud score
    fraud_found = row['FraudFound_P']
    if fraud_found == 1:
        status = random.choice([ClaimStatus.INVESTIGATING, ClaimStatus.DENIED])
        priority = ClaimPriority.HIGH
    else:
        status = random.choice([ClaimStatus.APPROVED, ClaimStatus.SETTLED])
        priority = ClaimPriority.MEDIUM if estimated_damage > 10000 else ClaimPriority.LOW

    return Claim(
        claim_number=f"CLM-{row.get('Year', 2024)}-{claim_number:06d}",
        claim_type=ClaimType.AUTO,
        status=status,
        priority=priority,
        policy_id=policy_id,
        assigned_adjuster_id=adjuster_id,
        incident_date=incident_date,
        incident_location=f"{row.get('AccidentArea', 'Urban')} area",
        description=f"Vehicle accident on {row.get('DayOfWeek', 'Monday')}. "
                   f"Driver fault: {row.get('Fault', 'Third Party')}. "
                   f"Police report: {row.get('PoliceReportFiled', 'No')}. "
                   f"Witness: {row.get('WitnessPresent', 'No')}.",
        estimated_damage=estimated_damage,
        deductible=deductible,
        claim_metadata={
            'month': row.get('Month', 'Jan'),
            'day_of_week': row.get('DayOfWeek', 'Monday'),
            'make': row.get('Make', 'Unknown'),
            'marital_status': row.get('MaritalStatus', 'Single'),
            'age': row.get('Age', 30),
            'fault': row.get('Fault', 'Third Party'),
            'driver_rating': row.get('DriverRating', 1),
            'past_claims': row.get('PastNumberOfClaims', 'none'),
            'police_report': row.get('PoliceReportFiled', 'No'),
            'witness': row.get('WitnessPresent', 'No'),
            'agent_type': row.get('AgentType', 'Internal'),
        }
    )


def generate_fraud_score_from_row(row, claim_id) -> FraudScore:
    """Create fraud score object from CSV row"""

    fraud_found = row['FraudFound_P']

    # Generate realistic fraud scores
    if fraud_found == 1:
        ml_score = random.uniform(0.6, 0.95)
        graph_score = random.uniform(0.5, 0.9)
    else:
        ml_score = random.uniform(0.05, 0.45)
        graph_score = random.uniform(0.05, 0.40)

    pattern_score = random.uniform(0.1, 0.5)

    # Composite score: 60% ML + 40% graph
    composite_score = 0.6 * ml_score + 0.4 * graph_score

    # Determine risk level
    if composite_score >= 0.8:
        risk_level = "critical"
    elif composite_score >= 0.6:
        risk_level = "high"
    elif composite_score >= 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Identify fraud flags
    fraud_flags = []

    if row.get('PastNumberOfClaims', 'none') != 'none':
        fraud_flags.append('multiple_prior_claims')

    if row.get('WitnessPresent', 'No') == 'No' and row.get('PoliceReportFiled', 'No') == 'No':
        fraud_flags.append('no_corroboration')

    if 'more than 30' in str(row.get('Days_Policy_Claim', '')):
        fraud_flags.append('quick_claim_after_policy')

    if row.get('AddressChange_Claim', 'no change') != 'no change':
        fraud_flags.append('address_change_before_claim')

    # Determine if investigation needed
    requires_investigation = composite_score >= 0.5

    return FraudScore(
        claim_id=claim_id,
        fraud_score=composite_score,
        ml_model_score=ml_score,
        graph_network_score=graph_score,
        pattern_matching_score=pattern_score,
        fraud_flags=fraud_flags,
        risk_level=risk_level,
        model_version="xgboost_v1_kaggle",
        requires_investigation=requires_investigation
    )


def ingest_data(limit: int = None):
    """Main ingestion function"""

    print("\n" + "="*60)
    print("ClaimGuard Data Ingestion - Kaggle Vehicle Fraud Dataset")
    print("="*60 + "\n")

    # Initialize database
    print("Initializing database...")
    init_db()

    # Create engine and session
    engine = create_engine(settings.DATABASE_URL)
    session = Session(engine)

    try:
        # Create seed users
        print("\nCreating seed users...")
        create_seed_users(session)

        # Get adjusters for assignment
        adjusters = session.query(User).filter(
            User.roles.any(Role.name == RoleType.ADJUSTER)
        ).all()

        if not adjusters:
            print("Error: No adjusters found")
            return

        # Load CSV data
        df = load_csv_data()

        if limit:
            df = df.head(limit)
            print(f"\nLimiting to first {limit} records for testing")

        print(f"\nProcessing {len(df)} records...")

        # Process records in batches
        batch_size = 100
        total_processed = 0

        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            for idx, row in batch_df.iterrows():
                # Create policy
                policy = generate_policy_from_row(row, total_processed + 1)
                session.add(policy)
                session.flush()

                # Create claim
                adjuster = random.choice(adjusters)
                claim = generate_claim_from_row(row, policy.id, adjuster.id, total_processed + 1)
                session.add(claim)
                session.flush()

                # Create fraud score
                fraud_score = generate_fraud_score_from_row(row, claim.id)
                session.add(fraud_score)

                total_processed += 1

            # Commit batch
            session.commit()
            print(f"Processed {total_processed}/{len(df)} records...", end='\r')

        print(f"\n\n✅ Successfully ingested {total_processed} records!")

        # Print summary
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        print(f"Policies created: {session.query(Policy).count()}")
        print(f"Claims created: {session.query(Claim).count()}")
        print(f"Fraud scores: {session.query(FraudScore).count()}")

        fraud_claims = session.query(Claim).join(FraudScore).filter(
            FraudScore.fraud_score >= 0.5
        ).count()
        print(f"High-risk claims (fraud score >= 0.5): {fraud_claims}")

        print("\n" + "="*60)
        print("Next steps:")
        print("1. Start API: uvicorn app.api.main:app --reload")
        print("2. Train fraud model: python app/ml/training/train_fraud_model.py")
        print("3. Launch Gradio UI: python app/ui/app.py")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Kaggle fraud dataset")
    parser.add_argument("--limit", type=int, help="Limit number of records (for testing)")
    args = parser.parse_args()

    ingest_data(limit=args.limit)
