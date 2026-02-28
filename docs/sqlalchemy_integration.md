# SQLAlchemy Integration

## Overview
This document outlines the transition from using raw `psycopg` queries to utilizing **SQLAlchemy ORM** for database interactions within the project, specifically focusing on the `database_manager.py` script.

## Changes Made

### 1. Dependencies Update
We updated the project's dependencies to include SQLAlchemy and ensure compatibility with PostgreSQL.
- **Added**: `sqlalchemy>=2.0.0`
- **Added**: `psycopg2-binary` (To serve as the default driver for SQLAlchemy if needed, though SQLAlchemy 2.0+ supports `psycopg` natively)
- These were added to the `# Neon DB` section of `Requirements.txt`.

### 2. Refactoring `database_manager.py`
The `HostelDB` class originally utilized raw SQL execution (`conn.cursor().execute(...)`) to initialize tables and insert data. This was refactored to use a declarative model and a session-based approach.

#### ORM Model Definition
An explicit declarative model was introduced to represent the database table:
```python
from sqlalchemy import Column, String, DateTime, Text, Integer, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

Base = declarative_base()

class HostelIncident(Base):
    __tablename__ = 'hostel_incidents'

    complaint_id = Column(String(50), primary_key=True)
    timestamp = Column(DateTime)
    raw_text = Column(Text, nullable=False)
    preprocessing_data = Column(JSONB)
    vector_id = Column(Integer, unique=True, nullable=True) # Linked to FAISS index
    db_received_at = Column(DateTime, default=func.now())
```

#### Session and Engine Setup
The `HostelDB` initialization was updated to create an SQLAlchemy engine and a session maker:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class HostelDB:
    def __init__(self):
        url = os.getenv("DATABASE_URL")
        # Ensure url format is compatible with SQLAlchemy (postgresql:// over postgres://)
        if url and url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        
        self.engine = create_engine(url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
```

#### Table Initialization
Instead of executing a `CREATE TABLE IF NOT EXISTS` string, it now utilizes SQLAlchemy's mapping:
```python
def initialize_tables(self):
    """Creates the hostel incident table if it doesn't exist."""
    Base.metadata.create_all(bind=self.engine)
    
    # Safe fallback migration for existing tables to add vector_id
    try:
        with self.engine.connect() as conn:
            conn.execute(text("ALTER TABLE hostel_incidents ADD COLUMN vector_id INTEGER UNIQUE;"))
            conn.commit()
    except Exception:
        pass
```

#### Incident Insertion (with Upsert logic)
The `ON CONFLICT (complaint_id) DO NOTHING` logic from raw SQL was replicated using the PostgreSQL specific dialect tools provided by SQLAlchemy:
```python
from sqlalchemy.dialects.postgresql import insert

def insert_incident(self, complaint_id, timestamp, raw_text, preprocessing_data, vector_id=None):
    with self.get_session() as session:
        stmt = insert(HostelIncident).values(
            complaint_id=complaint_id,
            timestamp=timestamp,
            raw_text=raw_text,
            preprocessing_data=preprocessing_data,
            vector_id=vector_id
        )
        # Use PostgreSQL ON CONFLICT DO NOTHING
        stmt = stmt.on_conflict_do_nothing(index_elements=['complaint_id'])
        
        session.execute(stmt)
        session.commit()
```

## Validation
To verify the smooth running of the application, running `python scripts/database_manager.py` directly executes an `initialize_tables()` successfully, ensuring that connection strings are properly processed and standard tables are maintained correctly out of the mapped base.
