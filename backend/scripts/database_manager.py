import os
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert, JSONB
from sqlalchemy.sql import func

load_dotenv()

Base = declarative_base()

class HostelIncident(Base):
    __tablename__ = 'hostel_incidents'

    complaint_id = Column(String(50), primary_key=True)
    timestamp = Column(DateTime)
    raw_text = Column(Text, nullable=False)
    preprocessing_data = Column(JSONB)
    db_received_at = Column(DateTime, default=func.now())

class HostelDB:
    def __init__(self):
        url = os.getenv("DATABASE_URL")
        # Ensure compatibility with SQLAlchemy which prefers postgresql:// over postgres://
        if url and url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        
        self.engine = create_engine(url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self):
        return self.SessionLocal()

    def initialize_tables(self):
        """Creates the hostel incident table if it doesn't exist."""
        Base.metadata.create_all(bind=self.engine)
        print("Database tables initialized.")

    def insert_incident(self, complaint_id, timestamp, raw_text, preprocessing_data):
        """Inserts a new incident into the database."""
        with self.get_session() as session:
            stmt = insert(HostelIncident).values(
                complaint_id=complaint_id,
                timestamp=timestamp,
                raw_text=raw_text,
                preprocessing_data=preprocessing_data
            )
            # Use PostgreSQL ON CONFLICT DO NOTHING
            stmt = stmt.on_conflict_do_nothing(index_elements=['complaint_id'])
            
            session.execute(stmt)
            session.commit()
            print(f"Inserted incident with ID: {complaint_id}")

# Quick Test
if __name__ == "__main__":
    db = HostelDB()
    db.initialize_tables()
    # test_id = db.insert_incident("101-A", "Broken fan in the common room", "Medium")
    # print(f"Inserted test incident with ID: {test_id}")