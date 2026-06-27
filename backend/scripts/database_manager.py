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

    def insert_incidents_bulk(self, incidents):
        """Bulk inserts incidents into the database with a single commit."""
        if not incidents:
            return

        with self.get_session() as session:
            stmt = insert(HostelIncident).values(incidents)
            stmt = stmt.on_conflict_do_nothing(index_elements=['complaint_id'])

            session.execute(stmt)
            session.commit()
            print(f"Inserted {len(incidents)} incidents in bulk")

    def get_sla_breach_rate(self, cluster_id: str, lookback_days: int = 30) -> float:
        """
        Return the fraction of incidents in cluster_id that breached SLA
        within the last lookback_days days.

        Returns 0.5 (neutral cold-start prior) when no historical data exists.
        """
        from datetime import datetime, timedelta, timezone
        from sqlalchemy import text
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            with self.get_session() as session:
                rows = session.execute(
                    text(
                        "SELECT preprocessing_data FROM hostel_incidents "
                        "WHERE timestamp >= :cutoff"
                    ),
                    {"cutoff": cutoff},
                ).fetchall()

            if not rows:
                return 0.5

            matching = [
                r for r in rows
                if (r[0] or {}).get("cluster_id") == cluster_id
            ]

            if not matching:
                return 0.5

            breach_count = sum(
                1 for r in matching if bool((r[0] or {}).get("sla_breach", False))
            )
            return breach_count / len(matching)

        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).error(
                "get_sla_breach_rate failed for cluster %s: %s", cluster_id, exc
            )
            return 0.5

    def get_recent_incidents(self, limit: int = 200) -> list:
        """
        Return the most recent incidents as normalized dicts.

        Dict shape matches recent_predictions records:
        id, text, summary, urgency, score, cluster, created_at, sla_breach
        """
        from sqlalchemy import text
        try:
            with self.get_session() as session:
                rows = session.execute(
                    text(
                        "SELECT complaint_id, timestamp, raw_text, preprocessing_data "
                        "FROM hostel_incidents ORDER BY timestamp DESC LIMIT :lim"
                    ),
                    {"lim": limit},
                ).fetchall()

            result = []
            for complaint_id, timestamp, raw_text, pdata in rows:
                pdata = pdata or {}
                result.append({
                    "id": complaint_id,
                    "text": raw_text,
                    "summary": pdata.get("summary"),
                    "urgency": pdata.get("urgency_level", "Unknown"),
                    "score": float(pdata.get("confidence", 0.0)),
                    "cluster": pdata.get("cluster_id"),
                    "created_at": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
                    "sla_breach": bool(pdata.get("sla_breach", False)),
                })
            return result

        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).error("get_recent_incidents failed: %s", exc)
            return []

# Quick Test
if __name__ == "__main__":
    db = HostelDB()
    db.initialize_tables()
    # test_id = db.insert_incident("101-A", "Broken fan in the common room", "Medium")
    # print(f"Inserted test incident with ID: {test_id}")