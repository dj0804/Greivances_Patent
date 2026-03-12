import os
import sys
import unittest
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_engine.engine.orchestrator import DecisionEngineOrchestrator
from decision_engine.services.data_interface import Complaint


def _build_complaint(
    complaint_id: str,
    cluster_id: str,
    urgency: float,
    hours_ago: int,
    reference_time: datetime,
) -> Complaint:
    return Complaint(
        complaint_id=complaint_id,
        cluster_id=cluster_id,
        urgency_score=urgency,
        timeline_score=0.5,
        timestamp=reference_time - timedelta(hours=hours_ago),
        text=f"Complaint {complaint_id} in {cluster_id}",
    )


class TestBackendPipeline(unittest.TestCase):
    def test_backend_pipeline_end_to_end(self):
        reference_time = datetime.now()
        orchestrator = DecisionEngineOrchestrator()

        complaints = [
            _build_complaint("A1", "CLUSTER_A", 0.9, 1, reference_time),
            _build_complaint("A2", "CLUSTER_A", 0.8, 2, reference_time),
            _build_complaint("B1", "CLUSTER_B", 0.4, 3, reference_time),
            _build_complaint("B2", "CLUSTER_B", 0.3, 4, reference_time),
            _build_complaint("C1", "CLUSTER_C", 0.7, 1, reference_time),
        ]

        results = orchestrator.process_complaints(complaints, reference_time=reference_time)

        self.assertEqual(len(results), 3)
        self.assertTrue(all(0.0 <= item.final_urgency <= 1.0 for item in results))

        urgencies = [cluster.final_urgency for cluster in results]
        self.assertEqual(urgencies, sorted(urgencies, reverse=True))

        cluster_ids = {cluster.cluster_id for cluster in results}
        self.assertSetEqual(cluster_ids, {"CLUSTER_A", "CLUSTER_B", "CLUSTER_C"})

        for cluster in results:
            self.assertGreaterEqual(cluster.complaint_count, 1)
            self.assertTrue(cluster.top_complaints)
            self.assertTrue(cluster.summary_text)
            self.assertIsNotNone(cluster.breakdown)

        orchestrator.clear_calibration_memory()


if __name__ == "__main__":
    unittest.main()
