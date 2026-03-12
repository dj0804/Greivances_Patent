# Dataset Columns Dictionary

This table defines the canonical fields for dataset creation and exchange.

| Column / Path | Type | Required | Description |
|---|---|---:|---|
| `schema_version` | string | ✅ | Dictionary schema version for compatibility. |
| `model_version` | string/null | ❌ | Version of model that produced predictions. |
| `complaint_id` | string | ✅ | Unique complaint identifier. |
| `timestamp` | datetime (ISO-8601) | ✅ | Complaint filing timestamp. |
| `raw_text` | string | ✅ | Original complaint text. |
| `preprocessing.cleaned_text` | string | ✅ | Cleaned text after normalization and urgency tag preservation. |
| `preprocessing.tokens` | array[string] | ✅ | Token list after stop-word and punctuation filtering. |
| `preprocessing.temporal_features.hour` | int (0-23) | ✅ | Hour of complaint filing. |
| `preprocessing.temporal_features.day_of_week` | int (0-6) | ✅ | Day index (`0=Monday`). |
| `preprocessing.temporal_features.is_peak_hour` | bool | ✅ | Whether complaint arrived in peak window. |
| `preprocessing.temporal_features.is_weekend` | bool | ✅ | Whether complaint was filed on weekend. |
| `preprocessing.temporal_features.is_night` | bool | ✅ | Whether complaint is in night window. |
| `preprocessing.temporal_features.month` | int (1-12) | ✅ | Month number. |
| `preprocessing.temporal_features.day_of_month` | int (1-31) | ✅ | Day number within month. |
| `preprocessing.temporal_features.complaint_frequency_last_24h` | int | ✅ | Complaint count in preceding 24h. |
| `preprocessing.temporal_features.complaint_frequency_last_week` | int | ✅ | Complaint count in preceding 7 days. |
| `preprocessing.context.hostel_id` | string/null | ❌ | Hostel/building identifier. |
| `preprocessing.context.complaint_type` | string/null | ❌ | Issue type (maintenance/food/safety/etc.). |
| `model_prediction.urgency_score` | float (0-1) | ✅ | Scalar urgency model score. |
| `model_prediction.predicted_class` | enum | ✅ | One of `Low/Medium/High/Critical`. |
| `model_prediction.class_probabilities.Low` | float (0-1) | ✅ | Low urgency probability. |
| `model_prediction.class_probabilities.Medium` | float (0-1) | ✅ | Medium urgency probability. |
| `model_prediction.class_probabilities.High` | float (0-1) | ✅ | High urgency probability. |
| `semantic_analysis.embedding_id` | string | ✅ | Embedding record identifier. |
| `semantic_analysis.similar_complaints` | array[object] | ✅ | Top similar complaints and cosine similarity values. |
| `semantic_analysis.semantic_similarity_score` | float (0-1) | ✅ | Similarity to known critical patterns. |
| `semantic_analysis.cluster_id` | int | ✅ | Cluster label (`-1` outlier). |
| `semantic_analysis.cluster_density` | float (>=0) | ✅ | Density score from clustering module. |
| `semantic_analysis.is_outlier` | bool | ✅ | Outlier indicator from clustering. |
| `historical_analysis.avg_resolution_time_similar` | float | ✅ | Mean resolution time for similar complaints. |
| `historical_analysis.sla_target_hours` | float | ✅ | SLA benchmark hours for this complaint type. |
| `historical_analysis.historical_urgency_calibration` | float (0-1) | ✅ | Calibration factor from historical performance. |
| `decision_inputs.U_model` | float (0-1) | ✅ | Urgency model signal. |
| `decision_inputs.D_cluster` | float (>=0) | ✅ | Cluster density signal. |
| `decision_inputs.T_temporal` | float (0-1) | ✅ | Temporal urgency signal. |
| `decision_inputs.H_history` | float (0-1) | ✅ | Historical calibration signal. |
| `decision_inputs.S_semantic` | float (0-1) | ✅ | Semantic similarity signal. |
| `decision_inputs.weights.alpha` | float (0-1) | ✅ | Weight for `U_model`. |
| `decision_inputs.weights.beta` | float (0-1) | ✅ | Weight for `D_cluster`. |
| `decision_inputs.weights.gamma` | float (0-1) | ✅ | Weight for `T_temporal`. |
| `decision_inputs.weights.delta` | float (0-1) | ✅ | Weight for `H_history`. |
| `decision_inputs.weights.epsilon` | float (0-1) | ✅ | Weight for `S_semantic`. |
| `final_decision.priority_score` | float | ✅ | Fused final priority score. |
| `final_decision.urgency_level` | enum | ✅ | One of `Low/Medium/High/Critical`. |
| `final_decision.routing_recommendation` | string | ✅ | Queue/team recommendation. |
| `final_decision.sla_hours` | float | ✅ | Response/resolve SLA in hours. |
| `final_decision.requires_immediate_action` | bool | ✅ | Immediate escalation flag. |
| `final_decision.explanation` | string | ✅ | Human-readable decision rationale. |
| `summary.distilbert_summary` | string/null | ❌ | Optional generated concise summary. |

## Notes

- Keep this dictionary as the **single source of truth** for dataset contracts.
- Additive changes should bump `schema_version` (e.g., `1.0.0` → `1.1.0`).
- For unknown signals, prefer explicit null/defaults over missing keys.
