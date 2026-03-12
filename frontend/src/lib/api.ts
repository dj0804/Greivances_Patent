export interface DecisionDetails {
  cluster_id?: number | string
  sla_hours?: number
}

export interface PredictionResult {
  urgency_level: string
  confidence: number
  summary?: string
  decision_details?: DecisionDetails
}

export interface PredictResponse {
  success: boolean
  data: PredictionResult
  processing_time_ms?: number
}

export interface DashboardComplaint {
  id: string
  text: string
  summary?: string
  urgency: string
  score: number
  cluster?: number | string
  time: string
  slaBreach: boolean
}

export interface DashboardData {
  stats: {
    total_active: number
    critical_urgency: number
    active_clusters: number
    sla_overdue: number
  }
  trend: Array<{ name: string; high: number; medium: number; low: number }>
  clusters: Array<{ name: string; count: number }>
  recent_complaints: DashboardComplaint[]
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  })

  if (!response.ok) {
    const errorBody = await response.text()
    throw new Error(errorBody || `Request failed with status ${response.status}`)
  }

  return response.json() as Promise<T>
}

export function predictComplaint(payload: {
  complaint_text: string
  complaint_type?: string
  hostel_id?: string
  timestamp?: string
}) {
  return request<PredictResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  })
}

export function getDashboardData() {
  return request<DashboardData>("/api/dashboard")
}
