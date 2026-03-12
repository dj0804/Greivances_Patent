import { NextRequest, NextResponse } from "next/server"

const BACKEND_API_BASE_URL = process.env.BACKEND_API_BASE_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const payload = await request.json()

    const response = await fetch(`${BACKEND_API_BASE_URL}/api/v1/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      cache: "no-store",
    })

    const contentType = response.headers.get("content-type") || ""
    const responseBody = contentType.includes("application/json")
      ? await response.json()
      : await response.text()

    if (!response.ok) {
      return NextResponse.json(
        {
          success: false,
          error: "Backend request failed",
          detail: typeof responseBody === "string" ? responseBody : responseBody?.detail,
        },
        { status: response.status }
      )
    }

    return NextResponse.json(responseBody)
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unexpected proxy error"
    return NextResponse.json(
      {
        success: false,
        error: "Proxy error",
        detail: message,
      },
      { status: 500 }
    )
  }
}
