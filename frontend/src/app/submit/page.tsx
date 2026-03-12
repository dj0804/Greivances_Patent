"use client"

import { useState } from "react"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { AlertCircle, FileText, CheckCircle2, Clock, Loader2, ArrowLeft, Send, Activity } from "lucide-react"
import { predictComplaint, PredictionResult } from "@/lib/api"

export default function SubmitComplaint() {
    const [complaintText, setComplaintText] = useState("")
    const [department, setDepartment] = useState("")
    const [location, setLocation] = useState("")
    const [isLoading, setIsLoading] = useState(false)
    const [result, setResult] = useState<PredictionResult | null>(null)
    const [error, setError] = useState<string | null>(null)

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()

        if (!complaintText) {
            setError("Complaint text is required")
            return
        }

        setIsLoading(true)
        setError(null)

        try {
            const response = await predictComplaint({
                complaint_text: complaintText,
                complaint_type: department || "general",
                hostel_id: location || "unknown",
                timestamp: new Date().toISOString()
            })

            if (response?.success && response?.data) {
                setResult(response.data)
            } else {
                setError("Invalid response format from server.")
            }
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Failed to submit complaint. Is the API backend running?"
            setError(message)
        } finally {
            setIsLoading(false)
        }
    }

    const getUrgencyStyles = (level: string) => {
        const l = level?.toLowerCase() || ""
        if (l === "high") return "bg-slate-900 border-rose-500/30 ring-1 ring-rose-500/20"
        if (l === "medium") return "bg-slate-900 border-amber-500/30 ring-1 ring-amber-500/20"
        if (l === "low") return "bg-slate-900 border-emerald-500/30 ring-1 ring-emerald-500/20"
        return "bg-slate-900 border-slate-800 ring-1 ring-slate-800"
    }

    const getGradientText = (level: string) => {
        const l = level?.toLowerCase() || ""
        if (l === "high") return "text-transparent bg-clip-text bg-gradient-to-r from-red-400 to-rose-400"
        if (l === "medium") return "text-transparent bg-clip-text bg-gradient-to-r from-amber-300 to-orange-400"
        if (l === "low") return "text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-400"
        return "text-slate-200"
    }

    return (
        <div className="container mx-auto max-w-5xl py-12 px-4">
            <div className="mb-10">
                <Link href="/" className="inline-flex items-center text-sm font-medium text-slate-400 hover:text-indigo-400 mb-6 transition-colors">
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back to Overview
                </Link>
                <h1 className="text-4xl font-extrabold tracking-tight text-slate-50">Log a Grievance</h1>
                <p className="text-lg text-slate-400 mt-2 max-w-2xl">
                    Describe your issue in detail. Our neural network will analyze the semantics and temporal context to route it instantly.
                </p>
            </div>

            <div className="grid md:grid-cols-[1.2fr_1fr] gap-8 items-start">
                {/* Form Card */}
                <Card className="border-slate-800 shadow-xl bg-slate-900/50 backdrop-blur-md">
                    <CardHeader className="pb-6 border-b border-slate-800">
                        <CardTitle className="text-xl text-slate-100 flex items-center">
                            <FileText className="w-5 h-5 mr-2 text-indigo-400" />
                            Complaint Detail
                        </CardTitle>
                        <CardDescription className="text-slate-400">Provide accurate information to help the AI cluster your issue.</CardDescription>
                    </CardHeader>
                    <CardContent className="pt-6">
                        <form onSubmit={handleSubmit} className="space-y-6">
                            <div className="space-y-2">
                                <label className="text-sm font-semibold text-slate-300">
                                    Description <span className="text-rose-500">*</span>
                                </label>
                                <textarea
                                    className="flex min-h-[160px] w-full rounded-xl border border-slate-700 bg-slate-950/50 px-4 py-3 text-sm text-slate-100 shadow-inner placeholder:text-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:border-transparent transition-all disabled:opacity-50 resize-y"
                                    placeholder="E.g., The AC in my room is making loud noises and leaking water. It has been like this since yesterday."
                                    value={complaintText}
                                    onChange={(e) => setComplaintText(e.target.value)}
                                    disabled={isLoading}
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-semibold text-slate-300">
                                        Department <span className="text-slate-500 font-normal ml-1">(Optional)</span>
                                    </label>
                                    <div className="relative">
                                        <select
                                            className="flex h-11 w-full appearance-none rounded-xl border border-slate-700 bg-slate-950/50 px-4 text-sm text-slate-100 shadow-inner transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:border-transparent"
                                            value={department}
                                            onChange={(e) => setDepartment(e.target.value)}
                                            disabled={isLoading}
                                        >
                                            <option value="">Auto-detect</option>
                                            <option value="maintenance">Maintenance</option>
                                            <option value="it">IT & Network</option>
                                            <option value="housekeeping">Housekeeping</option>
                                            <option value="food">Mess / Dining</option>
                                            <option value="security">Security</option>
                                        </select>
                                        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-slate-500">
                                            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                                        </div>
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-semibold text-slate-300">
                                        Location <span className="text-slate-500 font-normal ml-1">(Optional)</span>
                                    </label>
                                    <input
                                        type="text"
                                        className="flex h-11 w-full rounded-xl border border-slate-700 bg-slate-950/50 px-4 text-sm text-slate-100 shadow-inner transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:border-transparent placeholder:text-slate-600"
                                        placeholder="E.g., Block B, Room 204"
                                        value={location}
                                        onChange={(e) => setLocation(e.target.value)}
                                        disabled={isLoading}
                                    />
                                </div>
                            </div>

                            {error && (
                                <div className="p-4 text-sm text-rose-200 bg-rose-500/20 border border-rose-500/30 rounded-xl shadow-sm flex items-start">
                                    <AlertCircle className="w-5 h-5 mr-3 shrink-0 text-rose-400" />
                                    <p className="font-medium">{error}</p>
                                </div>
                            )}

                            <button
                                type="submit"
                                disabled={isLoading}
                                className="group relative w-full inline-flex items-center justify-center rounded-xl overflow-hidden bg-indigo-600 text-white shadow-[0_0_20px_rgba(79,70,229,0.3)] transition-all hover:bg-indigo-500 hover:shadow-[0_0_25px_rgba(79,70,229,0.5)] focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-950 disabled:pointer-events-none disabled:opacity-70 h-12 px-8 font-semibold text-base"
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Processing AI Pipeline...
                                    </>
                                ) : (
                                    <>
                                        <span className="mr-2">Evaluate Grievance</span>
                                        <Send className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                                    </>
                                )}
                            </button>
                        </form>
                    </CardContent>
                </Card>

                {/* Results Panel */}
                <div className="relative h-full">
                    {result ? (
                        <div className="sticky top-24 transform transition-all duration-500 ease-out opacity-100 translate-y-0">
                            <Card className={`border overflow-hidden shadow-2xl ${getUrgencyStyles(result.urgency_level)}`}>
                                <div className="absolute top-0 right-0 p-4 opacity-5">
                                    <Activity className="w-32 h-32 text-slate-100" />
                                </div>
                                <CardHeader className="pb-4 relative z-10 border-b border-white/5">
                                    <div className="flex items-center space-x-3">
                                        <div className="bg-slate-950/50 p-2 rounded-xl border border-white/10 shadow-inner">
                                            <CheckCircle2 className={`w-6 h-6 ${result.urgency_level.toLowerCase() === 'high' ? 'text-rose-400' : result.urgency_level.toLowerCase() === 'medium' ? 'text-amber-400' : 'text-emerald-400'}`} />
                                        </div>
                                        <div>
                                            <CardTitle className="text-xl font-bold text-slate-100">Analysis Complete</CardTitle>
                                            <CardDescription className="text-slate-400 font-medium">Multi-stage pipeline processed</CardDescription>
                                        </div>
                                    </div>
                                </CardHeader>

                                <CardContent className="space-y-5 relative z-10 pt-5">
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="bg-slate-950/40 p-5 rounded-2xl border border-slate-700/50 shadow-inner flex flex-col justify-center">
                                            <span className="text-xs uppercase font-bold tracking-wider text-slate-500 mb-1">Classified Urgency</span>
                                            <span className={`text-3xl font-extrabold capitalize ${getGradientText(result.urgency_level)} drop-shadow-sm`}>
                                                {result.urgency_level}
                                            </span>
                                        </div>

                                        <div className="bg-slate-950/40 p-5 rounded-2xl border border-slate-700/50 shadow-inner flex flex-col justify-center">
                                            <span className="text-xs uppercase font-bold tracking-wider text-slate-500 mb-1">Confidence Score</span>
                                            <div className="flex items-baseline">
                                                <span className="text-3xl font-extrabold text-slate-100">
                                                    {(result.confidence * 100).toFixed(1)}
                                                </span>
                                                <span className="text-lg font-bold text-slate-500 ml-1">%</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="bg-slate-950/40 p-5 rounded-2xl border border-slate-700/50 shadow-inner">
                                        <div className="flex items-center mb-2">
                                            <AlertCircle className="w-4 h-4 mr-2 text-indigo-400" />
                                            <span className="text-xs uppercase font-bold tracking-wider text-slate-400">AI Distilled Summary</span>
                                        </div>
                                        <p className="font-medium text-slate-300 leading-relaxed">
                                            {result.summary || "Complaint text was very concise, no further condensation needed."}
                                        </p>
                                    </div>

                                    <div className="flex gap-4">
                                        <div className="flex-1 bg-slate-950/40 px-4 py-3 rounded-xl border border-slate-700/50 shadow-inner flex items-center justify-between">
                                            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Cluster Identity</span>
                                            <span className="text-sm font-black font-mono bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 px-2 py-1 rounded">
                                                #{result.decision_details?.cluster_id ?? "N/A"}
                                            </span>
                                        </div>
                                        <div className="flex-1 bg-slate-950/40 px-4 py-3 rounded-xl border border-slate-700/50 shadow-inner flex items-center justify-between">
                                            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Target SLA</span>
                                            <span className="text-sm font-black bg-slate-800 text-slate-300 px-2 py-1 rounded flex items-center">
                                                <Clock className="w-3 h-3 mr-1" />
                                                {result.decision_details?.sla_hours || 24}H
                                            </span>
                                        </div>
                                    </div>
                                </CardContent>

                                <CardFooter className="pt-2 pb-6 relative z-10 border-t border-white/5 mt-2">
                                    <button
                                        onClick={() => { setResult(null); setComplaintText(""); }}
                                        className="mt-4 w-full text-sm font-bold text-slate-400 hover:text-slate-200 transition-colors flex items-center justify-center underline underline-offset-4 decoration-slate-600 hover:decoration-slate-400"
                                    >
                                        Submit another grievance
                                    </button>
                                </CardFooter>
                            </Card>
                        </div>
                    ) : (
                        <div className="sticky top-24 h-[500px] flex flex-col items-center justify-center p-12 text-center rounded-2xl border-2 border-dashed border-slate-800 bg-slate-900/30">
                            <div className="w-20 h-20 bg-slate-950 rounded-3xl border border-slate-800 flex items-center justify-center mb-6 shadow-inner">
                                <Activity className="w-10 h-10 text-slate-600" />
                            </div>
                            <h3 className="text-xl font-bold text-slate-300 mb-2">Awaiting Payload</h3>
                            <p className="text-sm text-slate-500 max-w-sm leading-relaxed">
                                Submit a grievance to see our multi-signal intelligence engine analyze semantics, urgency, and cluster density in real-time.
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
