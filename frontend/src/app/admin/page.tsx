"use client"

import { useEffect, useMemo, useState } from "react"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from "recharts"
import { AlertTriangle, Clock, Activity, ArrowLeft, RefreshCw, Filter, ShieldAlert } from "lucide-react"
import { DashboardComplaint, getDashboardData } from "@/lib/api"

export default function AdminDashboard() {
    const [isRefreshing, setIsRefreshing] = useState(false)
    const [filter, setFilter] = useState("all")
    const [isLoading, setIsLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [stats, setStats] = useState({
        total_active: 0,
        critical_urgency: 0,
        active_clusters: 0,
        sla_overdue: 0,
    })
    const [trendData, setTrendData] = useState<Array<{ name: string; high: number; medium: number; low: number }>>([])
    const [clusterData, setClusterData] = useState<Array<{ name: string; count: number }>>([])
    const [complaints, setComplaints] = useState<DashboardComplaint[]>([])

    const loadDashboard = async () => {
        setError(null)
        try {
            const data = await getDashboardData()
            setStats(data.stats)
            setTrendData(data.trend)
            setClusterData(data.clusters)
            setComplaints(data.recent_complaints)
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Failed to load dashboard data"
            setError(message)
        } finally {
            setIsLoading(false)
        }
    }

    useEffect(() => {
        loadDashboard()
    }, [])

    const handleRefresh = async () => {
        setIsRefreshing(true)
        await loadDashboard()
        setIsRefreshing(false)
    }

    const filteredComplaints = useMemo(
        () => complaints.filter(c => filter === 'all' || (filter === 'high' && c.urgency === 'High') || (filter === 'breaches' && c.slaBreach)),
        [complaints, filter]
    )

    const getUrgencyBadge = (level: string) => {
        const l = level?.toLowerCase() || ""
        if (l === "high") return <span className="inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20 shadow-sm">HIGH</span>
        if (l === "medium") return <span className="inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20 shadow-sm">MEDIUM</span>
        if (l === "low") return <span className="inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-sm">LOW</span>
        return <span>{level}</span>
    }

    return (
        <div className="container mx-auto p-4 md:p-8 max-w-7xl">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
                <div>
                    <Link href="/" className="inline-flex items-center text-sm font-medium text-slate-400 hover:text-indigo-400 mb-2 transition-colors">
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Back to Home
                    </Link>
                    <h1 className="text-3xl font-extrabold tracking-tight text-slate-50">Intelligence Operations</h1>
                    <p className="text-slate-400 mt-1">Real-time system monitoring, clustering models, and SLA metrics.</p>
                </div>

                <div className="flex gap-4">
                    <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 px-4 py-2 rounded-xl shadow-inner">
                        <Filter className="w-4 h-4 text-slate-400" />
                        <select
                            className="bg-transparent text-sm font-medium outline-none text-slate-300"
                            value={filter}
                            onChange={(e) => setFilter(e.target.value)}
                        >
                            <option value="all" className="bg-slate-900">All Grievances</option>
                            <option value="high" className="bg-slate-900">Critical Only</option>
                            <option value="breaches" className="bg-slate-900">SLA Breaches</option>
                        </select>
                    </div>
                    <button
                        onClick={handleRefresh}
                        className="inline-flex items-center justify-center p-2 rounded-xl border border-slate-700 bg-slate-800 hover:bg-slate-700 hover:text-indigo-400 text-slate-300 transition-colors shadow-[0_0_15px_rgba(0,0,0,0.2)] focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                    >
                        <RefreshCw className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </div>

            {/* Top Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
                <Card className="border-0 shadow-lg bg-slate-900/50 backdrop-blur-md overflow-hidden relative group border-t border-slate-800">
                    <div className="absolute inset-y-0 left-0 w-1 bg-slate-500" />
                    <CardContent className="pt-6 relative z-10">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Total Active</p>
                                <h3 className="text-3xl font-extrabold text-slate-100">{stats.total_active}</h3>
                            </div>
                            <Activity className="w-10 h-10 text-slate-700 group-hover:scale-110 transition-transform duration-300 group-hover:text-slate-500" />
                        </div>
                    </CardContent>
                </Card>

                <Card className="border-0 shadow-lg bg-slate-900/50 backdrop-blur-md overflow-hidden relative group border-t border-slate-800">
                    <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-rose-500 to-red-600" />
                    <CardContent className="pt-6 relative z-10">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs font-bold text-rose-400 uppercase tracking-wider mb-1">Critical Urgency</p>
                                <h3 className="text-3xl font-extrabold text-slate-100">{stats.critical_urgency}</h3>
                            </div>
                            <ShieldAlert className="w-10 h-10 text-rose-900/50 group-hover:scale-110 transition-transform duration-300 group-hover:text-rose-500" />
                        </div>
                    </CardContent>
                </Card>

                <Card className="border-0 shadow-lg bg-slate-900/50 backdrop-blur-md overflow-hidden relative group border-t border-slate-800">
                    <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-indigo-500 to-violet-600" />
                    <CardContent className="pt-6 relative z-10">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs font-bold text-indigo-400 uppercase tracking-wider mb-1">Active Clusters</p>
                                <h3 className="text-3xl font-extrabold text-slate-100">{stats.active_clusters}</h3>
                            </div>
                            <LayoutDashboard className="w-10 h-10 text-indigo-900/50 group-hover:scale-110 transition-transform duration-300 group-hover:text-indigo-500" />
                        </div>
                    </CardContent>
                </Card>

                <Card className="border-0 shadow-lg bg-slate-900/50 backdrop-blur-md overflow-hidden relative group border-t border-slate-800">
                    <div className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-amber-500 to-orange-500" />
                    <CardContent className="pt-6 relative z-10">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs font-bold text-amber-400 uppercase tracking-wider mb-1">SLA Overdue</p>
                                <h3 className="text-3xl font-extrabold text-slate-100">{stats.sla_overdue}</h3>
                            </div>
                            <Clock className="w-10 h-10 text-amber-900/50 group-hover:scale-110 transition-transform duration-300 group-hover:text-amber-500" />
                        </div>
                    </CardContent>
                </Card>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                {/* Trend Graph */}
                <Card className="h-[400px] flex flex-col border-0 shadow-xl bg-slate-900/70 backdrop-blur-md border-t border-slate-800">
                    <CardHeader>
                        <CardTitle className="text-lg font-bold text-slate-100">Urgency Trajectory</CardTitle>
                        <CardDescription className="text-slate-400">Volume categorized by priority over 5 days</CardDescription>
                    </CardHeader>
                    <CardContent className="flex-1 min-h-0">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={trendData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" />
                                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} dy={10} />
                                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} dx={-10} />
                                <Tooltip
                                    contentStyle={{ borderRadius: '12px', border: '1px solid #334155', backgroundColor: '#0f172a', color: '#f8fafc', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.5)' }}
                                    itemStyle={{ fontWeight: 600 }}
                                />
                                <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px', color: '#cbd5e1' }} />
                                <Line type="monotone" dataKey="high" name="Critical" stroke="#f43f5e" strokeWidth={3} dot={{ r: 4, strokeWidth: 2, fill: '#0f172a' }} activeDot={{ r: 6, stroke: '#f43f5e', strokeWidth: 2 }} />
                                <Line type="monotone" dataKey="medium" name="Elevated" stroke="#f59e0b" strokeWidth={3} dot={{ r: 4, strokeWidth: 2, fill: '#0f172a' }} />
                                <Line type="monotone" dataKey="low" name="Standard" stroke="#10b981" strokeWidth={3} dot={{ r: 4, strokeWidth: 2, fill: '#0f172a' }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>

                {/* Cluster Distribution */}
                <Card className="h-[400px] flex flex-col border-0 shadow-xl bg-slate-900/70 backdrop-blur-md border-t border-slate-800">
                    <CardHeader>
                        <CardTitle className="text-lg font-bold text-slate-100">Semantic Clustering Map</CardTitle>
                        <CardDescription className="text-slate-400">Systemic issue prevalence detected by HDBSCAN</CardDescription>
                    </CardHeader>
                    <CardContent className="flex-1 min-h-0 pl-0">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={clusterData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#334155" />
                                <XAxis type="number" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: '#e2e8f0', fontWeight: 600, fontSize: 12 }} width={90} />
                                <Tooltip
                                    cursor={{ fill: '#1e293b' }}
                                    contentStyle={{ borderRadius: '12px', border: '1px solid #334155', backgroundColor: '#0f172a', color: '#f8fafc', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.5)' }}
                                />
                                <Bar dataKey="count" name="Grievances" fill="url(#colorUv)" radius={[0, 6, 6, 0]} barSize={28}>
                                    <defs>
                                        <linearGradient id="colorUv" x1="0" y1="0" x2="1" y2="0">
                                            <stop offset="0%" stopColor="#4f46e5" />
                                            <stop offset="100%" stopColor="#8b5cf6" />
                                        </linearGradient>
                                    </defs>
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </div>

            {/* Complaint Table */}
            <Card className="border-0 shadow-xl bg-slate-900/70 backdrop-blur-md overflow-hidden border-t border-slate-800">
                <CardHeader className="border-b border-slate-800 bg-slate-900/50">
                    <CardTitle className="text-lg font-bold text-slate-100">Incoming Streams</CardTitle>
                    <CardDescription className="text-slate-400">Real-time data from prediction pipeline</CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                    {error && (
                        <div className="px-6 py-4 text-sm text-rose-300 bg-rose-500/10 border-b border-rose-500/20">
                            {error}
                        </div>
                    )}
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left border-collapse">
                            <thead className="text-xs text-slate-400 bg-slate-950/80 uppercase font-semibold">
                                <tr>
                                    <th className="px-6 py-4 font-semibold tracking-wider">Reference ID</th>
                                    <th className="px-6 py-4 font-semibold tracking-wider">Payload / AI Summary</th>
                                    <th className="px-6 py-4 font-semibold tracking-wider">Priority</th>
                                    <th className="px-6 py-4 font-semibold tracking-wider text-center">Engine Score</th>
                                    <th className="px-6 py-4 font-semibold tracking-wider text-center">Node</th>
                                    <th className="px-6 py-4 font-semibold tracking-wider text-right">Age</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800">
                                {filteredComplaints.map((c) => (
                                    <tr key={c.id} className={`hover:bg-slate-800/50 transition-colors ${c.urgency === 'High' ? 'bg-rose-900/10' : ''}`}>
                                        <td className="px-6 py-5 font-bold text-slate-200 whitespace-nowrap">
                                            {c.id}
                                            {c.slaBreach && (
                                                <div className="flex items-center mt-1.5 text-[10px] uppercase text-rose-400 font-extrabold tracking-wider bg-rose-500/10 border border-rose-500/20 w-fit px-1.5 py-0.5 rounded shadow-sm">
                                                    <AlertTriangle className="w-3 h-3 mr-1" /> SLA Failure
                                                </div>
                                            )}
                                        </td>
                                        <td className="px-6 py-5 max-w-md">
                                            <p className="text-slate-300 font-medium line-clamp-1">{c.text}</p>
                                            <p className="text-slate-500 text-xs mt-1.5 line-clamp-1 italic">{c.summary}</p>
                                        </td>
                                        <td className="px-6 py-5 whitespace-nowrap">
                                            {getUrgencyBadge(c.urgency)}
                                        </td>
                                        <td className="px-6 py-5 text-center">
                                            <span className="font-mono font-medium text-slate-400 bg-slate-950/50 border border-slate-800 px-2 py-1 rounded-md shadow-inner">
                                                {c.score.toFixed(2)}
                                            </span>
                                        </td>
                                        <td className="px-6 py-5 text-center text-indigo-400 font-bold">
                                            #{c.cluster}
                                        </td>
                                        <td className="px-6 py-5 text-right text-slate-500 whitespace-nowrap font-medium">
                                            {c.time}
                                        </td>
                                    </tr>
                                ))}
                                {isLoading && (
                                    <tr>
                                        <td colSpan={6} className="px-6 py-16 text-center text-slate-500">
                                            <div className="flex flex-col items-center justify-center">
                                                <RefreshCw className="w-8 h-8 text-slate-700 mb-2 animate-spin" />
                                                <p className="font-medium text-slate-500">Loading dashboard data...</p>
                                            </div>
                                        </td>
                                    </tr>
                                )}
                                {!isLoading && filteredComplaints.length === 0 && (
                                    <tr>
                                        <td colSpan={6} className="px-6 py-16 text-center text-slate-500">
                                            <div className="flex flex-col items-center justify-center">
                                                <Activity className="w-8 h-8 text-slate-700 mb-2" />
                                                <p className="font-medium text-slate-500">No active grievances passing filter</p>
                                            </div>
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}
// Placeholder for LayoutDashboard icon
const LayoutDashboard = ({ className }: { className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
        <rect width="7" height="9" x="3" y="3" rx="1" />
        <rect width="7" height="5" x="14" y="3" rx="1" />
        <rect width="7" height="9" x="14" y="12" rx="1" />
        <rect width="7" height="5" x="3" y="16" rx="1" />
    </svg>
)
