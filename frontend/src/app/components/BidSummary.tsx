import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { api } from "../api/client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Separator } from "./ui/separator";
import {
    Building,
    MapPin,
    Calendar,
    DollarSign,
    FileText,
    AlertTriangle,
    CheckCircle2,
    Clock,
    TrendingUp,
    ArrowLeft,
    Sparkles,
} from "lucide-react";
import { Progress } from "./ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { toast } from "sonner";

type Bid = {
    id: number;
    title: string;
    agency: string;
    region: string;
    budget: string;
    deadline: string;
    announcementDate: string;
    type: string;
    status: string;
    description: string;
    documentUrl?: string;
    documentFileName?: string;
    requirements: {
        license: string[];
        location: string;
        experience: string;
        technicalStaff: string;
    };
    risks: { level: "high" | "medium" | "low"; text: string }[];
    checklist: { item: string; checked: boolean }[];
    priceGuidance: {
        recommended: string;
        historical: string;
        competitors: string;
    };
};

function safeFileName(name: string) {
    return name.replace(/[\\/:*?"<>|]/g, "_");
}

async function downloadFromUrl(url: string, fileName: string) {
    const res = await fetch(url);
    if (!res.ok) throw new Error("download_failed");
    const blob = await res.blob();

    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(objectUrl);
}

function downloadText(content: string, fileName: string) {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const objectUrl = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(objectUrl);
}

function buildTextNotice(bid: Bid) {
    const lines: string[] = [];

    lines.push("입찰 공고문(텍스트 버전)");
    lines.push("=".repeat(60));
    lines.push("");
    lines.push(`공고 ID: ${bid.id}`);
    lines.push(`공고명: ${bid.title}`);
    lines.push(`발주기관: ${bid.agency}`);
    lines.push(`지역: ${bid.region}`);
    lines.push(`예산: ${bid.budget}`);
    lines.push(`공고일: ${bid.announcementDate}`);
    lines.push(`마감일: ${bid.deadline}`);
    lines.push(`유형/상태: ${bid.type} / ${bid.status}`);
    lines.push("");
    lines.push("설명");
    lines.push("-".repeat(60));
    lines.push(bid.description || "");
    lines.push("");
    lines.push("자격 요건");
    lines.push("-".repeat(60));
    bid.requirements.license.forEach((x, i) => lines.push(`${i + 1}. ${x}`));
    lines.push("");
    lines.push(`지역 요건: ${bid.requirements.location}`);
    lines.push(`실적 요건: ${bid.requirements.experience}`);
    lines.push(`기술인력 요건: ${bid.requirements.technicalStaff}`);
    lines.push("");
    lines.push("제출 서류 체크리스트");
    lines.push("-".repeat(60));
    bid.checklist.forEach((x) => lines.push(`- [${x.checked ? "x" : " "}] ${x.item}`));
    lines.push("");
    lines.push("리스크");
    lines.push("-".repeat(60));
    bid.risks.forEach((x) => lines.push(`- (${x.level}) ${x.text}`));
    lines.push("");
    lines.push("투찰 가이드");
    lines.push("-".repeat(60));
    lines.push(`추천 투찰 범위: ${bid.priceGuidance.recommended}`);
    lines.push(`과거 데이터: ${bid.priceGuidance.historical}`);
    lines.push(`예상 경쟁: ${bid.priceGuidance.competitors}`);
    lines.push("");

    return lines.join("\n");
}

function levelToKor(level: "high" | "medium" | "low") {
    if (level === "high") return "높음";
    if (level === "medium") return "보통";
    return "낮음";
}

function buildAiAnalysisReport(bid: Bid, completionRate: number) {
    const lines: string[] = [];

    lines.push("AI 분석 리포트");
    lines.push("=".repeat(70));
    lines.push("");
    lines.push(`[기본 정보]`);
    lines.push(`- 공고 ID: ${bid.id}`);
    lines.push(`- 공고명: ${bid.title}`);
    lines.push(`- 발주기관: ${bid.agency}`);
    lines.push(`- 지역: ${bid.region}`);
    lines.push(`- 예산: ${bid.budget}`);
    lines.push(`- 공고일: ${bid.announcementDate}`);
    lines.push(`- 마감일: ${bid.deadline}`);
    lines.push(`- 유형/상태: ${bid.type} / ${bid.status}`);
    lines.push("");

    lines.push(`[핵심 요약]`);
    lines.push(`- 사업 개요: ${bid.description || ""}`);
    lines.push("");

    lines.push(`[입찰 요건 분석]`);
    lines.push(`1) 자격 요건`);
    bid.requirements.license.forEach((x, i) => lines.push(`   ${i + 1}. ${x}`));
    lines.push("");
    lines.push(`2) 지역 요건`);
    lines.push(`- ${bid.requirements.location}`);
    lines.push("");
    lines.push(`3) 실적 요건`);
    lines.push(`- ${bid.requirements.experience}`);
    lines.push("");
    lines.push(`4) 기술인력 요건`);
    lines.push(`- ${bid.requirements.technicalStaff}`);
    lines.push("");

    lines.push(`[제출서류 준비도(체크리스트 기반)]`);
    lines.push(`- 진행률: ${completionRate.toFixed(0)}%`);
    lines.push(`- 완료 항목`);
    bid.checklist.filter((x) => x.checked).forEach((x) => lines.push(`  - ${x.item}`));
    lines.push(`- 미완료 항목`);
    bid.checklist.filter((x) => !x.checked).forEach((x) => lines.push(`  - ${x.item}`));
    lines.push("");

    lines.push(`[리스크/주의사항]`);
    bid.risks.forEach((r, i) => {
        lines.push(`${i + 1}. 중요도(${levelToKor(r.level)}): ${r.text}`);
    });
    lines.push("");

    lines.push(`[투찰 전략 가이드]`);
    lines.push(`- 추천 투찰 범위: ${bid.priceGuidance.recommended}`);
    lines.push(`- 과거 데이터: ${bid.priceGuidance.historical}`);
    lines.push(`- 예상 경쟁: ${bid.priceGuidance.competitors}`);
    lines.push("");

    lines.push(`[권고 액션]`);
    lines.push(`1) 미완료 서류를 우선 확보(특히 실적/재무/인증 관련).`);
    lines.push(`2) 지역/면허/기술인력 요건이 내부 보유 현황과 일치하는지 재검증.`);
    lines.push(`3) 마감 일정 역산하여 결재/제출 프로세스 사전 확정.`);
    lines.push("");

    lines.push("※ 본 리포트는 데모 데이터 기반 생성본이며, 실제 공고문 원문 기준으로 검증이 필요합니다.");
    lines.push("");

    return lines.join("\n");
}

export function BidSummary() {
    const navigate = useNavigate();
    const { bidId } = useParams();
    const numericBidId = Number(bidId);

    const [bid, setBid] = useState<Bid | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [checklist, setChecklist] = useState<Array<{ item: string; checked: boolean }>>([]);

    useEffect(() => {
        if (!Number.isFinite(numericBidId)) {
            setError("잘못된 공고 ID 입니다.");
            setBid(null);
            return;
        }

        const run = async () => {
            try {
                setLoading(true);
                setError(null);

                // ✅ API: GET /api/bid/{bidId}
                const res = await api(`/bids/${numericBidId}`, { method: "GET" });

                const item =
                    (res as any)?.data?.items?.[0] ??
                    (Array.isArray((res as any)?.data) ? (res as any).data[0] : null);

                if (!item) {
                    setError("상세 정보를 찾을 수 없습니다.");
                    setBid(null);
                    return;
                }

                // ✅ 서버 필드 -> 프론트 Bid 타입 매핑
                const mapped: Bid = {
                    id: Number(item.id ?? item.bid_id ?? item.bidId ?? numericBidId),
                    title: String(item.name ?? item.title ?? ""),
                    agency: String(item.organization ?? item.agency ?? ""),
                    region: String(item.region ?? ""),
                    budget: String(item.estimatePrice ?? item.estimate_Price ?? item.baseAmount ?? ""),
                    deadline: String(item.endDate ?? item.bidEnd ?? ""),
                    announcementDate: String(item.startDate ?? ""),
                    type: "공사",
                    status: "진행중",
                    description: String(item.analysisResult ?? ""),

                    documentUrl: item.bidReportURL ? String(item.bidReportURL) : undefined,
                    documentFileName: item.bidReportURL ? "첨부파일" : undefined,

                    requirements: { license: [], location: "", experience: "", technicalStaff: "" },
                    risks: [],
                    checklist: [],
                    priceGuidance: { recommended: "", historical: "", competitors: "" },
                };


                setBid(mapped);
            } catch (e: any) {
                setError(e?.message || "상세 조회 실패");
                setBid(null);
            } finally {
                setLoading(false);
            }
        };

        void run();
    }, [numericBidId]);

    useEffect(() => {
        if (!bid) return;
        setChecklist(bid.checklist ?? []);
    }, [bid?.id]);

    const completedItems = checklist.filter((item) => item.checked).length;
    const completionRate = checklist.length ? (completedItems / checklist.length) * 100 : 0;

    const handleDownloadNotice = async () => {
        if (!bid) return;
        const baseName = safeFileName(`공고문_${bid.id}_${bid.title}`);
        const pdfName = bid.documentFileName ? safeFileName(bid.documentFileName) : `${baseName}.pdf`;

        if (bid.documentUrl) {
            try {
                await downloadFromUrl(bid.documentUrl, pdfName);
                toast.success("공고문 다운로드가 시작되었습니다.");
                return;
            } catch {
                // 실패 시 텍스트로 폴백
            }
        }

        const txt = buildTextNotice({ ...bid, checklist });
        downloadText(txt, `${baseName}.txt`);
        toast.info("PDF가 없어 텍스트 공고문으로 다운로드했습니다.");
    };

    const handleDownloadAiReport = () => {
        if (!bid) return;
        const baseName = safeFileName(`AI_분석_리포트_${bid.id}_${bid.title}`);
        const report = buildAiAnalysisReport({ ...bid, checklist }, completionRate);
        downloadText(report, `${baseName}.txt`);
        toast.success("AI 분석 리포트 다운로드가 시작되었습니다.");
    };

    if (loading) return <div className="p-6">불러오는 중...</div>;
    if (error) return <div className="p-6 text-red-600">{error}</div>;
    if (!bid) return null;

    return (
        <div className="space-y-6">
            <div className="flex items-center gap-4">
                <Button variant="outline" size="sm" onClick={() => navigate(-1)}>
                    <ArrowLeft className="h-4 w-4 mr-1" />
                    뒤로가기
                </Button>

                <Button variant="ghost" size="sm" onClick={() => navigate("/bids")}>
                    목록으로
                </Button>

            </div>

            {/* Header */}
            <Card>
                <CardHeader>
                    <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                            <div className="flex items-center gap-2 mb-3">
                                <Badge>{bid.type}</Badge>
                                <Badge variant="outline">{bid.status}</Badge>
                                <Badge variant="destructive">마감임박</Badge>
                            </div>
                            <CardTitle className="text-2xl mb-2">{bid.title}</CardTitle>
                            <CardDescription>{bid.description || "상세 설명(analysisResult) 준비 중"}</CardDescription>
                        </div>
                    </div>
                </CardHeader>

                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        <div className="flex items-center gap-3">
                            <Building className="h-5 w-5 text-muted-foreground" />
                            <div>
                                <p className="text-sm text-muted-foreground">발주기관</p>
                                <p className="font-semibold">{bid.agency}</p>
                            </div>
                        </div>

                        <div className="flex items-center gap-3">
                            <MapPin className="h-5 w-5 text-muted-foreground" />
                            <div>
                                <p className="text-sm text-muted-foreground">지역</p>
                                <p className="font-semibold">{bid.region}</p>
                            </div>
                        </div>

                        <div className="flex items-center gap-3">
                            <DollarSign className="h-5 w-5 text-muted-foreground" />
                            <div>
                                <p className="text-sm text-muted-foreground">예산</p>
                                <p className="font-semibold">{bid.budget}</p>
                            </div>
                        </div>

                        <div className="flex items-center gap-3">
                            <Calendar className="h-5 w-5 text-muted-foreground" />
                            <div>
                                <p className="text-sm text-muted-foreground">마감일</p>
                                <p className="font-semibold text-red-600">{bid.deadline}</p>
                            </div>
                        </div>

                        <div className="flex items-center gap-3">
                            <div>
                                <p className="text-sm text-muted-foreground">첨부파일</p>
                                <p
                                    className="mt-1 text-blue-600 cursor-pointer hover:underline"
                                    onClick={handleDownloadNotice}
                                >
                                    {bid.documentFileName ?? "없음"}
                                </p>
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Main Content Tabs */}
            <Tabs defaultValue="summary" className="space-y-4">
                <TabsList>
                    <TabsTrigger value="summary">AI 요약</TabsTrigger>
                    <TabsTrigger value="checklist">서류 체크리스트</TabsTrigger>
                    <TabsTrigger value="risks">리스크 분석</TabsTrigger>
                    <TabsTrigger value="price">투찰 가이드</TabsTrigger>
                </TabsList>

                <TabsContent value="summary" className="space-y-4">
                    <Card>
                        <CardHeader>
                            <div className="flex items-start justify-between gap-4">
                                <div className="flex-1">
                                    <CardTitle className="flex items-center gap-2">
                                        <FileText className="h-5 w-5" />
                                        공고 핵심 요약
                                    </CardTitle>
                                    <CardDescription>AI가 분석한 주요 입찰 요건</CardDescription>
                                </div>

                                <div className="shrink-0">
                                    <Button variant="outline" className="gap-2" onClick={handleDownloadAiReport}>
                                        <Sparkles className="h-4 w-4" />
                                        AI 분석 리포트 다운로드
                                    </Button>
                                </div>
                            </div>
                        </CardHeader>

                        <CardContent className="space-y-6">
                            <div>
                                <h4 className="font-semibold mb-3">📋 자격 요건</h4>
                                {bid.requirements.license.length === 0 ? (
                                    <div className="text-sm text-muted-foreground">데이터 준비 중</div>
                                ) : (
                                    <ul className="space-y-2">
                                        {bid.requirements.license.map((item, index) => (
                                            <li key={index} className="flex items-start gap-2">
                                                <CheckCircle2 className="h-4 w-4 mt-0.5 text-green-600" />
                                                <span className="text-sm">{item}</span>
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </div>

                            <Separator />

                            <div>
                                <h4 className="font-semibold mb-3">📍 지역 요건</h4>
                                <p className="text-sm">{bid.requirements.location || "데이터 준비 중"}</p>
                            </div>

                            <Separator />

                            <div>
                                <h4 className="font-semibold mb-3">📈 실적 요건</h4>
                                <p className="text-sm">{bid.requirements.experience || "데이터 준비 중"}</p>
                            </div>

                            <Separator />

                            <div>
                                <h4 className="font-semibold mb-3">👷 기술인력 요건</h4>
                                <p className="text-sm">{bid.requirements.technicalStaff || "데이터 준비 중"}</p>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="checklist" className="space-y-4">
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <CheckCircle2 className="h-5 w-5" />
                                제출서류 체크리스트
                            </CardTitle>
                            <CardDescription>
                                진행률: {completedItems}/{checklist.length} ({completionRate.toFixed(0)}%)
                            </CardDescription>
                            <Progress value={completionRate} className="mt-2" />
                        </CardHeader>
                        <CardContent>
                            {checklist.length === 0 ? (
                                <div className="text-sm text-muted-foreground">체크리스트 데이터 준비 중</div>
                            ) : (
                                <div className="space-y-3">
                                    {checklist.map((item, index) => (
                                        <button
                                            key={index}
                                            type="button"
                                            onClick={() => {
                                                setChecklist((prev) =>
                                                    prev.map((x, i) => (i === index ? { ...x, checked: !x.checked } : x)),
                                                );
                                            }}
                                            className={`w-full text-left flex items-center gap-3 p-3 rounded-lg border transition ${
                                                item.checked
                                                    ? "bg-green-50 border-green-200"
                                                    : "bg-gray-50 hover:bg-gray-100"
                                            }`}
                                        >
                                            {item.checked ? (
                                                <CheckCircle2 className="h-5 w-5 text-green-600" />
                                            ) : (
                                                <Clock className="h-5 w-5 text-gray-400" />
                                            )}

                                            <span className={item.checked ? "line-through text-muted-foreground" : ""}>
                        {item.item}
                      </span>

                                            <span className="ml-auto text-xs text-muted-foreground">
                        {item.checked ? "완료" : "미완료"}
                      </span>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="risks" className="space-y-4">
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <AlertTriangle className="h-5 w-5" />
                                리스크 경고
                            </CardTitle>
                            <CardDescription>참여 전 확인이 필요한 사항</CardDescription>
                        </CardHeader>
                        <CardContent>
                            {bid.risks.length === 0 ? (
                                <div className="text-sm text-muted-foreground">리스크 데이터 준비 중</div>
                            ) : (
                                <div className="space-y-3">
                                    {bid.risks.map((risk, index) => (
                                        <div
                                            key={index}
                                            className={`flex items-start gap-3 p-4 rounded-lg border ${
                                                risk.level === "high"
                                                    ? "bg-red-50 border-red-200"
                                                    : risk.level === "medium"
                                                        ? "bg-yellow-50 border-yellow-200"
                                                        : "bg-blue-50 border-blue-200"
                                            }`}
                                        >
                                            <AlertTriangle
                                                className={`h-5 w-5 mt-0.5 ${
                                                    risk.level === "high"
                                                        ? "text-red-600"
                                                        : risk.level === "medium"
                                                            ? "text-yellow-600"
                                                            : "text-blue-600"
                                                }`}
                                            />
                                            <div>
                                                <Badge
                                                    variant={risk.level === "high" ? "destructive" : "outline"}
                                                    className="mb-2"
                                                >
                                                    {levelToKor(risk.level)}
                                                </Badge>
                                                <p className="text-sm">{risk.text}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="price" className="space-y-4">
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <TrendingUp className="h-5 w-5" />
                                투찰 범위 가이드
                            </CardTitle>
                            <CardDescription>과거 데이터 기반 추천 투찰률</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg">
                                <p className="text-sm text-muted-foreground mb-2">추천 투찰 범위</p>
                                <p className="text-3xl font-bold text-blue-600">
                                    {bid.priceGuidance.recommended || "데이터 준비 중"}
                                </p>
                            </div>

                            <Separator />

                            <div className="space-y-4">
                                <div>
                                    <h4 className="font-semibold mb-2">📊 과거 낙찰 데이터</h4>
                                    <p className="text-sm text-muted-foreground">
                                        {bid.priceGuidance.historical || "데이터 준비 중"}
                                    </p>
                                </div>

                                <div>
                                    <h4 className="font-semibold mb-2">🏢 예상 경쟁 현황</h4>
                                    <p className="text-sm text-muted-foreground">
                                        {bid.priceGuidance.competitors || "데이터 준비 중"}
                                    </p>
                                </div>
p
                                <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                                    <p className="text-sm">
                                        <strong>💡 인사이트:</strong> 유사 규모·지역의 낙찰률 분포를 참고해 84.0% 전후의
                                        투찰가를 검토하세요.
                                    </p>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    );
}
