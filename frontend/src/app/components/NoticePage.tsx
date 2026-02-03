import { useEffect, useState, useMemo } from "react";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
    Bell,
    Megaphone,
    Search,
    Trash2,
    Settings,
    MoreHorizontal,
    Plus,
    X,
} from "lucide-react";
import { toast } from "sonner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import {
    getUserKeywords,
    addUserKeyword,
    deleteUserKeyword,
    getMyAlarms,
    deleteAlarm
} from "../api/community";
import type { UserKeyword, Alarm } from "../types/community";
import { useNavigate } from "react-router-dom";

export function NoticePage() {
    const navigate = useNavigate();
    const userId = Number(localStorage.getItem("userId") || 0);

    const [activeTab, setActiveTab] = useState("all");
    const [alarms, setAlarms] = useState<Alarm[]>([]);
    const [keywords, setKeywords] = useState<UserKeyword[]>([]);
    const [loading, setLoading] = useState(false);

    // Form states for adding keyword
    const [newKeyword, setNewKeyword] = useState("");
    const [minPrice, setMinPrice] = useState("");
    const [maxPrice, setMaxPrice] = useState("");

    useEffect(() => {
        if (!userId) return;
        loadData();
    }, [userId, activeTab]);

    const loadData = async () => {
        setLoading(true);
        try {
            if (activeTab === "settings") {
                const data = await getUserKeywords(userId);
                setKeywords(data);
            } else {
                const data = await getMyAlarms(userId);
                setAlarms(data);
            }
        } catch (error) {
            console.error(error);
            // toast.error("데이터를 불러오지 못했습니다.");
        } finally {
            setLoading(false);
        }
    };

    const handleAddKeyword = async () => {
        if (!newKeyword.trim()) {
            toast.error("키워드를 입력해주세요.");
            return;
        }

        try {
            await addUserKeyword({
                userId,
                keyword: newKeyword.trim(),
                minPrice: minPrice ? Number(minPrice) : null,
                maxPrice: maxPrice ? Number(maxPrice) : null,
            });
            toast.success("키워드가 추가되었습니다.");
            setNewKeyword("");
            setMinPrice("");
            setMaxPrice("");
            loadData();
        } catch (e) {
            toast.error("키워드 추가 실패");
        }
    };

    const handleDeleteKeyword = async (id: number) => {
        if (!confirm("이 키워드를 삭제하시겠습니까?")) return;
        try {
            await deleteUserKeyword(id);
            toast.success("삭제되었습니다.");
            loadData();
        } catch (e) {
            toast.error("삭제 실패");
        }
    };

    const handleDeleteAlarm = async (id: number) => {
        try {
            await deleteAlarm(id);
            setAlarms((prev) => prev.filter((a) => a.alarmId !== id));
            toast.success("알림이 삭제되었습니다.");
        } catch (e) {
            toast.error("삭제 실패");
        }
    };

    const handleAlarmClick = (alarm: Alarm) => {
        if (alarm.bidId) {
            navigate(`/bids/${alarm.bidId}`);
        }
    };

    const filteredAlarms = useMemo(() => {
        if (activeTab === "keyword") {
            return alarms.filter((a) => a.alarmType === "KEYWORD");
        }
        return alarms;
    }, [alarms, activeTab]);

    const formatPrice = (price: number | null) => {
        if (price === null) return "제한 없음";
        return new Intl.NumberFormat("ko-KR", {
            style: "currency",
            currency: "KRW",
            maximumFractionDigits: 0
        }).format(price);
    };

    if (!userId) {
        return (
            <div className="flex flex-col items-center justify-center py-20 bg-slate-50 rounded-lg">
                <Bell className="h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-xl font-semibold text-slate-900 mb-2">로그인이 필요합니다</h3>
                <p className="text-muted-foreground mb-6">알림을 확인하려면 먼저 로그인해주세요.</p>
                <Button onClick={() => navigate("/")}>로그인하러 가기</Button>
            </div>
        );
    }

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            <div>
                <h2 className="text-3xl font-bold tracking-tight mb-2">알림 센터</h2>
                <p className="text-muted-foreground">
                    키워드 알림과 시스템 공지를 한눈에 확인하세요.
                </p>
            </div>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
                <TabsList>
                    <TabsTrigger value="all">전체</TabsTrigger>
                    <TabsTrigger value="keyword">키워드</TabsTrigger>
                    <TabsTrigger value="settings">설정</TabsTrigger>
                </TabsList>

                <TabsContent value="all" className="space-y-4">
                    <AlarmList
                        alarms={filteredAlarms}
                        loading={loading}
                        onDelete={handleDeleteAlarm}
                        onClick={handleAlarmClick}
                        emptyMessage="도착한 알림이 없습니다."
                    />
                </TabsContent>

                <TabsContent value="keyword" className="space-y-4">
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-base flex items-center gap-2">
                                <Search className="h-4 w-4" />
                                키워드 알림
                            </CardTitle>
                            <CardDescription>
                                설정한 키워드와 가격 조건에 맞는 공고 알림만 모아봅니다.
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <AlarmList
                                alarms={filteredAlarms}
                                loading={loading}
                                onDelete={handleDeleteAlarm}
                                onClick={handleAlarmClick}
                                emptyMessage="키워드 알림이 없습니다."
                            />
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="settings" className="space-y-4">
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-base">키워드 설정</CardTitle>
                            <CardDescription>
                                관심 있는 단어와 가격 범위를 등록하면 알림을 보내드립니다.
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            {/* 입력 폼 */}
                            <div className="grid gap-4 p-4 border rounded-lg bg-slate-50">
                                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
                                    <div className="space-y-2 md:col-span-2">
                                        <Label>키워드</Label>
                                        <Input
                                            placeholder="예: 서버, AI, 유지보수"
                                            value={newKeyword}
                                            onChange={(e) => setNewKeyword(e.target.value)}
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>최소 금액 (원)</Label>
                                        <Input
                                            type="number"
                                            placeholder="0"
                                            value={minPrice}
                                            onChange={(e) => setMinPrice(e.target.value)}
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label>최대 금액 (원)</Label>
                                        <Input
                                            type="number"
                                            placeholder="제한 없음"
                                            value={maxPrice}
                                            onChange={(e) => setMaxPrice(e.target.value)}
                                        />
                                    </div>
                                </div>
                                <div className="flex justify-end">
                                    <Button onClick={handleAddKeyword} className="gap-2">
                                        <Plus className="h-4 w-4" />
                                        키워드 추가
                                    </Button>
                                </div>
                            </div>

                            {/* 목록 */}
                            <div className="space-y-4">
                                <h4 className="font-semibold text-sm text-foreground">
                                    등록된 키워드 ({keywords.length})
                                </h4>
                                {loading && <div className="text-sm text-muted-foreground">로드 중...</div>}
                                {!loading && keywords.length === 0 && (
                                    <div className="text-center py-8 text-sm text-muted-foreground border rounded-md border-dashed">
                                        등록된 키워드가 없습니다.
                                    </div>
                                )}
                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                                    {keywords.map((k) => (
                                        <div
                                            key={k.id}
                                            className="flex items-center justify-between p-3 border rounded-md bg-white shadow-sm"
                                        >
                                            <div className="min-w-0">
                                                <div className="font-medium truncate text-blue-700">
                                                    {k.keyword}
                                                </div>
                                                <div className="text-xs text-muted-foreground mt-1">
                                                    {k.minPrice != null || k.maxPrice != null ? (
                                                        <span>
                                                            {formatPrice(k.minPrice)} ~ {formatPrice(k.maxPrice)}
                                                        </span>
                                                    ) : (
                                                        "가격 제한 없음"
                                                    )}
                                                </div>
                                            </div>
                                            <Button
                                                variant="ghost"
                                                size="sm"
                                                onClick={() => handleDeleteKeyword(k.id)}
                                                className="h-8 w-8 p-0 text-muted-foreground hover:text-red-600"
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    );
}

function AlarmList({
    alarms,
    loading,
    onDelete,
    onClick,
    emptyMessage,
}: {
    alarms: Alarm[];
    loading: boolean;
    onDelete: (id: number) => void;
    onClick: (alarm: Alarm) => void;
    emptyMessage: string;
}) {
    if (loading) {
        return <div className="py-8 text-center text-muted-foreground">불러오는 중...</div>;
    }

    if (alarms.length === 0) {
        return (
            <Card>
                <CardContent className="py-12 text-center text-muted-foreground">
                    <Bell className="h-8 w-8 mx-auto mb-3 opacity-20" />
                    {emptyMessage}
                </CardContent>
            </Card>
        );
    }

    return (
        <div className="space-y-3">
            {alarms.map((alarm) => (
                <Card
                    key={alarm.alarmId}
                    className="hover:shadow-md transition cursor-pointer group"
                    onClick={() => onClick(alarm)}
                >
                    <CardContent className="p-4 flex items-start gap-3">
                        <div className="mt-1">
                            {alarm.alarmType === "KEYWORD" ? (
                                <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
                                    <Search className="h-4 w-4" />
                                </div>
                            ) : (
                                <div className="h-8 w-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-600">
                                    <Bell className="h-4 w-4" />
                                </div>
                            )}
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                                <Badge variant={alarm.alarmType === "KEYWORD" ? "default" : "secondary"}>
                                    {alarm.alarmType === "KEYWORD" ? "키워드" : "알림"}
                                </Badge>
                                <span className="text-xs text-muted-foreground">
                                    {new Date(alarm.date).toLocaleString()}
                                </span>
                            </div>
                            <p className="text-sm font-medium text-foreground line-clamp-2">
                                {alarm.content}
                            </p>
                            {alarm.bidName && (
                                <p className="text-xs text-muted-foreground mt-1 truncate">
                                    공고명: {alarm.bidName}
                                </p>
                            )}
                        </div>
                        <Button
                            variant="ghost"
                            size="sm"
                            className="opacity-0 group-hover:opacity-100 transition-opacity"
                            onClick={(e) => {
                                e.stopPropagation();
                                onDelete(alarm.alarmId);
                            }}
                        >
                            <X className="h-4 w-4" />
                        </Button>
                    </CardContent>
                </Card>
            ))}
        </div>
    );
}
