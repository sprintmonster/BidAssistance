import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Send, Bot, User } from "lucide-react";
import { Badge } from "./ui/badge";
import { useState, useRef, useEffect } from "react";

interface Message {
    id: number;
    text: string;
    sender: "user" | "bot";
    timestamp: string;
    suggestions?: string[];
}

export function ChatbotPage() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: 1,
            text: "안녕하세요! 입찰 인텔리전스 AI 어시스턴트입니다. 무엇을 도와드릴까요?",
            sender: "bot",
            timestamp: "10:00",
            suggestions: ["서울 지역 신규 공고 알려줘", "30억 이하 공사 찾아줘", "마감 임박한 공고는?", "최근 낙찰률 분석해줘"],
        },
    ]);

    const [inputValue, setInputValue] = useState("");
    const bottomRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "end",
        });
    }, [messages]);

    const generateBotResponse = (userInput: string): Message => {
        const input = userInput.toLowerCase();
        let responseText = "";
        let suggestions: string[] = [];

        if (input.includes("서울") || input.includes("지역")) {
            responseText =
                "서울 지역의 현재 진행 중인 공고는 총 23건입니다.\n\n주요 공고:\n• 서울시 강남구 도로 보수공사 (35억원, D-2)\n• 서울시 송파구 공공건물 신축 (52억원, D-7)\n• 서울시 마포구 학교시설 개선 (18억원, D-5)\n\n자세한 내용을 보시려면 '공고 찾기' 메뉴를 이용해주세요.";
            suggestions = ["강남구 공고 상세보기", "30억 이하만 보기", "마감일 빠른 순"];
        } else if (input.includes("30억") || input.includes("금액") || input.includes("예산")) {
            responseText =
                "30억원 이하의 공고는 현재 45건입니다.\n\n추천 공고:\n• 인천 연수구 학교시설 개선 (12억원)\n• 경기 수원시 주차장 건설 (23억원)\n• 부산 해운대구 복지센터 (18억원)\n\n평균 낙찰률은 84.3%입니다.";
            suggestions = ["지역별로 보기", "낙찰률 분석", "마감 임박 공고"];
        } else if (input.includes("마감") || input.includes("임박")) {
            responseText =
                "마감 3일 이내 공고는 8건입니다. ⚠️\n\n긴급:\n• 서울 강남구 도로공사 (D-2, 35억원)\n• 인천 연수구 학교시설 (D-4, 12억원)\n• 경기 성남시 건축공사 (D-5, 87억원)\n\n서류 준비를 서둘러주세요!";
            suggestions = ["체크리스트 확인", "투찰가 가이드", "장바구니 담기"];
        } else {
            responseText =
                '죄송합니다. 질문을 이해하지 못했습니다. 다음과 같이 질문해주세요:\n\n• "서울 지역 공고 알려줘"\n• "30억 이하 공사 찾아줘"\n• "마감 임박한 공고는?"\n• "최근 낙찰률 분석해줘"';
            suggestions = ["서울 지역 신규 공고", "30억 이하 공사", "마감 임박 공고"];
        }

        return {
            id: Date.now(),
            text: responseText,
            sender: "bot",
            timestamp: new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }),
            suggestions,
        };
    };

    //  입력창을 안 거치고, 전달받은 text로 바로 전송하는 함수
    const sendMessage = (text: string) => {
        const trimmed = text.trim();
        if (!trimmed) return;

        const userMessage: Message = {
            id: Date.now(),
            text: trimmed,
            sender: "user",
            timestamp: new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }),
        };

        setMessages((prev) => [...prev, userMessage]);
        setInputValue(""); // 전송 후 입력창 비우기

        setTimeout(() => {
            const botResponse = generateBotResponse(trimmed);
            setMessages((prev) => [...prev, botResponse]);
        }, 1000);
    };

    //  기존 전송 버튼/엔터는 입력값을 sendMessage로 보내기
    const handleSend = () => {
        sendMessage(inputValue);
    };

    //  추천 질문 클릭하면 "입력만"이 아니라 바로 전송
    const handleSuggestionClick = (suggestion: string) => {
        sendMessage(suggestion);
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="space-y-6 max-h-full overflow-hidden">
            <div>
                <h2 className="text-3xl mb-2">AI 어시스턴트</h2>
                <p className="text-muted-foreground">입찰 관련 질문을 자유롭게 물어보세요</p>
            </div>

            <Card className="h-[600px] flex flex-col">
                <CardHeader className="border-b shrink-0">
                    <div className="flex items-center gap-3">
                        <Avatar>
                            <AvatarFallback className="bg-blue-600 text-white">
                                <Bot className="h-5 w-5" />
                            </AvatarFallback>
                        </Avatar>
                        <div>
                            <CardTitle className="text-lg">입찰 AI 어시스턴트</CardTitle>
                            <CardDescription className="flex items-center gap-2">
                                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                                온라인
                            </CardDescription>
                        </div>
                    </div>
                </CardHeader>

                <ScrollArea className="flex-1 overflow-hidden">
                    <div className="h-full overflow-y-auto p-4 space-y-4">
                        {messages.map((message) => (
                            <div key={message.id} className={`flex gap-3 ${message.sender === "user" ? "flex-row-reverse" : ""}`}>
                                <Avatar className="flex-shrink-0">
                                    <AvatarFallback
                                        className={message.sender === "bot" ? "bg-blue-600 text-white" : "bg-gray-600 text-white"}
                                    >
                                        {message.sender === "bot" ? <Bot className="h-4 w-4" /> : <User className="h-4 w-4" />}
                                    </AvatarFallback>
                                </Avatar>

                                <div className={`flex-1 max-w-[80%] ${message.sender === "user" ? "flex flex-col items-end" : ""}`}>
                                    <div className={`rounded-lg p-4 ${message.sender === "bot" ? "bg-gray-100" : "bg-blue-600 text-white"}`}>
                                        <p className="text-sm whitespace-pre-line">{message.text}</p>
                                    </div>
                                    <p className="text-xs text-muted-foreground mt-1">{message.timestamp}</p>

                                    {message.suggestions && message.suggestions.length > 0 && (
                                        <div className="flex flex-wrap gap-2 mt-3">
                                            {message.suggestions.map((suggestion, index) => (
                                                <Badge
                                                    key={index}
                                                    variant="outline"
                                                    className="cursor-pointer hover:bg-blue-50"
                                                    onClick={() => handleSuggestionClick(suggestion)}
                                                >
                                                    {suggestion}
                                                </Badge>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                        <div ref={bottomRef} />
                    </div>
                </ScrollArea>

                <CardContent className="border-t pt-4 shrink-0">
                    <div className="flex gap-2">
                        <Input
                            placeholder="메시지를 입력하세요..."
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            onKeyPress={handleKeyPress}
                        />
                        <Button onClick={handleSend}>
                            <Send className="h-4 w-4" />
                        </Button>
                    </div>
                </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card>
                <CardHeader>
                    <CardTitle className="text-lg">자주 묻는 질문</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <Button
                            variant="outline"
                            className="justify-start h-auto py-3"
                            onClick={() => handleSuggestionClick("서울 지역 신규 공고 알려줘")}
                        >
                            <div className="text-left">
                                <p className="font-semibold">지역별 공고 검색</p>
                                <p className="text-xs text-muted-foreground">특정 지역의 공고 찾기</p>
                            </div>
                        </Button>

                        <Button
                            variant="outline"
                            className="justify-start h-auto py-3"
                            onClick={() => handleSuggestionClick("30억 이하 공사 찾아줘")}
                        >
                            <div className="text-left">
                                <p className="font-semibold">예산별 필터링</p>
                                <p className="text-xs text-muted-foreground">금액대별 공고 조회</p>
                            </div>
                        </Button>

                        <Button
                            variant="outline"
                            className="justify-start h-auto py-3"
                            onClick={() => handleSuggestionClick("마감 임박한 공고는?")}
                        >
                            <div className="text-left">
                                <p className="font-semibold">마감 임박 공고</p>
                                <p className="text-xs text-muted-foreground">긴급 입찰 기회 확인</p>
                            </div>
                        </Button>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
