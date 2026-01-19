import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Bot, Send, Sparkles } from "lucide-react";

import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { ScrollArea } from "./ui/scroll-area";
import { Textarea } from "./ui/textarea";
import { Separator } from "./ui/separator";
import { Avatar, AvatarFallback } from "./ui/avatar";

type Sender = "user" | "bot";

type Message = {
	id: number;
	sender: Sender;
	text: string;
	timestamp: string;
	suggestions?: string[];
};

const QUICK_PROMPTS = [
	{ title: "마감 임박 공고", desc: "D-3 이내만 요약", text: "마감 임박한 공고만 요약해줘" },
	{ title: "지역별 공고", desc: "서울/경기 등", text: "서울 지역 공고 알려줘" },
	{ title: "예산 조건", desc: "30억 이하", text: "30억 이하 공사 찾아줘" },
	{ title: "낙찰 트렌드", desc: "최근 6개월", text: "최근 낙찰률 분석해줘" },
];

function nowTime() {
	return new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
}

function generateBotResponse(userInput: string): { text: string; suggestions: string[] } {
	const input = userInput.toLowerCase();

	if (input.includes("서울") || input.includes("지역")) {
		return {
			text:
				"서울 지역의 현재 진행 중인 공고는 총 23건입니다.\n\n주요 공고:\n• 서울시 강남구 도로 보수공사 (35억원, D-2)\n• 서울시 송파구 공공건물 신축 (52억원, D-7)\n• 서울시 마포구 학교시설 개선 (18억원, D-5)\n\n자세한 내용은 ‘공고 찾기’에서 조건 필터로 확인하시면 정확합니다.",
			suggestions: ["강남구 공고만 보기", "30억 이하로 제한", "마감 빠른 순으로 정렬"],
		};
	}

	if (input.includes("30억") || input.includes("금액") || input.includes("예산")) {
		return {
			text:
				"30억원 이하 공고는 현재 45건입니다.\n\n추천 공고:\n• 인천 연수구 학교시설 개선 (12억원)\n• 경기 수원시 주차장 건설 (23억원)\n• 부산 해운대구 복지센터 (18억원)\n\n원하시면 ‘지역 + 업종 + 마감’ 조합으로 우선순위를 같이 잡아드릴까요?",
			suggestions: ["지역별로 분해", "마감 임박만 보기", "경쟁률 기준 추천"],
		};
	}

	if (input.includes("마감") || input.includes("임박")) {
		return {
			text:
				"마감 3일 이내 공고는 8건입니다. ⚠️\n\n긴급:\n• 서울 강남구 도로공사 (D-2, 35억원)\n• 인천 연수구 학교시설 (D-4, 12억원)\n• 경기 성남시 건축공사 (D-5, 87억원)\n\n서류 체크리스트가 필요하시면 항목별로 정리해드릴게요.",
			suggestions: ["서류 체크리스트 만들어줘", "투찰가 체크 포인트", "장바구니에 담는 기준"],
		};
	}

	if (input.includes("낙찰") || input.includes("분석")) {
		return {
			text:
				"최근 6개월 낙찰률 요약:\n\n• 평균 낙찰률: 84.2%\n• 낙찰 건수: 186건\n• 평균 경쟁률: 3.8:1\n• 유찰률: 7.2%\n\n10~30억 구간에서 상대적으로 성공률이 높게 나타나는 경향이 있습니다.",
			suggestions: ["기관별 분석", "월별 추이", "경쟁률 높은 업종"],
		};
	}

	return {
		text:
			"원하시는 질문을 조금만 더 구체적으로 적어주세요.\n\n예시:\n• “서울/경기 10~50억 시설공사, 마감 3일 이내”\n• “최근 6개월 낙찰률과 경쟁률 추이”\n• “마감 임박 공고 우선순위 기준”",
		suggestions: ["서울 지역 공고 알려줘", "30억 이하 공사 찾아줘", "마감 임박 공고는?", "낙찰률 분석해줘"],
	};
}

function Bubble({ message }: { message: Message }) {
	const isUser = message.sender === "user";

	return (
		<div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
			<Avatar className="h-9 w-9 shrink-0">
				<AvatarFallback
					className={isUser ? "bg-slate-800 text-white" : "bg-blue-600 text-white"}
				>
					{isUser ? "나" : <Bot className="h-4 w-4" />}
				</AvatarFallback>
			</Avatar>

			<div className={`max-w-[78%] ${isUser ? "items-end" : "items-start"} flex flex-col`}>
				<div
					className={[
						"rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-line",
						isUser ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-900",
					].join(" ")}
				>
					{message.text}
				</div>
				<div className="mt-1 text-[11px] text-slate-500 tabular-nums">{message.timestamp}</div>

				{!isUser && message.suggestions && message.suggestions.length > 0 && (
					<div className="mt-2 flex flex-wrap gap-2">
						{message.suggestions.map((s, idx) => (
							<Badge key={idx} variant="outline" className="cursor-pointer hover:bg-slate-50">
								{s}
							</Badge>
						))}
					</div>
				)}
			</div>
		</div>
	);
}

export function ChatbotModal({ onClose }: { onClose: () => void }) {
	// 현재는 "바깥 X"로만 닫도록 설계되어 내부에서는 onClose 미사용
	void onClose;

	const [messages, setMessages] = useState<Message[]>([
		{
			id: 1,
			sender: "bot",
			text: "안녕하세요. 입찰인사이트 AI 도우미입니다.\n원하시는 조건(지역/예산/마감)을 말해주시면 공고를 빠르게 좁혀드릴게요.",
			timestamp: nowTime(),
			suggestions: ["서울 지역 공고 알려줘", "30억 이하 공사 찾아줘", "마감 임박 공고는?", "낙찰률 분석해줘"],
		},
	]);

	const [input, setInput] = useState("");
	const [isTyping, setIsTyping] = useState(false);

	const bottomRef = useRef<HTMLDivElement | null>(null);

	useEffect(() => {
		bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
	}, [messages, isTyping]);

	const send = (text: string) => {
		const trimmed = text.trim();
		if (!trimmed) return;

		const userMsg: Message = {
			id: Date.now(),
			sender: "user",
			text: trimmed,
			timestamp: nowTime(),
		};

		setMessages((prev) => [...prev, userMsg]);
		setInput("");
		setIsTyping(true);

		window.setTimeout(() => {
			const res = generateBotResponse(trimmed);
			const botMsg: Message = {
				id: Date.now() + 1,
				sender: "bot",
				text: res.text,
				timestamp: nowTime(),
				suggestions: res.suggestions,
			};
			setMessages((prev) => [...prev, botMsg]);
			setIsTyping(false);
		}, 700);
	};

	const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			send(input);
		}
	};

	return (
		<div className="w-full h-[min(760px,calc(100vh-2rem))] max-w-[980px] bg-white border rounded-2xl shadow-2xl overflow-hidden flex">
			{/* Left: Chat */}
			<div className="flex-1 min-w-0 flex flex-col">
				{/* Header (새대화/X 없음) */}
				<div className="px-5 py-4 bg-slate-950 text-white flex items-center justify-between">
					<div className="flex items-center gap-3">
						<div className="h-10 w-10 rounded-xl bg-white/10 border border-white/10 flex items-center justify-center">
							<Sparkles className="h-5 w-5" />
						</div>
						<div className="leading-tight">
							<div className="text-sm font-semibold tracking-[-0.01em]">AI 도우미</div>
							<div className="mt-0.5 flex items-center gap-2 text-xs text-white/70">
								<span className="inline-flex items-center gap-1">
									<span className="h-2 w-2 rounded-full bg-emerald-400" />
									온라인
								</span>
								<span>·</span>
								<span>질문을 구체적으로 적을수록 정확도가 올라갑니다</span>
							</div>
						</div>
					</div>
					<div className="hidden" />
				</div>

				{/* Messages */}
				<ScrollArea className="flex-1 bg-white">
					<div className="p-5 space-y-4">
						{messages.map((m) => (
							<div key={m.id}>
								<Bubble message={m} />
							</div>
						))}

						{isTyping && (
							<div className="flex gap-3">
								<Avatar className="h-9 w-9 shrink-0">
									<AvatarFallback className="bg-blue-600 text-white">
										<Bot className="h-4 w-4" />
									</AvatarFallback>
								</Avatar>
								<div className="rounded-2xl bg-slate-100 px-4 py-3 text-sm text-slate-600">
									작성 중…
								</div>
							</div>
						)}

						<div ref={bottomRef} />
					</div>
				</ScrollArea>

				{/* ✅ Input (겹치는 안내/추천 버튼 제거 버전) */}
				<div className="border-t bg-white p-4">
					<div className="flex gap-2 items-end">
						<div className="flex-1">
							<Textarea
								value={input}
								onChange={(e) => setInput(e.target.value)}
								onKeyDown={onKeyDown}
								placeholder="예: 서울/경기 10~50억 시설공사, 마감 3일 이내만"
								className="min-h-[44px] max-h-[140px] resize-none"
							/>
						</div>

						<Button onClick={() => send(input)} className="h-11 px-4">
							<Send className="h-4 w-4 mr-2" />
							전송
						</Button>
					</div>
				</div>
			</div>

			{/* Right: Quick Panel */}
			<div className="hidden lg:flex w-[320px] border-l bg-slate-50 flex-col">
				<div className="p-5">
					<div className="text-sm font-semibold text-slate-900">빠른 질문</div>
					<div className="mt-1 text-xs text-slate-600">
						자주 쓰는 질문을 한 번에 보내세요.
					</div>
				</div>

				<Separator />

				<div className="p-4 space-y-2">
					{QUICK_PROMPTS.map((p) => (
						<button
							key={p.title}
							type="button"
							onClick={() => send(p.text)}
							className="w-full text-left rounded-xl border bg-white p-3 hover:border-slate-300 hover:shadow-sm transition"
						>
							<div className="text-sm font-semibold text-slate-900">{p.title}</div>
							<div className="mt-0.5 text-xs text-slate-600">{p.desc}</div>
							<div className="mt-2 text-[11px] text-slate-500">{p.text}</div>
						</button>
					))}
				</div>

			</div>
		</div>
	);
}
