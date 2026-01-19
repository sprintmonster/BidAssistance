import { useState } from "react";
import { MessageSquareText, X } from "lucide-react";
import { ChatbotModal } from "./ChatbotModal";

export function ChatbotFloatingButton() {
	const [open, setOpen] = useState(false);

	return (
		<>
			{/* Footer 안 가리도록 bottom 값을 위로 올림 (safe-area 고려) */}
			<div className="fixed right-4 md:right-6 bottom-[calc(env(safe-area-inset-bottom)+5rem)] z-50">
				<div className="rounded-full bg-gradient-to-r from-blue-500/70 via-cyan-400/60 to-indigo-500/70 p-[1px] shadow-lg hover:shadow-xl transition">
					<button
						type="button"
						onClick={() => setOpen(true)}
						aria-label="AI 도우미 열기"
						className={[
							"relative inline-flex items-center gap-2",
							"rounded-full px-4 py-3",
							"bg-slate-950/90 text-white backdrop-blur",
							"border border-white/10",
							"hover:bg-slate-950 hover:-translate-y-0.5 transition",
							"focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400",
						].join(" ")}
					>
						<span className="h-9 w-9 rounded-full bg-white/10 flex items-center justify-center border border-white/10">
							<MessageSquareText className="h-5 w-5" />
						</span>

						<span className="hidden sm:inline text-sm font-semibold tracking-[-0.01em]">
							AI 도우미
						</span>

						<span className="pointer-events-none absolute inset-0 rounded-full overflow-hidden">
							<span className="absolute -left-10 top-0 h-full w-16 rotate-12 bg-white/10 blur-md opacity-0 hover:opacity-100 transition-opacity" />
						</span>
					</button>
				</div>
			</div>

			{open && (
				<div className="fixed inset-0 z-[60]">
					{/* Overlay 클릭으로 닫기 */}
					<button
						type="button"
						className="absolute inset-0 bg-black/30 backdrop-blur-[2px]"
						aria-label="닫기 배경"
						onClick={() => setOpen(false)}
					/>

					<div className="absolute inset-0 flex items-center justify-center p-4">
						<div className="relative w-full max-w-4xl">
							{/* ✅ 바깥 X만 유지 */}
							<button
								type="button"
								aria-label="AI 도우미 닫기"
								onClick={() => setOpen(false)}
								className="absolute right-2 top-2 z-10 rounded-md bg-white/90 p-2 shadow hover:bg-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
							>
								<X className="h-5 w-5" />
							</button>

							{/* ✅ ChatbotModal에는 닫기 UI가 없다는 전제 */}
							<ChatbotModal onClose={() => setOpen(false)} />
						</div>
					</div>
				</div>
			)}
		</>
	);
}
