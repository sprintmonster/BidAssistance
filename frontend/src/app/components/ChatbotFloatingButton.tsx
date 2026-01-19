import { useState } from "react";
import { MessageSquareText, X } from "lucide-react";
import { ChatbotModal } from "./ChatbotModal";

export function ChatbotFloatingButton() {
	const [open, setOpen] = useState(false);

	return (
		<>
			<button
				type="button"
				onClick={() => setOpen(true)}
				aria-label="AI 챗봇 열기"
				className="fixed bottom-6 right-6 z-50 inline-flex items-center gap-2 rounded-full bg-black px-4 py-3 text-white shadow-lg hover:shadow-xl hover:translate-y-[-1px] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
			>
				<MessageSquareText className="h-5 w-5" />
				<span className="text-sm font-medium">AI 챗봇</span>
			</button>

			{open && (
				<div className="fixed inset-0 z-[60]">
					<button
						type="button"
						className="absolute inset-0 bg-black/30"
						aria-label="닫기 배경"
						onClick={() => setOpen(false)}
					/>
					<div className="absolute inset-0 flex items-center justify-center p-4">
						<div className="relative w-full max-w-4xl">
							<button
								type="button"
								aria-label="AI 챗봇 닫기"
								onClick={() => setOpen(false)}
								className="absolute right-2 top-2 z-10 rounded-md p-2 hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
							>
								<X className="h-5 w-5" />
							</button>

							<ChatbotModal onClose={() => setOpen(false)} />
						</div>
					</div>
				</div>
			)}
		</>
	);
}
