export function ChatbotModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 bg-black/30 flex items-center justify-center">
      <div className="bg-white w-[420px] h-[600px] rounded-xl shadow-xl flex flex-col">
        {/* Header */}
        <div className="px-4 py-3 border-b flex justify-between items-center">
          <div className="font-semibold">AI 챗봇</div>
          <button onClick={onClose}>✕</button>
        </div>

        {/* Body */}
        <div className="flex-1 p-4 text-sm text-gray-600">
          입찰 관련 질문을 자유롭게 물어보세요.
        </div>
      </div>
    </div>
  );
}
