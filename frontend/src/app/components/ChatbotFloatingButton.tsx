import { useState } from "react";
import { ChatbotModal } from "./ChatbotModal";

export function ChatbotFloatingButton() {
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-black text-white flex items-center justify-center shadow-lg hover:scale-105 transition"
      >
        AI
      </button>

      {/* Chatbot Modal */}
      {open && <ChatbotModal onClose={() => setOpen(false)} />}
    </>
  );
}
