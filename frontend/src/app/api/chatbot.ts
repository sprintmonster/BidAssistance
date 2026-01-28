export interface ChatRequest {
    query: string;
    thread_id?: string; // 세션 스레드 ID
}

export interface ChatResponse {
    status: string;
    message: string;
    data: {
        message: string;
    }
}
export const fetchChatResponse = async (request: ChatRequest): Promise<ChatResponse> => {
    try {
        const response = await fetch("/api/chatbots", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(request),
        });
        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error("Chatbot API Error:", error);
        throw error;
    }
}