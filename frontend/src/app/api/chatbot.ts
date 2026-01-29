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

export const fetchChatWithFile = async (formData: FormData): Promise<ChatResponse> => {
    try {
        // 파일 업로드는 헤더에 Content-Type을 쓰면 안 됩니다! (브라우저가 자동 설정)
        const response = await fetch("/api/chatbots/upload", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error(`Server Error: ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error("Chatbot File Upload Error:", error);
        throw error;
    }
};