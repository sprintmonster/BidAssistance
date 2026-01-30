import { api } from "./client";

export interface ChatRequest {
    query: string;
    thread_id?: string; // 세션 스레드 ID
    payload?: any; // 파일 담기 위해
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
        const response = await api<ChatResponse>("/chatbots", {
            method: "POST",
            body: JSON.stringify(request),
        });


        return response;
    } catch (error) {
        console.error("Chatbot API Error:", error);
        throw error;
    }
}