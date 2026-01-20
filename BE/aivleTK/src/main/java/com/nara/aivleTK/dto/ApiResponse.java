package com.nara.aivleTK.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class ApiResponse<T> {
    private String status;
    private String messages;
    private T data;

    public static <T> ApiResponse<T> success(T data) {
        return new ApiResponse<>("success", "요청 성공", data);
    }

    public static <T> ApiResponse<T> success(String messages, T data) {
        return new ApiResponse<>("success", messages, data);
    }

    public static <T> ApiResponse<T> success(String messages) {
        return new ApiResponse<>("success", messages, null);
    }

    public static <T> ApiResponse<T> error(String messages) {
        return new ApiResponse<>("error", messages, null);
    }
}
