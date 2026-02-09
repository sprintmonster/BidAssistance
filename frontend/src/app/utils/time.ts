// utils/time.ts
let serverNowOffsetMs = 0;

/**
 * 서버 시간과 클라이언트 시간 차이를 설정
 */
export function syncServerTime(serverNowIso: string) {
    const serverMs = new Date(serverNowIso).getTime();
    const clientMs = Date.now();
    serverNowOffsetMs = serverMs - clientMs;
}

/**
 * "통일된 현재 시간"
 */
export function nowMs() {
    return Date.now() + serverNowOffsetMs;
}
