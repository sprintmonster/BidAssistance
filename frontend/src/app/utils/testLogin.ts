export const ENABLE_TEST_LOGIN = import.meta.env.VITE_ENABLE_TEST_LOGIN === "true";

export const TEST_LOGIN = {
    email: "test@test.com",
    password: "login123!",
} as const;
