import { useState } from "react";
import { login } from "../api/auth";
import { Button } from "./ui/button";

export function Login({
  onSuccess,
}: {
  onSuccess: () => void;
}) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async () => {
    try {
      setLoading(true);
      setError("");
      const res = await login(email, password);
      localStorage.setItem("accessToken", res.accessToken);
      onSuccess();
    } catch (e: any) {
      setError("로그인 실패: 아이디 또는 비밀번호 확인");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-sm mx-auto space-y-4">
      <h2 className="text-xl font-bold">로그인</h2>

      <input
        className="border p-2 w-full"
        placeholder="이메일"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />

      <input
        type="password"
        className="border p-2 w-full"
        placeholder="비밀번호"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />

      {error && <p className="text-red-600 text-sm">{error}</p>}

      <Button onClick={submit} disabled={loading} className="w-full">
        {loading ? "로그인 중..." : "로그인"}
      </Button>
    </div>
  );
}
