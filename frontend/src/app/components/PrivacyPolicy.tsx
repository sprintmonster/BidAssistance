// 입찰인사이트 개인정보처리방침 페이지
import { useMemo, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../api/auth";

import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "./ui/card";

export function PrivacyPolicyPage() {
	const navigate = useNavigate();
	const isAuthed = useMemo(() => !!localStorage.getItem("accessToken"), []);

	return (
		<Card>
			<CardHeader>
				<CardTitle>개인정보처리방침</CardTitle>
			</CardHeader>
			<CardContent>
				<p>개인정보처리방침은 다음과 같습니다.</p>
			</CardContent>
		</Card>
	);
}
