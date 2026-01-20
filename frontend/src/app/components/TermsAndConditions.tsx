// 입찰인사이트 서비스 이용약관 페이지
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

export function TermsAndConditionsPage() {
	const navigate = useNavigate();
	const isAuthed = useMemo(() => !!localStorage.getItem("accessToken"), []);

	return (
		<Card>
			<CardHeader>
				<CardTitle>서비스 이용약관</CardTitle>
			</CardHeader>
			<CardContent>
				<p>서비스 이용약관은 다음과 같습니다.</p>
			</CardContent>
		</Card>
	);
}
