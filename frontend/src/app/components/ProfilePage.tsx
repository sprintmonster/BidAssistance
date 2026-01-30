import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { getUserProfile, updateUserProfile } from "../api/users";
import { getCompany, upsertCompany } from "../api/company";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Separator } from "./ui/separator";
import { Badge } from "./ui/badge";

import {
	ACCESS_CONTROL,
	get_password_changed_at_ms,
	is_password_expired,
	set_password_changed_now_for_user,
} from "../utils/accessControl";

interface ProfilePageProps {
	userEmail?: string;
}

type ProfileTab = "info" | "company" | "notifications" | "subscription";
const PROFILE_TAB_STORAGE_KEY = "profile.activeTab";

function isProfileTab(v: string | null): v is ProfileTab {
	return v === "info" || v === "company" || v === "notifications" || v === "subscription";
}

function safeGetLocalStorage(key: string) {
	if (typeof window === "undefined") return null;
	return localStorage.getItem(key);
}

function safeSetLocalStorage(key: string, value: string) {
	if (typeof window === "undefined") return;
	localStorage.setItem(key, value);
}

function readInitialTab(): ProfileTab {
	const v = safeGetLocalStorage(PROFILE_TAB_STORAGE_KEY);
	if (isProfileTab(v)) return v;
	return "info";
}

function resolveUserId() {
	return safeGetLocalStorage("userId") || "";
}

function resolveEmail(userEmail?: string) {
	if (userEmail && userEmail.trim()) return userEmail.trim();
	return safeGetLocalStorage("email") || "";
}

function avatarLetter(email: string) {
	if (!email) return "?";
	return email.charAt(0).toUpperCase();
}

function days_until_expiry(user_id: string) {
	const ts = get_password_changed_at_ms(user_id);
	if (!ts) return null;

	const max_ms = ACCESS_CONTROL.PASSWORD_MAX_AGE_DAYS * 24 * 60 * 60 * 1000;
	const remain = ts + max_ms - Date.now();
	return Math.ceil(remain / (24 * 60 * 60 * 1000));
}

export function ProfilePage({ userEmail }: ProfilePageProps) {
	const navigate = useNavigate();
	const location = useLocation() as any;

	const passwordExpiredFromRoute = Boolean(location?.state?.passwordExpired);
	const fromAfterChange = String(location?.state?.fromAfterChange || "");

	const [activeTab, setActiveTab] = useState<ProfileTab>(() => readInitialTab());
	const [loading, setLoading] = useState(false);
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const userId = useMemo(() => resolveUserId(), []);
	const [name, setName] = useState("김철수");
	const [email, setEmail] = useState(() => resolveEmail(userEmail));
	const [role, setRole] = useState<number>(0);

	const [currentPassword, setCurrentPassword] = useState("");
	const [newPassword, setNewPassword] = useState("");
	const [confirmPassword, setConfirmPassword] = useState("");

	const [companyLoading, setCompanyLoading] = useState(false);
	const [companySaving, setCompanySaving] = useState(false);
	const [companyError, setCompanyError] = useState<string | null>(null);
	const [companyNotice, setCompanyNotice] = useState<string | null>(null);
	const [companyId, setCompanyId] = useState<number | undefined>(undefined);
	const [companyName, setCompanyName] = useState("");
	const [companyPosition, setCompanyPosition] = useState("");

	const isExpired = useMemo(() => {
		if (!userId) return false;
		return is_password_expired(userId);
	}, [userId, saving, loading]);

	const expiryDays = useMemo(() => {
		if (!userId) return null;
		return days_until_expiry(userId);
	}, [userId, saving, loading]);

	useEffect(() => {
		safeSetLocalStorage(PROFILE_TAB_STORAGE_KEY, activeTab);
	}, [activeTab]);

	useEffect(() => {
		if (!userId) return;

		let ignore = false;
		setLoading(true);
		setError(null);

		getUserProfile(userId)
			.then((res) => {
				if (ignore) return;
				setName(res.data.name || "");
				setEmail(res.data.email || "");
				setRole(typeof res.data.role === "number" ? res.data.role : 0);
				localStorage.setItem("email", res.data.email || "");
				localStorage.setItem("name", res.data.name || "");
				setLoading(false);
			})
			.catch((e: any) => {
				if (ignore) return;
				setError(e?.message || "프로필 정보를 불러오지 못했습니다.");
				setLoading(false);
			});

		return () => {
			ignore = true;
		};
	}, [userId]);

	useEffect(() => {
		if (!userId) return;
		if (activeTab !== "company") return;

		let ignore = false;
		setCompanyLoading(true);
		setCompanyError(null);
		setCompanyNotice(null);

		getCompany(userId)
			.then((c) => {
				if (ignore) return;
				setCompanyId(c.companyId);
				setCompanyName(c.name || "");
				setCompanyPosition(c.performanceHistory || "");
			})
			.catch((e: any) => {
				if (ignore) return;
				setCompanyError(e?.message || "회사 정보를 불러오지 못했습니다.");
				setCompanyId(undefined);
				setCompanyName("");
				setCompanyPosition("");
			})
			.finally(() => {
				if (ignore) return;
				setCompanyLoading(false);
			});

		return () => {
			ignore = true;
		};
	}, [activeTab, userId]);

	const tabTriggerClass = useMemo(() => {
		return [
			"px-4 py-2 rounded-md",
			"data-[state=active]:bg-black data-[state=active]:text-white",
			"data-[state=active]:shadow-sm",
			"data-[state=inactive]:bg-transparent data-[state=inactive]:text-muted-foreground",
			"hover:text-foreground",
		].join(" ");
	}, []);

	const onSaveProfile = async () => {
		setError(null);

		if (!userId) {
			setError("로그인이 필요합니다.");
			return;
		}

		if (!newPassword && !confirmPassword) {
			setError("변경할 비밀번호를 입력해 주세요.");
			return;
		}

		if (newPassword !== confirmPassword) {
			setError("새 비밀번호와 확인이 일치하지 않습니다.");
			return;
		}

		const payload: { email: string; password: string; name: string; role: number } = {
			email: email.trim(),
			name: name.trim(),
			role,
			password: newPassword,
		};

		try {
			setSaving(true);
			await updateUserProfile(userId, payload);
			set_password_changed_now_for_user(userId);

			if (passwordExpiredFromRoute && fromAfterChange) {
				navigate(fromAfterChange, { replace: true });
			}

			setCurrentPassword("");
			setNewPassword("");
			setConfirmPassword("");
		} catch (e: any) {
			setError(e?.message || "저장에 실패했습니다.");
		} finally {
			setSaving(false);
		}
	};

	const onSaveCompany = async () => {
		setCompanyError(null);
		setCompanyNotice(null);

		if (!userId) {
			setCompanyError("로그인이 필요합니다.");
			return;
		}

		if (!companyName.trim()) {
			setCompanyError("회사명을 입력해 주세요.");
			return;
		}

		try {
			setCompanySaving(true);

			const payload = {
				id: companyId,
				name: companyName.trim(),
				license: "",
				performanceHistory: companyPosition.trim(),
			};

			const res = await upsertCompany(payload);

			if (res.status === "success") {
				setCompanyNotice("회사 정보가 저장되었습니다.");
			} else {
				setCompanyError(res.message || "회사 정보 저장에 실패했습니다.");
			}
		} catch (e: any) {
			setCompanyError(e?.message || "회사 정보 저장에 실패했습니다.");
		} finally {
			setCompanySaving(false);
		}
	};

	return (
		<div className="space-y-6">
			<div>
				<h2 className="text-3xl mb-2">마이페이지</h2>
				<p className="text-muted-foreground">계정 정보 및 설정을 관리하세요</p>
			</div>

			<Card>
				<CardContent className="pt-6">
					<div className="flex items-center gap-6">
						<Avatar className="h-20 w-20">
							<AvatarFallback className="bg-blue-600 text-white text-2xl">
								{avatarLetter(email)}
							</AvatarFallback>
						</Avatar>

						<div className="flex-1">
							<h3 className="text-2xl font-bold mb-1">{name || "—"}</h3>
							<p className="text-muted-foreground mb-2">{email || "이메일 정보 없음"}</p>
							<div className="flex gap-2 flex-wrap">
								<Badge variant="outline">{loading ? "불러오는 중..." : "계정"}</Badge>
								{userId ? (
									<Badge variant="outline">비밀번호 정책: {ACCESS_CONTROL.PASSWORD_MAX_AGE_DAYS}일</Badge>
								) : null}
								{userId && expiryDays !== null && !isExpired ? (
									<Badge variant="outline">만료까지 {expiryDays}일</Badge>
								) : null}
								{userId && isExpired ? <Badge variant="destructive">비밀번호 만료</Badge> : null}
							</div>
						</div>
					</div>
				</CardContent>
			</Card>

			<Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as ProfileTab)} className="space-y-4">
				<TabsList className="bg-transparent p-0 gap-2">
					<TabsTrigger value="info" className={tabTriggerClass}>
						계정 정보
					</TabsTrigger>
					<TabsTrigger value="company" className={tabTriggerClass}>
						회사 정보
					</TabsTrigger>
					<TabsTrigger value="notifications" className={tabTriggerClass}>
						알림 설정
					</TabsTrigger>
					<TabsTrigger value="subscription" className={tabTriggerClass}>
						구독 관리
					</TabsTrigger>
				</TabsList>

				<TabsContent value="info" className="space-y-4">
					<Card>
						<CardHeader>
							<CardTitle>개인 정보</CardTitle>
							<CardDescription>계정의 기본 정보를 관리합니다</CardDescription>
						</CardHeader>
						<CardContent className="space-y-4">
							{(passwordExpiredFromRoute || isExpired) ? (
								<div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
									비밀번호 유효기간이 만료되었습니다. 안전을 위해 비밀번호를 변경해 주세요.
								</div>
							) : null}

							{(!isExpired && expiryDays !== null && expiryDays <= 7) ? (
								<div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
									비밀번호 유효기간이 곧 만료됩니다. ({expiryDays}일 남음) 미리 변경을 권장합니다.
								</div>
							) : null}

							{error ? (
								<div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
									{error}
								</div>
							) : null}

							<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
								<div className="space-y-2">
									<Label>이름</Label>
									<div className="h-10 rounded-md border px-3 flex items-center text-sm">{name || "—"}</div>
								</div>

								<div className="space-y-2">
									<Label>이메일</Label>
									<div className="h-10 rounded-md border px-3 flex items-center text-sm">{email || "—"}</div>
								</div>
							</div>

							<Separator />

							<div className="space-y-4">
								<h4 className="font-semibold">비밀번호 변경</h4>

								<div className="space-y-2">
									<Label htmlFor="currentPassword">현재 비밀번호</Label>
									<Input
										id="currentPassword"
										type="password"
										value={currentPassword}
										onChange={(e) => setCurrentPassword(e.target.value)}
									/>
								</div>

								<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
									<div className="space-y-2">
										<Label htmlFor="newPassword">새 비밀번호</Label>
										<Input
											id="newPassword"
											type="password"
											value={newPassword}
											onChange={(e) => setNewPassword(e.target.value)}
										/>
									</div>
									<div className="space-y-2">
										<Label htmlFor="confirmPassword">비밀번호 확인</Label>
										<Input
											id="confirmPassword"
											type="password"
											value={confirmPassword}
											onChange={(e) => setConfirmPassword(e.target.value)}
										/>
									</div>
								</div>
							</div>

							<div className="flex justify-end">
								<Button onClick={onSaveProfile} disabled={saving || loading}>
									{saving ? "저장 중..." : "변경사항 저장"}
								</Button>
							</div>
						</CardContent>
					</Card>
				</TabsContent>

				<TabsContent value="company" className="space-y-4">
					<Card>
						<CardHeader>
							<CardTitle>회사 정보</CardTitle>
							<CardDescription>회사명과 직위를 저장합니다</CardDescription>
						</CardHeader>
						<CardContent className="space-y-4">
							{companyError ? (
								<div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
									{companyError}
								</div>
							) : null}

							{companyNotice ? (
								<div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800">
									{companyNotice}
								</div>
							) : null}

							<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
								<div className="space-y-2">
									<Label htmlFor="companyName">회사명</Label>
									<Input
										id="companyName"
										value={companyName}
										onChange={(e) => setCompanyName(e.target.value)}
										placeholder={companyLoading ? "불러오는 중..." : "회사명을 입력하세요"}
										disabled={companyLoading}
									/>
								</div>

								<div className="space-y-2">
									<Label htmlFor="companyPosition">직위</Label>
									<Input
										id="companyPosition"
										value={companyPosition}
										onChange={(e) => setCompanyPosition(e.target.value)}
										placeholder={companyLoading ? "불러오는 중..." : "예: 입찰 담당자"}
										disabled={companyLoading}
									/>
								</div>
							</div>

							<div className="flex justify-end">
								<Button onClick={onSaveCompany} disabled={companySaving || companyLoading}>
									{companySaving ? "저장 중..." : "회사 정보 저장"}
								</Button>
							</div>
						</CardContent>
					</Card>
				</TabsContent>

				<TabsContent value="notifications" className="space-y-4">
					<Card>
						<CardHeader>
							<CardTitle>알림 설정</CardTitle>
							<CardDescription>준비 중입니다</CardDescription>
						</CardHeader>
						<CardContent className="text-sm text-muted-foreground">곧 제공됩니다.</CardContent>
					</Card>
				</TabsContent>

				<TabsContent value="subscription" className="space-y-4">
					<Card>
						<CardHeader>
							<CardTitle>구독 관리</CardTitle>
							<CardDescription>준비 중입니다</CardDescription>
						</CardHeader>
						<CardContent className="text-sm text-muted-foreground">곧 제공됩니다.</CardContent>
					</Card>
				</TabsContent>
			</Tabs>
		</div>
	);
}
