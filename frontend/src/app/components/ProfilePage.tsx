import { useEffect, useMemo, useState } from "react";
import { getUserProfile, updateUserProfile } from "../api/users";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Switch } from "./ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Separator } from "./ui/separator";
import { Badge } from "./ui/badge";

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

export function ProfilePage({ userEmail }: ProfilePageProps) {
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
				setName(res.data.name || "");      // API: name
				setEmail(res.data.email || "");    // API: email
				setRole(typeof res.data.role === "number" ? res.data.role : 0); // API: role
				localStorage.setItem("email", res.data.email || "");
				localStorage.setItem("name", res.data.name || "");
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
		if (!name.trim()) {
			setError("이름을 입력해 주세요.");
			return;
		}
		if (!email.trim()) {
			setError("이메일을 입력해 주세요.");
			return;
		}
		if (newPassword || confirmPassword) {
			if (newPassword !== confirmPassword) {
				setError("새 비밀번호와 확인이 일치하지 않습니다.");
				return;
			}
		}

		const payload: { email: string; name: string; role: number; password?: string } = {
			email: email.trim(),
			name: name.trim(),
			role,
		};

		// 정의서 수정 API가 password를 받으므로, 변경할 때만 포함
		if (newPassword) payload.password = newPassword;

		try {
			setSaving(true);
			await updateUserProfile(userId, payload);
			localStorage.setItem("email", email.trim());
			localStorage.setItem("name", name.trim());
			setCurrentPassword("");
			setNewPassword("");
			setConfirmPassword("");
		} catch (e: any) {
			setError(e?.message || "저장에 실패했습니다.");
		} finally {
			setSaving(false);
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
							<div className="flex gap-2">
								<Badge variant="outline">{loading ? "불러오는 중..." : "계정"}</Badge>
							</div>
						</div>
					</div>
				</CardContent>
			</Card>

			<Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as ProfileTab)} className="space-y-4">
				<TabsList className="bg-transparent p-0 gap-2">
					<TabsTrigger value="info" className={tabTriggerClass}>계정 정보</TabsTrigger>
					<TabsTrigger value="company" className={tabTriggerClass}>회사 정보</TabsTrigger>
					<TabsTrigger value="notifications" className={tabTriggerClass}>알림 설정</TabsTrigger>
					<TabsTrigger value="subscription" className={tabTriggerClass}>구독 관리</TabsTrigger>
				</TabsList>

				<TabsContent value="info" className="space-y-4">
					<Card>
						<CardHeader>
							<CardTitle>개인 정보</CardTitle>
							<CardDescription>계정의 기본 정보를 관리합니다</CardDescription>
						</CardHeader>
						<CardContent className="space-y-4">
							{error && (
								<div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
									{error}
								</div>
							)}

							<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
								<div className="space-y-2">
									<Label htmlFor="name">이름</Label>
									<Input id="name" value={name} onChange={(e) => setName(e.target.value)} />
								</div>
								<div className="space-y-2">
									<Label htmlFor="email">이메일</Label>
									<Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} />
								</div>

								{/* 아래 phone/position은 정의서에 필드가 없어 “UI only” */}
								<div className="space-y-2">
									<Label htmlFor="phone">연락처</Label>
									<Input id="phone" defaultValue="010-1234-5678" />
								</div>
								<div className="space-y-2">
									<Label htmlFor="position">직위</Label>
									<Input id="position" defaultValue="입찰 담당자" />
								</div>
							</div>

							<Separator />

							<div className="space-y-4">
								<h4 className="font-semibold">비밀번호 변경</h4>

								{/* 현재 비밀번호 검증 API가 정의서에 없어서 일단 UI만 유지 */}
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

				{/* company/notifications/subscription 탭은 기존 UI 그대로 두되,
				    현재 정의서엔 저장용 API가 없으니(추후 연동) */}
				<TabsContent value="company" className="space-y-4">{/* 기존 코드 유지 */}</TabsContent>
				<TabsContent value="notifications" className="space-y-4">{/* 기존 코드 유지 */}</TabsContent>
				<TabsContent value="subscription" className="space-y-4">{/* 기존 코드 유지 */}</TabsContent>
			</Tabs>
		</div>
	);
}
