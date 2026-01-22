import { Navigate, useLocation } from "react-router-dom";

import {
	ensure_password_changed_at_initialized,
	is_password_expired,
} from "../utils/accessControl";

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
	const userId = localStorage.getItem("userId");
	const location = useLocation();

	if (!userId) {
		return <Navigate to="/login" replace state={{ from: location.pathname }} />;
	}

	ensure_password_changed_at_initialized(userId);

	// 비밀번호 만료 시: 프로필 페이지에서 비밀번호 변경을 유도
	if (is_password_expired(userId) && location.pathname !== "/profile") {
		return (
			<Navigate
				to="/profile"
				replace
				state={{ passwordExpired: true, fromAfterChange: location.pathname }}
			/>
		);
	}

	return <>{children}</>;
}
