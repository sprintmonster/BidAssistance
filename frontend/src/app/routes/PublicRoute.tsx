import { Navigate, Outlet } from "react-router-dom";

export function PublicRoute() {
  const isAuthenticated = !!localStorage.getItem("accessToken");

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return <Outlet />;
}
