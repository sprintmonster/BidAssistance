import { Outlet } from "react-router-dom";

export function AppLayout() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="h-14 border-b flex items-center px-6 font-semibold">
        입찰인사이트
      </header>

      {/* Page Content */}
      <main className="flex-1">
        <Outlet />
      </main>
    </div>
  );
}
