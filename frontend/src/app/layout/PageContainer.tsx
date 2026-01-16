import { ReactNode } from "react";

export function PageContainer({ children }: { children: ReactNode }) {
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {children}
    </div>
  );
}
