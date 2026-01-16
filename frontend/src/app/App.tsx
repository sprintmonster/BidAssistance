import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppLayout } from "./layout/AppLayout";
import { ProtectedRoute } from "./routes/ProtectedRoute";
import { PublicRoute } from "./routes/PublicRoute";

import { Dashboard } from "./components/dashboard/Dashboard";
import { LoginPage } from "./components/LoginPage";
import { HomePage } from "./components/HomePage";
import { BidDiscovery } from "./components/BidDiscovery";
import { CartPage } from "./components/CartPage";
import { CommunityPage } from "./components/CommunityPage";
import { ChatbotPage } from "./components/ChatbotPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          {/* Public */}
          <Route path="/" element={<HomePage />} />

          <Route element={<PublicRoute />}>
            <Route path="/login" element={<LoginPage />} />
          </Route>

          {/* Protected */}
          <Route element={<ProtectedRoute />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/bids" element={<BidDiscovery />} />
            <Route path="/cart" element={<CartPage />} />
            <Route path="/community" element={<CommunityPage />} />
            <Route path="/chatbot" element={<ChatbotPage />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

