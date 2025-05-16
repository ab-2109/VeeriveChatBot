"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();
  
  useEffect(() => {
    // Redirect to the chat page
    window.location.href = "/chat/chat.html";
  }, []);

  return (
    <div className="min-h-screen flex items-center justify-center">
      <p>Redirecting to chat...</p>
    </div>
  );
}
