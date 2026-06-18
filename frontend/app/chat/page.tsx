"use client";

import * as React from "react";
import { MessageSquare, Plus, Trash2 } from "lucide-react";

import { ChatPanel } from "@/components/chat-panel";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/lib/types";

interface Session {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
}

function uuid(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function newSession(): Session {
  return {
    id: uuid(),
    title: "New chat",
    messages: [],
    createdAt: Date.now(),
  };
}

function titleFromMessages(msgs: ChatMessage[]): string {
  const first = msgs.find((m) => m.role === "user")?.content ?? "";
  if (!first) return "New chat";
  return first.length > 40 ? first.slice(0, 40) + "…" : first;
}

const STORAGE_KEY = "vllm-chat-sessions";

function loadSessions(): Session[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as Session[];
  } catch {
    return [];
  }
}

function saveSessions(sessions: Session[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch (e) {
    if (e instanceof DOMException) return;
    throw e;
  }
}

export default function ChatPage() {
  const [sessions, setSessions] = React.useState<Session[]>([]);
  const [activeId, setActiveId] = React.useState<string>("");
  const [hydrated, setHydrated] = React.useState(false);

  React.useEffect(() => {
    queueMicrotask(() => {
      const s = loadSessions();
      const initial = s.length > 0 ? s : [newSession()];
      setSessions(initial);
      setActiveId(initial[0].id);
      setHydrated(true);
    });
  }, []);

  const activeSession = sessions.find((s) => s.id === activeId) ?? sessions[0];

  React.useEffect(() => {
    if (hydrated) saveSessions(sessions);
  }, [sessions, hydrated]);

  function createSession() {
    const s = newSession();
    setSessions((prev) => [s, ...prev]);
    setActiveId(s.id);
  }

  function deleteSession(id: string) {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      if (next.length === 0) {
        const s = newSession();
        setActiveId(s.id);
        return [s];
      }
      if (id === activeId) setActiveId(next[0].id);
      return next;
    });
  }

  function updateMessages(messages: ChatMessage[]) {
    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeId
          ? { ...s, messages, title: titleFromMessages(messages) }
          : s
      )
    );
  }

  if (!hydrated) return null;

  return (
    <div className="flex h-full min-h-0 flex-1 overflow-hidden">
      <aside className="flex min-h-0 w-52 shrink-0 flex-col overflow-hidden border-r bg-sidebar">
        <div className="flex h-12 shrink-0 items-center justify-between px-3">
          <span className="text-xs font-semibold text-sidebar-foreground/70 uppercase tracking-wider">
            Sessions
          </span>
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={createSession}
            title="New chat"
            className="size-6"
          >
            <Plus className="size-3.5" />
          </Button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={cn(
                "group flex cursor-pointer items-center gap-1 rounded-lg px-2 py-1.5 text-xs",
                s.id === activeId
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground",
              )}
              onClick={() => setActiveId(s.id)}
            >
              <MessageSquare className="size-3 shrink-0 opacity-60" />
              <span className="min-w-0 flex-1 truncate">{s.title}</span>
              <button
                type="button"
                className="hidden shrink-0 rounded p-0.5 opacity-60 hover:opacity-100 group-hover:block"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteSession(s.id);
                }}
                title="Delete session"
              >
                <Trash2 className="size-3" />
              </button>
            </div>
          ))}
        </div>
      </aside>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b px-4">
          <MessageSquare className="size-4 text-primary" />
          <h1 className="truncate text-sm font-medium text-foreground">
            {activeSession?.title ?? "Chat"}
          </h1>
        </header>
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {activeSession && (
            <ChatPanel
              key={activeId}
              initialMessages={activeSession.messages}
              onMessagesChange={updateMessages}
            />
          )}
        </div>
      </div>
    </div>
  );
}
