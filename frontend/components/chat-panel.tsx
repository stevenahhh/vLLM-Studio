"use client";

import * as React from "react";
import { ChevronDown, ChevronRight, Copy, Check, Loader2, Send } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import type { Components } from "react-markdown";

import * as api from "@/lib/api";
import { useStudio } from "@/lib/store";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

// ---------------------------------------------------------------------------
// Markdown renderer
// ---------------------------------------------------------------------------

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = React.useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        });
      }}
      className="absolute right-2 top-2 rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:text-foreground group-hover:opacity-100"
      title="Copy code"
    >
      {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
    </button>
  );
}

const MD_COMPONENTS: Components = {
  // Headings
  h1: ({ children }) => (
    <h1 className="mb-3 mt-5 text-xl font-bold leading-tight first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="mb-2 mt-4 text-lg font-semibold leading-tight first:mt-0">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="mb-2 mt-3 text-base font-semibold leading-tight first:mt-0">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="mb-1 mt-2 text-sm font-semibold first:mt-0">{children}</h4>
  ),
  // Paragraphs
  p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
  // Lists
  ul: ({ children }) => <ul className="mb-2 ml-5 list-disc space-y-0.5 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 ml-5 list-decimal space-y-0.5 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  // Blockquote
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-4 border-border pl-4 italic text-muted-foreground">
      {children}
    </blockquote>
  ),
  // Inline code
  code: ({ children, className, ...props }) => {
    const isBlock = className?.startsWith("language-");
    if (isBlock) {
      const text = String(children).replace(/\n$/, "");
      const lang = className?.replace("language-", "") ?? "";
      return (
        <div className="group relative my-3 overflow-hidden rounded-lg border border-border bg-muted/50">
          {lang && (
            <div className="flex items-center justify-between border-b border-border bg-muted/80 px-3 py-1.5">
              <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                {lang}
              </span>
            </div>
          )}
          <CopyButton text={text} />
          <pre className="overflow-x-auto p-4 text-xs leading-relaxed">
            <code className={className} {...props}>
              {children}
            </code>
          </pre>
        </div>
      );
    }
    return (
      <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.8em]" {...props}>
        {children}
      </code>
    );
  },
  pre: ({ children }) => <>{children}</>,
  // Horizontal rule
  hr: () => <hr className="my-4 border-border" />,
  // Links
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-primary underline underline-offset-2 hover:opacity-80"
    >
      {children}
    </a>
  ),
  // Tables
  table: ({ children }) => (
    <div className="my-3 overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-muted/60">{children}</thead>,
  tbody: ({ children }) => <tbody className="divide-y divide-border">{children}</tbody>,
  tr: ({ children }) => <tr className="divide-x divide-border">{children}</tr>,
  th: ({ children }) => (
    <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
      {children}
    </th>
  ),
  td: ({ children }) => <td className="px-3 py-2 leading-relaxed">{children}</td>,
  // Strong / em
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
};

function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={MD_COMPONENTS}
    >
      {children}
    </ReactMarkdown>
  );
}

// ---------------------------------------------------------------------------
// Think block parsing
// ---------------------------------------------------------------------------

function parseThinkBlocks(content: string): Array<{ type: "text" | "think"; text: string }> {
  const parts: Array<{ type: "text" | "think"; text: string }> = [];
  const re = /<think>([\s\S]*?)<\/think>/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(content)) !== null) {
    if (m.index > last) parts.push({ type: "text", text: content.slice(last, m.index) });
    parts.push({ type: "think", text: m[1] });
    last = m.index + m[0].length;
  }
  const tail = content.slice(last);
  const openIdx = tail.indexOf("<think>");
  if (openIdx !== -1) {
    if (openIdx > 0) parts.push({ type: "text", text: tail.slice(0, openIdx) });
    parts.push({ type: "think", text: tail.slice(openIdx + 7) });
  } else if (tail) {
    parts.push({ type: "text", text: tail });
  }
  return parts;
}

function ThinkBlock({ text }: { text: string }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="my-1.5 rounded-md border border-border/60 bg-muted/30 text-xs">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-muted-foreground hover:text-foreground"
      >
        {open ? <ChevronDown className="size-3 shrink-0" /> : <ChevronRight className="size-3 shrink-0" />}
        <span className="font-medium italic">Thinking…</span>
      </button>
      {open && (
        <div className="border-t border-border/60 px-3 py-2 text-muted-foreground">
          <Markdown>{text}</Markdown>
        </div>
      )}
    </div>
  );
}

function AssistantContent({ content, streaming }: { content: string; streaming: boolean }) {
  if (!content) return <span className="text-muted-foreground">{streaming ? "..." : ""}</span>;
  const parts = parseThinkBlocks(content);
  return (
    <>
      {parts.map((p, i) =>
        p.type === "think" ? (
          <ThinkBlock key={i} text={p.text} />
        ) : (
          <Markdown key={i}>{p.text}</Markdown>
        )
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Context pie
// ---------------------------------------------------------------------------

// rough token estimate: chars / 3.5
function estimateTokens(messages: ChatMessage[], input: string): number {
  const allText = messages.map((m) => m.content).join("") + input;
  return Math.round(allText.length / 3.5);
}

function ContextPie({
  used,
  total,
}: {
  used: number;
  total: number;
}) {
  const R = 9;
  const C = 2 * Math.PI * R;
  const ratio = total > 0 ? Math.min(used / total, 1) : 0;
  const dash = ratio * C;
  const pct = Math.round(ratio * 100);

  const color =
    pct >= 90
      ? "text-destructive stroke-destructive"
      : pct >= 70
      ? "text-amber-500 stroke-amber-500"
      : "text-muted-foreground stroke-muted-foreground";

  const fmtK = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(0)}k` : String(n));

  return (
    <div
      className={cn("group relative flex shrink-0 cursor-default items-center justify-center", color)}
      title={`Context: ${fmtK(used)} / ${fmtK(total)} tokens (~${pct}%)`}
    >
      <svg width="22" height="22" viewBox="0 0 22 22" className="-rotate-90">
        {/* track */}
        <circle
          cx="11" cy="11" r={R}
          fill="none"
          strokeWidth="2.5"
          className="stroke-border"
        />
        {/* fill */}
        <circle
          cx="11" cy="11" r={R}
          fill="none"
          strokeWidth="2.5"
          strokeDasharray={`${dash} ${C}`}
          strokeLinecap="round"
          className={cn("transition-all duration-300", color)}
        />
      </svg>
      {/* tooltip on hover */}
      <div className="pointer-events-none absolute bottom-full left-1/2 mb-2 hidden -translate-x-1/2 whitespace-nowrap rounded-md border border-border bg-popover px-2 py-1 text-[10px] text-popover-foreground shadow-md group-hover:block">
        {fmtK(used)} / {fmtK(total)} tokens
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChatPanel
// ---------------------------------------------------------------------------

export function ChatPanel({
  initialMessages = [],
  onMessagesChange,
}: {
  initialMessages?: ChatMessage[];
  onMessagesChange?: (messages: ChatMessage[]) => void;
}) {
  const settings = useStudio((s) => s.settings);
  const engine = useStudio((s) => s.engine);
  const [messages, setMessages] = React.useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = React.useState("");
  const [streaming, setStreaming] = React.useState(false);
  const abortRef = React.useRef<AbortController | null>(null);
  const bottomRef = React.useRef<HTMLDivElement | null>(null);
  const scrollRef = React.useRef<HTMLDivElement | null>(null);

  const canSend = input.trim().length > 0 && !streaming;

  const maxCtx = engine?.load_request?.max_model_len ?? 0;
  const usedCtx = estimateTokens(messages, input);

  function updateMessages(next: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) {
    setMessages((prev) => {
      const resolved = typeof next === "function" ? next(prev) : next;
      onMessagesChange?.(resolved);
      return resolved;
    });
  }

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    const content = input.trim();
    if (!content || streaming) return;

    const nextMessages: ChatMessage[] = [
      ...messages,
      { role: "user", content },
      { role: "assistant", content: "" },
    ];
    updateMessages(nextMessages);
    setInput("");
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await api.streamChat(
        {
          messages: nextMessages.slice(0, -1),
          system_prompt: settings?.system_prompt,
          sampling: settings?.sampling,
          extra_body: settings?.diffusion,
        },
        (chunk) => {
          if (chunk.done && !chunk.delta) return;
          updateMessages((current) => {
            const copy = [...current];
            const last = copy[copy.length - 1];
            if (!last || last.role !== "assistant") return current;
            copy[copy.length - 1] = {
              ...last,
              content: last.content + chunk.delta,
            };
            return copy;
          });
        },
        controller.signal,
      );
    } catch (e) {
      if (!controller.signal.aborted) {
        const detail = e instanceof Error ? e.message : String(e);
        updateMessages((current) => {
          const copy = [...current];
          const last = copy[copy.length - 1];
          if (!last || last.role !== "assistant") return current;
          copy[copy.length - 1] = { ...last, content: `[error: ${detail}]` };
          return copy;
        });
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setStreaming(false);
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto p-4">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.length === 0 ? (
            <div className="rounded-2xl border border-dashed p-6 text-sm text-muted-foreground">
              {engine?.state === "ready"
                ? "Ask the loaded model a question."
                : "Load a model before chatting."}
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={index}
                className={cn(
                  "rounded-2xl px-4 py-3 text-sm",
                  message.role === "user"
                    ? "ml-auto max-w-[80%] bg-primary text-primary-foreground"
                    : "mr-auto max-w-[80%] bg-card text-card-foreground ring-1 ring-border",
                )}
              >
                {message.role === "assistant" ? (
                  <AssistantContent content={message.content} streaming={streaming && index === messages.length - 1} />
                ) : (
                  <Markdown>{message.content}</Markdown>
                )}
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="shrink-0 border-t p-4">
        <form
          className="mx-auto flex max-w-3xl items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void sendMessage();
          }}
        >
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Message the loaded model..."
            className="min-h-10 flex-1"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void sendMessage();
              }
            }}
          />
          {maxCtx > 0 && (
            <ContextPie used={usedCtx} total={maxCtx} />
          )}
          <Button type="submit" disabled={!canSend} size="icon" className="shrink-0">
            {streaming ? <Loader2 className="animate-spin" /> : <Send />}
          </Button>
        </form>
      </div>
    </div>
  );
}
