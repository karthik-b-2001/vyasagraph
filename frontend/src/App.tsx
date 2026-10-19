import { useState, useRef, useEffect } from "react";
import Architecture from "./Architecture";
import { streamChat, getHealth, type ChatMeta, type Health } from "./lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  meta?: ChatMeta;
  error?: string;
}

const SUGGESTIONS = [
  "Who were Arjuna's parents?",
  "How did Karna die?",
  "Who killed Duryodhana?",
  "What happened at the dice game?",
  "What was Abhimanyu's fate?",
  "Who is Vrikodara?",
];

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  // "slow" = no answer yet after 2 s: the free backend is probably waking up.
  const [server, setServer] = useState<"connecting" | "slow" | "ok" | "down">("connecting");
  const [tab, setTab] = useState<"chat" | "architecture">("chat");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Calling /api/health on page load also wakes the backend (it scales to
    // zero when idle), so it is usually ready by the time a question is sent.
    let cancelled = false;
    const slow = setTimeout(() => !cancelled && setServer((s) => (s === "connecting" ? "slow" : s)), 2000);
    const connect = async (attemptsLeft: number) => {
      try {
        const h = await getHealth();
        if (!cancelled) { setHealth(h); setServer("ok"); }
      } catch {
        if (cancelled) return;
        if (attemptsLeft > 0) setTimeout(() => connect(attemptsLeft - 1), 5000);
        else setServer("down");
      }
    };
    connect(2);
    return () => { cancelled = true; clearTimeout(slow); };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Apply a change to the assistant message being streamed (always the last one).
  const updateLast = (patch: (m: Message) => Partial<Message>) =>
    setMessages((prev) => {
      const copy = [...prev];
      copy[copy.length - 1] = { ...copy[copy.length - 1], ...patch(copy[copy.length - 1]) };
      return copy;
    });

  const send = async () => {
    const msg = input.trim();
    if (!msg || streaming) return;

    setInput("");
    setStreaming(true);
    setMessages((prev) => [...prev, { role: "user", content: msg }, { role: "assistant", content: "" }]);

    await streamChat(msg, {
      onMeta: (meta) => updateLast(() => ({ meta })),
      onToken: (token) => updateLast((m) => ({ content: m.content + token })),
      onError: (error) => updateLast(() => ({ error })),
      onDone: () => {
        setStreaming(false);
        setTimeout(() => inputRef.current?.focus(), 0);
      },
    });
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="h-screen flex flex-col bg-[#0c0a14] text-zinc-100">
      {/* Header */}
      <header className="flex items-center gap-3 px-6 py-4 border-b border-zinc-800">
        <div className="w-8 h-8 rounded-full bg-violet-600 flex items-center justify-center text-white font-bold text-sm">
          V
        </div>
        <div>
          <h1 className="text-lg font-semibold tracking-tight">VyasaGraph</h1>
          <p className="text-xs text-zinc-500">Mahabharata Knowledge Graph + RAG</p>
        </div>
        <nav className="ml-4 flex gap-1 text-sm">
          {(["chat", "architecture"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1.5 rounded-lg capitalize transition-colors ${
                tab === t ? "bg-zinc-800 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {t}
            </button>
          ))}
        </nav>
        {health && (
          <div className="ml-auto hidden lg:flex gap-2 text-xs text-zinc-600">
            <span className="px-2 py-1 rounded bg-zinc-900">{health.characters} characters</span>
            <span className="px-2 py-1 rounded bg-zinc-900">{health.relationships} relationships</span>
            <span className="px-2 py-1 rounded bg-zinc-900">{health.chunks.toLocaleString()} passages</span>
          </div>
        )}
      </header>

      {(server === "slow" || server === "down") && (
        <div className={`px-6 py-2 text-xs border-b border-zinc-800 ${server === "down" ? "text-red-400" : "text-violet-300 bg-violet-500/10"}`}>
          {server === "slow"
            ? "Waking up the server. The free host sleeps when idle, so the first load can take up to a minute."
            : "Can't reach the server right now. Please try again later."}
        </div>
      )}

      {tab === "architecture" && (
        <main className="flex-1 overflow-y-auto">
          <Architecture health={health} />
        </main>
      )}

      {/* Messages */}
      {tab === "chat" && (<>
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-20">
              <div className="w-16 h-16 rounded-full bg-violet-500/10 flex items-center justify-center mx-auto mb-4">
                <span className="text-3xl">🕉</span>
              </div>
              <h2 className="text-xl font-semibold text-zinc-300 mb-8">Ask about the Mahabharata</h2>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => { setInput(q); setTimeout(() => inputRef.current?.focus(), 0); }}
                    className="px-3 py-1.5 text-xs rounded-full border border-zinc-800 text-zinc-400 hover:border-violet-500/50 hover:text-violet-400 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-2xl rounded-2xl px-4 py-3 ${
                  msg.role === "user"
                    ? "bg-violet-500/15 text-violet-100"
                    : "bg-zinc-900 text-zinc-200"
                }`}
              >
                {/* Message content */}
                {msg.content && <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>}

                {/* Streaming cursor */}
                {msg.role === "assistant" && streaming && i === messages.length - 1 && !msg.error && (
                  <span className="inline-block w-2 h-4 bg-violet-500 animate-pulse ml-0.5" />
                )}

                {/* Error */}
                {msg.error && <p className="text-sm text-red-400 mt-1">{msg.error}</p>}

                {/* Meta: entities + graph facts + sources */}
                {msg.role === "assistant" && msg.meta?.on_topic && msg.content && (
                  <div className="mt-3 pt-3 border-t border-zinc-800 space-y-2">
                    {msg.meta.entities.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {msg.meta.entities.map((e) => (
                          <span key={e} className="px-2 py-0.5 text-xs rounded bg-violet-500/10 text-violet-400">
                            {e}
                          </span>
                        ))}
                      </div>
                    )}

                    {msg.meta.graph_edges.length > 0 && (
                      <details className="text-xs text-zinc-500">
                        <summary className="cursor-pointer hover:text-zinc-400">
                          {msg.meta.graph_edges.length} graph facts used
                        </summary>
                        <div className="mt-1 space-y-0.5 pl-2 border-l border-zinc-800">
                          {msg.meta.graph_edges.map((e, j) => (
                            <div key={j} className="font-mono">
                              {e.source} → <span className="text-zinc-600">{e.type}</span> → {e.target}
                              <span className="text-zinc-700"> ({e.origins})</span>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {msg.meta.sources.length > 0 && (
                      <details className="text-xs text-zinc-500">
                        <summary className="cursor-pointer hover:text-zinc-400">
                          {msg.meta.sources.length} text passages used
                        </summary>
                        <div className="mt-1 space-y-2 pl-2 border-l border-zinc-800">
                          {msg.meta.sources.map((s, j) => (
                            <div key={j}>
                              <div className="text-zinc-400">
                                {s.section === "commentary" && (
                                  <span className="mr-1 px-1 rounded bg-zinc-800 text-zinc-400">commentary</span>
                                )}
                                {s.citation} · {(s.score * 100).toFixed(0)}% match
                              </div>
                              <div className="text-zinc-600">{s.source_title}</div>
                              <div className="text-zinc-600 line-clamp-2">{s.text}</div>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </main>

      {/* Input */}
      <footer className="border-t border-zinc-800 px-4 py-4">
        <div className="max-w-3xl mx-auto flex gap-3">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about the Mahabharata..."
            maxLength={2000}
            disabled={streaming}
            className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-violet-500/50 disabled:opacity-50 transition-colors"
          />
          <button
            onClick={send}
            disabled={streaming || !input.trim()}
            className="px-5 py-3 rounded-xl bg-violet-600 text-white font-medium text-sm hover:bg-violet-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            {streaming ? "..." : "Ask"}
          </button>
        </div>
      </footer>
      </>)}
    </div>
  );
}

export default App;
