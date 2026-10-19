const BASE = `${(import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "")}/api`;

export interface Source {
  text: string;
  score: number;
  source: string;
  source_title: string;
  section: "narrative" | "commentary";
  parva: string;
  chapter: number;
  chapter_title: string;
  citation: string;
}

export interface ChatMeta {
  type: "meta";
  on_topic: boolean;
  entities: string[];
  graph_edges: { source: string; target: string; type: string; origins: string }[];
  sources: Source[];
}

export interface Health {
  status: string;
  neo4j: boolean;
  chunks: number;
  characters: number;
  relationships: number;
}

export interface StreamHandlers {
  onMeta: (meta: ChatMeta) => void;
  onToken: (token: string) => void;
  onError: (message: string) => void;
  onDone: () => void;
}

/** POST a question and dispatch the server's SSE events. Always ends with onDone. */
export async function streamChat(message: string, h: StreamHandlers) {
  let finished = false;
  try {
    const res = await fetch(`${BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!res.ok || !res.body) {
      h.onError(res.status === 422 ? "That question is too long." : `Server error (${res.status}).`);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        let data;
        try {
          data = JSON.parse(line.slice(6));
        } catch {
          continue;
        }
        if (data.type === "meta") h.onMeta(data);
        else if (data.type === "token") h.onToken(data.content);
        else if (data.type === "error") {
          h.onError(data.message);
          finished = true;
        } else if (data.type === "done") finished = true;
      }
    }
    if (!finished) h.onError("The connection closed before the answer finished.");
  } catch {
    h.onError("Could not reach the server.");
  } finally {
    h.onDone();
  }
}

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`health ${res.status}`);
  return res.json();
}
