const BASE = "/api";

export interface ChatMeta {
  type: "meta";
  entities: string[];
  graph_edges: { source: string; target: string; type: string }[];
  sources: { parva: string; chapter: number; chapter_title: string; score: number; text: string }[];
}

export interface GraphCharacter {
  name: string;
  relationships: {
    source: string;
    type: string;
    target: string;
    direction: string;
    origins: string;
  }[];
}

export interface SearchResult {
  text: string;
  parva: string;
  chapter: number;
  chapter_title: string;
  score: number;
}

export interface GraphStats {
  nodes: Record<string, number>;
  relationships: Record<string, number>;
  top_characters: { name: string; connections: number }[];
}

export async function streamChat(
  message: string,
  onMeta: (meta: ChatMeta) => void,
  onToken: (token: string) => void,
  onDone: () => void
) {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  const reader = res.body?.getReader();
  if (!reader) return;

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
      const json = line.slice(6);
      try {
        const data = JSON.parse(json);
        if (data.type === "meta") onMeta(data);
        else if (data.type === "token") onToken(data.content);
        else if (data.type === "done") onDone();
      } catch {}
    }
  }
}

export async function searchText(query: string, topK = 5): Promise<SearchResult[]> {
  const res = await fetch(`${BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  const data = await res.json();
  return data.results;
}

export async function getCharacter(name: string): Promise<GraphCharacter> {
  const res = await fetch(`${BASE}/graph/character/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`Character not found: ${name}`);
  return res.json();
}

export async function getGraphStats(): Promise<GraphStats> {
  const res = await fetch(`${BASE}/graph/stats`);
  return res.json();
}

export async function getEntities(): Promise<string[]> {
  const res = await fetch(`${BASE}/entities`);
  const data = await res.json();
  return data.entities;
}