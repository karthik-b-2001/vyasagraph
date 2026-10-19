import { useEffect, useState } from "react";
import type { Health } from "./lib/api";

// ── Diagram data ──────────────────────────────────────────────────────────
// Every number here is measured (see README); live counts come from /api/health.

type FlowId = "query" | "ingest";

interface Node {
  id: string;
  label: string;
  sub: string;
  col: number;
  row: number; // 0 = top lane, 1 = middle, 2 = bottom lane
  title: string;
  body: string[];
  files: string[];
  stats?: (h: Health | null) => [string, string][];
  jump?: FlowId;
}

interface Flow {
  name: string;
  intro: string;
  nodes: Node[];
  edges: [string, string, string?][];
}

const live = (h: Health | null, k: keyof Health) => (h ? Number(h[k]).toLocaleString() : "…");

const FLOWS: Record<FlowId, Flow> = {
  query: {
    name: "Answering a question",
    intro: "What happens between pressing Ask and the streamed answer.",
    nodes: [
      {
        id: "question", label: "Your question", sub: "chat UI → /api/chat", col: 0, row: 1,
        title: "Your question",
        body: [
          "Typed in the chat tab and sent to the FastAPI backend.",
          "Input is trimmed; empty or whitespace-only messages are rejected (HTTP 422); 2,000 characters at most.",
          "The answer streams back over Server-Sent Events as it is generated.",
        ],
        files: ["frontend/src/App.tsx", "frontend/src/lib/api.ts", "src/api.py"],
      },
      {
        id: "entities", label: "Entity detection", sub: "names + epithets", col: 1, row: 0,
        title: "Entity detection",
        body: [
          "One whole-word, case-insensitive pattern over every character name in the graph plus 36 epithets (Partha → Arjuna, Vrikodara → Bhima, Radheya → Karna …).",
          "Longest match wins, so “Lakshmana Kumara” is not read as “Lakshmana”.",
          "Whole words only: “chariot” does not match the epithet “Hari”.",
        ],
        files: ["src/rag.py"],
        stats: (h) => [["characters known", live(h, "characters")]],
      },
      {
        id: "embed", label: "Query embedding", sub: "bge-small-en-v1.5", col: 1, row: 2,
        title: "Query embedding",
        body: [
          "The question, prefixed with BGE's retrieval instruction, becomes a 384-dimensional unit vector.",
          "Same model that embedded the passages; it runs inside the backend, no API key needed.",
        ],
        files: ["src/rag.py", "src/config.py"],
      },
      {
        id: "graph", label: "Graph facts", sub: "Neo4j", col: 2, row: 0,
        title: "Graph facts",
        body: [
          "For each character found, relationships in both directions are read from Neo4j.",
          "A budget of 40 facts is split evenly across the characters, ordered parents → spouses → killings → siblings.",
          "The UI shows exactly the facts the model was given, with their origins (wikidata / attested / csv).",
        ],
        files: ["src/rag.py"],
        stats: (h) => [["relationships in graph", live(h, "relationships")]],
      },
      {
        id: "search", label: "Passage search", sub: "exact cosine, quotas", col: 2, row: 2,
        title: "Passage search",
        body: [
          "One matrix product scores the question against every stored passage: exact search, not approximate.",
          "Quota: the 5 best passages from the Ganguli translation plus the 2 best from other sources, so the shorter modern retellings cannot crowd out the translation.",
          "This replaced ChromaDB, whose filtered approximate search returned the true top 5 for only 66% of test questions.",
        ],
        files: ["src/rag.py"],
        stats: (h) => [["passages searched", live(h, "chunks")], ["right section in top 5", "80% of 25 test questions"]],
      },
      {
        id: "gate", label: "Guardrail gate", sub: "before any LLM call", col: 3, row: 1,
        title: "Guardrail gate",
        body: [
          "A question is on-topic if it names a character, or if its best passage scores at least 0.55.",
          "Names with everyday meanings (Samba, Vajra, Kali, “hari” …) do not count on their own.",
          "Otherwise the fixed refusal is sent at once, without calling the LLM (~0.1 s).",
          "Deliberately loose: it blocked none of 40 real test questions and stops 18 of 40 off-topic ones; the prompt rule handles the rest.",
        ],
        files: ["src/rag.py", "scripts/eval_guardrail.py"],
        stats: () => [["real questions passed", "40 / 40"], ["off-topic stopped here", "18 / 40"]],
      },
      {
        id: "prompt", label: "Prompt", sub: "rules + context", col: 4, row: 0,
        title: "Prompt",
        body: [
          "Rules: explain the Mahabharata only; answer off-topic questions and task requests (poems, code) with one fixed refusal sentence.",
          "Family relations and killings may only come from the curated list; the model may not infer new ones.",
          "Each passage is labelled with its citation; commentary is marked as a modern author's interpretation.",
          "Cite only labels that were shown, and say so when the material does not cover the question.",
        ],
        files: ["src/rag.py"],
      },
      {
        id: "llm", label: "LLM", sub: "OpenAI-compatible", col: 5, row: 0,
        title: "Language model",
        body: [
          "Any /chat/completions endpoint, set by three environment variables: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.",
          "Default: llama3.1:8b served locally by Ollama, 7–23 s per answer on an Apple Silicon Mac.",
          "Known weakness of the 8B model: it sometimes cites a section it was not given.",
        ],
        files: ["src/rag.py", "src/config.py"],
      },
      {
        id: "stream", label: "Streamed answer", sub: "SSE → chat UI", col: 5, row: 2,
        title: "Streamed answer",
        body: [
          "Events: one meta (characters, graph facts, passages), then tokens, then done; or error if the LLM is unreachable, so the UI never hangs.",
          "Under each answer: the characters found, the graph facts with their origins, and the passages with citations such as “Karna Parva, Section 91”.",
        ],
        files: ["src/api.py", "frontend/src/App.tsx"],
      },
    ],
    edges: [
      ["question", "entities"],
      ["question", "embed"],
      ["entities", "graph"],
      ["embed", "search"],
      ["graph", "gate"],
      ["search", "gate"],
      ["gate", "prompt", "on-topic"],
      ["gate", "stream", "off-topic: refusal"],
      ["prompt", "llm"],
      ["llm", "stream", "tokens"],
    ],
  },
  ingest: {
    name: "Building the data",
    intro: "How text and curated facts become the vector index and the graph (python -m src.ingest).",
    nodes: [
      {
        id: "sources", label: "Text sources", sub: "data/sources/*/", col: 0, row: 0,
        title: "Text sources",
        body: [
          "Each folder holds a source.yaml (id, title, kind, parser, files) and its text.",
          "Ganguli: the complete English translation (1883–1896), Project Gutenberg #15474–#15477, public domain: all 18 parvas, 2,107 sections, ~2.4M words.",
          "Summary Study of the Mahabharata: a modern prose retelling with per-chapter commentary (157 sections). Its source file is missing every occurrence of the word “day”.",
          "To add material: drop in a folder with a Markdown file (# Parva / ## Chapter / ### Commentary) and re-run ingestion.",
        ],
        files: ["data/sources/ganguli/", "data/sources/summary_study/"],
      },
      {
        id: "parsers", label: "Parsers", sub: "one per layout", col: 1, row: 0,
        title: "Parsers",
        body: [
          "ganguli: Gutenberg bodies only; roman-numeral SECTION headers, or bare numbers in books 8–11 and 16–18; sub-parva names; drops prefaces, footnotes and editor notes.",
          "summary_study: separates narrative from commentary, and handles the Bhagavad-gita nested inside Bhishma Parva and the file's header typos.",
          "markdown: the format for adding your own notes.",
        ],
        files: ["src/ingest/parsers.py"],
      },
      {
        id: "chunker", label: "Chunker", sub: "≤250 word-pieces", col: 2, row: 0,
        title: "Chunker",
        body: [
          "Sentence-aligned chunks of at most 250 word-pieces with a 40-token overlap, counted with the embedding model's own tokenizer.",
          "0 of 18,861 chunks exceed the model's input limit. Before the rework, 94% of chunks were longer than the old model could read.",
        ],
        files: ["src/ingest/chunker.py"],
      },
      {
        id: "ids", label: "Content-hash IDs", sub: "incremental", col: 3, row: 0,
        title: "Content-hash IDs",
        body: [
          "Each chunk's ID is a hash of the model, its metadata and its text.",
          "A re-run reuses vectors of unchanged chunks, embeds only new ones, and drops removed ones, including every chunk of a deleted source folder.",
          "A re-run with nothing changed takes about 7 s.",
        ],
        files: ["src/ingest/text.py"],
      },
      {
        id: "embedder", label: "Embedder", sub: "bge-small-en-v1.5", col: 4, row: 0,
        title: "Embedder",
        body: [
          "33M-parameter model producing 384-dimensional normalized vectors.",
          "Chosen by benchmark: the right Ganguli section is in the top 5 for 80% of 25 test questions, against 60% for all-MiniLM-L6-v2.",
          "Embedding the whole corpus takes about 2 minutes on an Apple Silicon Mac.",
        ],
        files: ["src/config.py", "scripts/eval_retrieval.py"],
      },
      {
        id: "index", label: "Vector index", sub: "npz + jsonl, 55 MB", col: 5, row: 0,
        title: "Vector index",
        body: [
          "data/index/embeddings.npz (vectors and chunk IDs) and chunks.jsonl (text and citation metadata).",
          "Each file is written under a temporary name and renamed; loading checks that both list the same chunk IDs.",
          "16,948 chunks from Ganguli and 1,913 from the summary.",
        ],
        files: ["src/ingest/text.py"],
        stats: (h) => [["chunks", live(h, "chunks")]],
      },
      {
        id: "extract", label: "spaCy extractor", sub: "offline, reviewed", col: 1, row: 2,
        title: "spaCy extractor",
        body: [
          "scripts/extract_attested.py parses every narrative sentence with spaCy: epithets (“X, son of Y”) and kill statements (subject–verb–object).",
          "Each relationship is scored by how many sentences attest it; the results are then reviewed by hand.",
        ],
        files: ["scripts/extract_attested.py"],
      },
      {
        id: "csvs", label: "Curated CSVs", sub: "data/graph/", col: 2, row: 2,
        title: "Curated CSVs",
        body: [
          "Wikidata exports: parents, spouses, siblings, children, killers, conflicts.",
          "reviewed_relationships.csv: the hand-approved spaCy extractions.",
          "csv_family.csv: a family-relations table.",
          "exclusions.csv: 11 documented removals, e.g. Krishna–Radha (later devotional tradition) and a Wikidata error listing Tilottama as Aniruddha's wife.",
        ],
        files: ["data/graph/"],
      },
      {
        id: "normalize", label: "Normalize & merge", sub: "names, types, dedupe", col: 3, row: 2,
        title: "Normalize & merge",
        body: [
          "Diacritics, titles and spellings mapped to one name (Śāntanu, Santanu → Shantanu).",
          "SON_OF and DAUGHTER_OF become CHILD_OF; only six relationship types are accepted.",
          "Duplicates are merged, keeping every origin that supports a fact; excluded facts are dropped.",
          "Fixed in the rework: 94 Wikidata spouse rows that never loaded, and three different Lakshmanas merged into one node.",
        ],
        files: ["src/ingest/graph.py"],
      },
      {
        id: "neo4j", label: "Neo4j graph", sub: "one-transaction rebuild", col: 4, row: 2,
        title: "Neo4j graph",
        body: [
          "210 characters, 1 event, 289 relationships; 75 are supported by two or more origins.",
          "Rebuilt from scratch on every run with batched writes, so deleted facts really disappear.",
        ],
        files: ["src/ingest/graph.py"],
        stats: (h) => [["characters", live(h, "characters")], ["relationships", live(h, "relationships")]],
      },
      {
        id: "serve", label: "Backend start-up", sub: "rag.load()", col: 5, row: 2,
        title: "Backend start-up",
        body: [
          "The FastAPI backend loads the vector index and the embedding model into memory, and the character names from Neo4j.",
          "From there, every question follows the “Answering a question” flow.",
        ],
        files: ["src/rag.py", "src/api.py"],
        jump: "query",
      },
    ],
    edges: [
      ["sources", "parsers"],
      ["parsers", "chunker"],
      ["chunker", "ids"],
      ["ids", "embedder", "new chunks"],
      ["embedder", "index"],
      ["sources", "extract", "narrative only"],
      ["extract", "csvs", "human review"],
      ["csvs", "normalize"],
      ["normalize", "neo4j"],
      ["index", "serve"],
      ["neo4j", "serve"],
    ],
  },
};

// ── Geometry ──────────────────────────────────────────────────────────────

const W = 160, H = 64, COL = 192, X0 = 16;
const ROW_Y = [40, 180, 320];
const VIEW_W = X0 * 2 + COL * 5 + W;
const VIEW_H = 420;

const pos = (n: Node) => ({ x: X0 + n.col * COL, y: ROW_Y[n.row] });

function edgePath(a: Node, b: Node) {
  const p = pos(a), q = pos(b);
  if (a.col === b.col) {
    // Same column: straight down (or up) between the boxes; label beside it.
    const x = p.x + W / 2, down = q.y > p.y;
    return { d: `M ${x} ${down ? p.y + H : p.y} L ${x} ${down ? q.y : q.y + H}`,
             mx: x + 8, my: (p.y + q.y + H) / 2, anchor: "start" as const };
  }
  const x1 = p.x + W, y1 = p.y + H / 2, x2 = q.x, y2 = q.y + H / 2, c = Math.max(24, (x2 - x1) / 2);
  // Neighbours in one row leave no room between the boxes, so label above the row.
  const my = a.row === b.row ? p.y - 10 : (y1 + y2) / 2 - 8;
  return { d: `M ${x1} ${y1} C ${x1 + c} ${y1}, ${x2 - c} ${y2}, ${x2} ${y2}`,
           mx: (x1 + x2) / 2, my, anchor: "middle" as const };
}

// ── Component ─────────────────────────────────────────────────────────────

export default function Architecture({ health }: { health: Health | null }) {
  const [flowId, setFlowId] = useState<FlowId>("query");
  const flow = FLOWS[flowId];
  const [selected, setSelected] = useState(flow.nodes[0].id);
  const [playing, setPlaying] = useState(false);

  const order = flow.nodes.map((n) => n.id);
  const node = flow.nodes.find((n) => n.id === selected) ?? flow.nodes[0];
  const byId = Object.fromEntries(flow.nodes.map((n) => [n.id, n]));

  const switchFlow = (id: FlowId) => {
    setFlowId(id);
    setSelected(FLOWS[id].nodes[0].id);
    setPlaying(false);
  };

  // Play: step through the flow one stage at a time.
  useEffect(() => {
    if (!playing) return;
    const ids = FLOWS[flowId].nodes.map((n) => n.id);
    const t = setTimeout(() => {
      const i = ids.indexOf(selected);
      if (i >= ids.length - 1) setPlaying(false);
      else setSelected(ids[i + 1]);
    }, 2200);
    return () => clearTimeout(t);
  }, [playing, selected, flowId]);

  const play = () => {
    if (!playing && order.indexOf(selected) === order.length - 1) setSelected(order[0]);
    setPlaying(!playing);
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <div className="mr-auto">
          <h2 className="text-lg font-semibold text-zinc-200">How VyasaGraph works</h2>
          <p className="text-sm text-zinc-500">{flow.intro} Click any stage.</p>
        </div>
        <div className="flex rounded-lg border border-zinc-800 p-0.5 text-xs">
          {(Object.keys(FLOWS) as FlowId[]).map((id) => (
            <button
              key={id}
              onClick={() => switchFlow(id)}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                flowId === id ? "bg-violet-600 text-white font-medium" : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {FLOWS[id].name}
            </button>
          ))}
        </div>
        <button
          onClick={play}
          className="px-3 py-2 rounded-lg border border-violet-500/40 text-violet-400 text-xs hover:bg-violet-500/10 transition-colors"
        >
          {playing ? "❚❚ Pause" : "▶ Play"}
        </button>
      </div>

      {/* Diagram */}
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 overflow-x-auto">
        <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="w-full min-w-180" role="img" aria-label={flow.name}>
          <defs>
            {["idle", "active"].map((k) => (
              <marker key={k} id={`arrow-${k}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" className={k === "active" ? "fill-violet-400" : "fill-zinc-600"} />
              </marker>
            ))}
          </defs>

          {flow.edges.map(([from, to, label]) => {
            const { d, mx, my, anchor } = edgePath(byId[from], byId[to]);
            const active = to === selected || from === selected;
            return (
              <g key={`${from}-${to}`}>
                <path
                  d={d}
                  fill="none"
                  strokeWidth={active ? 2 : 1.5}
                  className={`flow-edge ${active ? "stroke-violet-400 flow-edge-active" : "stroke-zinc-600"}`}
                  markerEnd={`url(#arrow-${active ? "active" : "idle"})`}
                />
                {label && (
                  <text x={mx} y={my} textAnchor={anchor} className={`text-[11px] ${active ? "fill-violet-300" : "fill-zinc-500"}`}>
                    {label}
                  </text>
                )}
              </g>
            );
          })}

          {flow.nodes.map((n) => {
            const { x, y } = pos(n);
            const isSel = n.id === selected;
            return (
              <g key={n.id} onClick={() => { setSelected(n.id); setPlaying(false); }} className="cursor-pointer">
                <rect
                  x={x} y={y} width={W} height={H} rx={12}
                  strokeWidth={isSel ? 2 : 1}
                  className={`transition-colors ${isSel ? "fill-violet-500/15 stroke-violet-400" : "fill-zinc-900 stroke-zinc-700 hover:stroke-zinc-500"}`}
                />
                <text x={x + W / 2} y={y + 28} textAnchor="middle" className={`text-[14px] font-medium ${isSel ? "fill-violet-200" : "fill-zinc-200"}`}>
                  {n.label}
                </text>
                <text x={x + W / 2} y={y + 47} textAnchor="middle" className="text-[11px] fill-zinc-500">
                  {n.sub}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Details */}
      <div className="grid md:grid-cols-[1fr_260px] gap-4">
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
          <div className="flex items-baseline gap-2 mb-3">
            <span className="text-xs text-zinc-600">
              {order.indexOf(node.id) + 1}/{order.length}
            </span>
            <h3 className="text-base font-semibold text-violet-300">{node.title}</h3>
          </div>
          <ul className="space-y-2 text-sm text-zinc-300 leading-relaxed list-disc pl-5">
            {node.body.map((b) => <li key={b}>{b}</li>)}
          </ul>
          {node.jump && (
            <button
              onClick={() => switchFlow(node.jump!)}
              className="mt-4 text-xs text-violet-400 hover:text-violet-300"
            >
              Follow a question → {FLOWS[node.jump].name}
            </button>
          )}
        </div>
        <div className="space-y-4">
          {node.stats && (
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-4 space-y-3">
              {node.stats(health).map(([k, v]) => (
                <div key={k}>
                  <div className="text-lg font-semibold text-zinc-100">{v}</div>
                  <div className="text-xs text-zinc-500">{k}</div>
                </div>
              ))}
            </div>
          )}
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-4">
            <div className="text-xs text-zinc-500 mb-2">Code</div>
            <div className="space-y-1">
              {node.files.map((f) => <div key={f} className="font-mono text-xs text-zinc-400 break-all">{f}</div>)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
