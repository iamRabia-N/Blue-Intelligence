// Workaround for Node 20+: some TFJS deps call util.isNullOrUndefined which was removed in Node 20.
import util from "util";
if (typeof (util as any).isNullOrUndefined !== "function") {
  (util as any).isNullOrUndefined = (v: any) => v === null || v === undefined;
}

let useModel: any = null; // Will be set after dynamic import
const embeddingCache = new Map<string, number[]>();

export async function getUseModel(): Promise<any> {
  if (!useModel) {
    // Ensure TensorFlow backend is registered before loading USE.
    await import("@tensorflow/tfjs-node");
    const use = await import("@tensorflow-models/universal-sentence-encoder");
    useModel = await use.load();
  }
  return useModel;
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  if (!texts || texts.length === 0) return [];

  const toCompute: { text: string; idx: number }[] = [];
  const results: (number[] | null)[] = texts.map((t, idx) => {
    const clean = (t || "").trim();
    const cached = embeddingCache.get(clean);
    if (cached) return cached;
    toCompute.push({ text: clean, idx });
    return null;
  });

  if (toCompute.length > 0) {
    const model = await getUseModel();
    const tensor = await model.embed(toCompute.map((t) => t.text));
    try {
      const computed: number[][] = await tensor.array();
      for (let i = 0; i < toCompute.length; i++) {
        const { text, idx } = toCompute[i];
        const emb = computed[i];
        embeddingCache.set(text, emb);
        results[idx] = emb;
      }
    } finally {
      tensor.dispose();
    }
  }

  return results.map((r) => r ?? []);
}

export function cosineSimilarity(a: number[] | null | undefined, b: number[] | null | undefined): number {
  if (!a || !b || a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] ?? 0;
    const bi = b[i] ?? 0;
    dot += ai * bi;
    normA += ai * ai;
    normB += bi * bi;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function serializeEmbedding(emb: number[] | null | undefined): string | null {
  if (!emb || !Array.isArray(emb) || emb.length === 0) return null;
  return JSON.stringify(emb);
}

export function parseEmbedding(embStr: string | null | undefined): number[] | null {
  if (!embStr) return null;
  try {
    const parsed = JSON.parse(embStr);
    if (Array.isArray(parsed) && parsed.every((v) => typeof v === "number")) {
      return parsed as number[];
    }
  } catch {
    // ignore
  }
  return null;
}
