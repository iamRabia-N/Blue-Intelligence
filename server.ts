import { createRequire } from "module";

// Polyfill Node 20's removed util.isNullOrUndefined before any other module loads.
// Some TensorFlow-related dependencies still call util.isNullOrUndefined.
const require = createRequire(import.meta.url);
const util = require("util");
if (typeof (util as any).isNullOrUndefined !== "function") {
  (util as any).isNullOrUndefined = (v: any) => v === null || v === undefined;
}

import express from "express";
import { createServer as createViteServer } from "vite";
import Database from "better-sqlite3";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";
import { JSDOM } from "jsdom";
import { Readability } from "@mozilla/readability";
import TurndownService from "turndown";
import Anthropic from "@anthropic-ai/sdk";
import { isInland as isInlandGSHHG, distanceToCoastKm } from "./lib/gshhg-landmask";
import { embedTexts, cosineSimilarity, serializeEmbedding, parseEmbedding } from "./lib/semanticDedup";

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.join(__dirname, "data");

// Load .env from project root (explicit path so it works regardless of cwd)
const envPath = path.join(__dirname, ".env");
const envLocalPath = path.join(__dirname, ".env.local");
const result = dotenv.config({ path: envPath, quiet: true });
dotenv.config({ path: envLocalPath, override: true, quiet: true });

// Fix BOM: if env key has BOM prefix (e.g. from Windows/editor), copy to correct key
if (result.parsed) {
  for (const key of Object.keys(result.parsed)) {
    const cleanKey = key.replace(/^\uFEFF/, "");
    if (cleanKey !== key) {
      process.env[cleanKey] = result.parsed![key];
    }
  }
}

const db = new Database("blue_intelligence.db");

// Initialize database
db.exec(`
  CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    url TEXT UNIQUE,
    description TEXT,
    funder TEXT,
    lat REAL,
    lng REAL,
    relevance_score REAL,
    category TEXT,
    status TEXT,
    image_url TEXT,
    start_date TEXT,
    end_date TEXT,
    title_embedding TEXT,
    description_embedding TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

try {
  db.exec(`ALTER TABLE projects ADD COLUMN image_url TEXT`);
} catch (e) {
  // Column might already exist, ignore error
}

try {
  db.exec(`ALTER TABLE projects ADD COLUMN start_date TEXT`);
  db.exec(`ALTER TABLE projects ADD COLUMN end_date TEXT`);
} catch (e) {
  // Columns might already exist, ignore error
}

// Migration: Allow null URLs
try {
  const info = db.prepare("PRAGMA table_info(projects)").all();
  const urlCol = info.find((c: any) => c.name === 'url');
  if (urlCol && urlCol.notnull === 1) {
    console.log("[Migration] Making url column nullable...");
    db.transaction(() => {
      db.exec(`
        CREATE TABLE projects_new (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          title TEXT NOT NULL,
          url TEXT UNIQUE,
          description TEXT,
          funder TEXT,
          lat REAL,
          lng REAL,
          relevance_score REAL,
          category TEXT,
          status TEXT,
          image_url TEXT,
          start_date TEXT,
          end_date TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO projects_new (id, title, url, description, funder, lat, lng, relevance_score, category, status, image_url, start_date, end_date, created_at)
        SELECT id, title, url, description, funder, lat, lng, relevance_score, category, status, image_url, start_date, end_date, created_at FROM projects;
        DROP TABLE projects;
        ALTER TABLE projects_new RENAME TO projects;
      `);
    })();
  }
} catch (e) {
  console.error("[Migration] Failed to migrate projects table:", e);
}

db.exec(`
  CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    engine TEXT NOT NULL,
    target_url TEXT NOT NULL,
    status TEXT NOT NULL,
    projects_found INTEGER DEFAULT 0,
    duration_ms INTEGER NOT NULL,
    error_message TEXT,
    raw_response TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

try {
  db.exec(`ALTER TABLE telemetry ADD COLUMN raw_response TEXT`);
} catch (e) { }

try {
  db.exec(`ALTER TABLE projects ADD COLUMN s_ocean_score REAL`);
} catch (e) { }

try {
  db.exec(`ALTER TABLE projects ADD COLUMN title_embedding TEXT`);
} catch (e) { }

try {
  db.exec(`ALTER TABLE projects ADD COLUMN description_embedding TEXT`);
} catch (e) { }

db.exec(`
  CREATE TABLE IF NOT EXISTS failed_extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_url TEXT NOT NULL,
    project_url TEXT UNIQUE NOT NULL,
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

// --- ETL: Structural Memory (MasterSeeds + DeepLinkCache) ---
function ensureDataDir() {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
}

function loadMasterSeeds(): { name: string; url: string }[] {
  ensureDataDir();
  const p = path.join(DATA_DIR, "MasterSeeds.json");
  if (!fs.existsSync(p)) return [];
  try {
    const data = JSON.parse(fs.readFileSync(p, "utf-8"));
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

function loadDeepLinkCache(filename: string): { urls: string[] } {
  ensureDataDir();
  const p = path.join(DATA_DIR, filename);
  if (!fs.existsSync(p)) return { urls: [] };
  try {
    const data = JSON.parse(fs.readFileSync(p, "utf-8"));
    return { urls: Array.isArray(data?.urls) ? data.urls : [] };
  } catch {
    return { urls: [] };
  }
}

function saveDeepLinkCache(filename: string, urls: string[]) {
  ensureDataDir();
  const p = path.join(DATA_DIR, filename);
  const existing = loadDeepLinkCache(filename);
  const merged = [...new Set([...existing.urls, ...urls])];
  fs.writeFileSync(p, JSON.stringify({ urls: merged, updated_at: new Date().toISOString() }, null, 2));
}

function appendToDeepLinkCache(filename: string, newUrls: string[]) {
  const existing = loadDeepLinkCache(filename);
  saveDeepLinkCache(filename, [...existing.urls, ...newUrls]);
}

const GOAL_PROMPT = `
### Objective
L'agent doit uniquement naviguer, gérer la pagination (cliquer sur "Suivant" ou "Charger plus") et identifier les liens vers les fiches individuelles.

### Instructions
1. Navigate to the project directory or listing page.
2. If there is pagination (e.g., "Next", "Page 2", numbers), you MUST visit every page.
3. If there is a "Load More" button or infinite scroll, you MUST trigger it repeatedly until ALL projects are visible. Keep clicking until the button disappears.
4. Identify the links to the individual project detail pages.
5. DO NOT extract detailed data (no descriptions, no coordinates). ONLY extract the URLs.

### Output Format
Return ONLY a clean JSON array of strings representing the absolute URLs of the projects.
Example: ["https://example.com/project1", "https://example.com/project2"]
`;

const EXTRACT_PROMPT = `
### Objective
L'agent doit lire la page du projet et extraire les informations détaillées. Base de données UNIQUEMENT pour projets MARINS/OCÉAN (conservation marine, océans, mers, côtes, espèces marines, récifs coralliens).

### Instructions
1. Read the project details on the page.
2. Extract: title, description, funder, latitude, longitude, category, status, image_url, start_date, end_date.
3. marine_relevance (0-1): 1=clairement marin/océan, 0=pas marin. Si projet terrestre/freshwater/général sans lien océan → 0.
4. location_type: "coastal" si près de la mer, "inland" si en terres (loin des côtes), "unknown" si incertain.
5. Si coordonnées suggèrent un lieu INLAND, être TRÈS strict: marine_relevance >= 0.9 uniquement si lien océan explicite.

### Output Format
Return ONLY a valid JSON object:
{
  "title": "string",
  "url": "string",
  "description": "string",
  "funder": "string",
  "lat": number,
  "lng": number,
  "category": "string",
  "status": "string",
  "image_url": "string",
  "start_date": "string",
  "end_date": "string",
  "marine_relevance": number,
  "location_type": "coastal" | "inland" | "unknown"
}
`;

type GatekeeperConfig = { marine_threshold?: number; inland_threshold?: number; coast_distance_km?: number };
type ExtractionConfig = { concurrency?: number; claudeGatekeeperModel?: string; claudeExtractModel?: string; claudeScoringModel?: string };
type AgentConfig = { maxConcurrentAgents?: number };
type TaskConfig = { gatekeeper?: GatekeeperConfig; extraction?: ExtractionConfig; agent?: AgentConfig };
const agentQueue: { url: string; proxy?: string; mode?: "discover" | "extract"; useTinyFish?: boolean; config?: TaskConfig }[] = [];
let activeAgents = 0;
let maxConcurrentAgents = 2; // Surchargé par config.agent.maxConcurrentAgents au deploy

// Store active runs for SSE proxying and cancellation
let agentCounter = 0;
let extractCounter = 0;
const activeRuns = new Map<string, { streamingUrl: string, logs: any[], aborted?: boolean, status?: string, targetUrl?: string, mode?: string, agentLabel?: string }>();
const activeExtractRuns = new Map<string, { targetUrl: string; status: string; agentLabel: string }>();
let hybridExtractCounter = 0;
const activeHybridExtractions = new Map<string, { targetUrl: string; agentLabel: string; totalUrls: number }>();

// When true, GET /api/agent/active-runs returns [] until next deploy (prevents stale poll data)
// Start stopped so hard reload / server restart shows clean state (no ghost agents)
let swarmStopped = true;

// Mesure du temps de processus (deploy → dernière tâche terminée)
let deployStartTime: number | null = null;

// Extraction hybride en arrière-plan (ne bloque pas le slot TinyFish)
let extractingCount = 0;

function checkSwarmComplete() {
  if (deployStartTime && activeAgents === 0 && agentQueue.length === 0 && extractingCount === 0) {
    const elapsedSec = ((Date.now() - deployStartTime) / 1000).toFixed(1);
    broadcastLog({ key: "etl_swarm_done", sec: elapsedSec });
    console.log(`[Swarm] Process completed in ${elapsedSec}s`);
    deployStartTime = null;
  }
}

/** Extract-only: Readability + Claude pipeline (no TinyFish). Used for Pages cache URLs. */
async function runExtractOnly(projectUrl: string, taskConfig?: TaskConfig): Promise<number> {
  if (swarmStopped) return 0;
  extractCounter++;
  const extractId = `extract-${extractCounter}`;
  const label = `Readability.js ${extractCounter}`;
  activeExtractRuns.set(extractId, { targetUrl: projectUrl, status: "RUNNING", agentLabel: label });
  try {
    const host = (() => { try { return new URL(projectUrl).hostname; } catch { return projectUrl.slice(0, 30); } })();
    broadcastLog({ key: "etl_fetch", host });
    const { markdown, method, imageUrls } = await fetchMarkdown(projectUrl);
    const projectData = await extractProjectData(markdown, projectUrl, taskConfig?.extraction, imageUrls);
    if (swarmStopped) return 0;
    const gatekeeper = passesMarineGatekeeper(projectData, taskConfig?.gatekeeper);
    if (!gatekeeper.pass) {
      broadcastLog({ key: "etl_gatekeeper_rejected", host });
      console.log(`[Gatekeeper] REJECTED ${projectUrl}: ${gatekeeper.reason}`);
      return 0;
    }
    const relevanceScore = typeof projectData.marine_relevance === "number" ? projectData.marine_relevance : 0.95;
    broadcastLog({ key: "etl_claude_extract", title: (projectData.title || "?").slice(0, 40) });
    const upsertResult = await upsertProject({
      title: projectData.title || "Unknown",
      url: projectData.url || projectUrl,
      description: projectData.description || "",
      funder: projectData.funder || "Unknown",
      lat: projectData.lat || 0,
      lng: projectData.lng || 0,
      category: projectData.category || "Marine Conservation",
      status: projectData.status || "Active",
      image_url: projectData.image_url || "",
      start_date: projectData.start_date || null,
      end_date: projectData.end_date || null,
      relevance_score: relevanceScore,
      s_ocean_score: projectData.s_ocean_score ?? 0.75,
    });
    if (upsertResult !== "skipped") {
      broadcastLog({ key: "etl_saved", title: (projectData.title || "?").slice(0, 35) });
    }
    return upsertResult !== "skipped" ? 1 : 0;
  } catch (err: any) {
    console.error(`[Extract] Error for ${projectUrl}:`, err.message);
    throw err;
  } finally {
    activeExtractRuns.delete(extractId);
  }
}

function processQueue() {
  while (!swarmStopped && agentQueue.length > 0) {
    const task = agentQueue[0];
    const mode = task.mode || "discover";
    const useTinyFish = task.useTinyFish === true;
    const atAgentLimit = activeAgents >= maxConcurrentAgents;
    const atTinyFishLimit = (mode === "discover" || useTinyFish) && activeRuns.size >= maxConcurrentAgents;
    if (atAgentLimit || atTinyFishLimit) break;
    activeAgents++;
    agentQueue.shift();
    const shortUrl = task.url.length > 45 ? task.url.slice(0, 42) + "…" : task.url;
    if (mode === "extract") {
      broadcastLog({ key: "etl_extract_only", url: shortUrl });
    }
    (async () => {
      try {
        if (mode === "extract" && useTinyFish) {
          await runTinyFishAgent(task.url, task.proxy, 0, "extract", task.config);
        } else if (mode === "extract") {
          await runExtractOnly(task.url, task.config);
        } else {
          await runTinyFishAgent(task.url, task.proxy, 0, "discover", task.config);
        }
      } catch (error) {
        console.error(`Agent failed for ${task.url}:`, error);
      } finally {
        activeAgents--;
        processQueue();
        checkSwarmComplete();
      }
    })();
  }
}

// --- Dédoublonnage (Follow the Money: 500m, near-identical description) ---
const DEDUP_TITLE_SIMILARITY = 0.85;
const DEDUP_DESC_SIMILARITY = 0.85;
const DEDUP_COORD_KM = 0.5; // 500m per target pipeline

function normalizeText(s: string): string {
  return (s || "").toLowerCase().trim().replace(/\s+/g, " ");
}

function textSimilarity(a: string, b: string, maxLen?: number): number {
  // Fallback similarity: token-based Jaccard. Used when embeddings are not available.
  const na = normalizeText(a);
  const nb = normalizeText(b);
  if (!na || !nb) return na === nb ? 1 : 0;
  let sa = na,
    sb = nb;
  if (maxLen) {
    sa = sa.slice(0, maxLen);
    sb = sb.slice(0, maxLen);
  }
  if (sa === sb) return 1;
  const wa = new Set(sa.split(/\s+/).filter(Boolean));
  const wb = new Set(sb.split(/\s+/).filter(Boolean));
  const inter = [...wa].filter((w) => wb.has(w)).length;
  const union = wa.size + wb.size - inter;
  return union > 0 ? inter / union : 0;
}

async function computeEmbeddingsForProject(project: {
  title: string;
  description?: string;
}): Promise<{ titleEmbedding: number[]; descriptionEmbedding: number[] | null }> {
  const title = normalizeText(project.title);
  const description = project.description ? normalizeText(project.description) : "";
  const texts = [title];
  if (description) texts.push(description);
  const embeddings = await embedTexts(texts);
  return {
    titleEmbedding: embeddings[0] ?? [],
    descriptionEmbedding: description ? embeddings[1] ?? null : null,
  };
}

const updateEmbeddingsStmt = db.prepare(`
  UPDATE projects SET title_embedding = ?, description_embedding = ? WHERE id = ?
`);

async function findDuplicateProject(project: {
  title: string;
  description: string;
  lat: number;
  lng: number;
  title_embedding?: number[];
  description_embedding?: number[] | null;
}): Promise<{ id: number; url: string } | null> {
  const { title, description, lat, lng } = project;
  const { titleEmbedding, descriptionEmbedding } =
    project.title_embedding && project.description_embedding !== undefined
      ? {
        titleEmbedding: project.title_embedding,
        descriptionEmbedding: project.description_embedding ?? null,
      }
      : await computeEmbeddingsForProject({ title, description });

  const delta = 0.02; // ~2km bounding box
  const rows = selectCandidatesByCoords.all(
    lat - delta,
    lat + delta,
    lng - delta,
    lng + delta
  ) as { id: number; title: string; url: string; description: string; lat: number; lng: number; title_embedding?: string; description_embedding?: string }[];

  for (const row of rows) {
    const dist = haversineDistanceKm(lat, lng, row.lat, row.lng);
    if (dist > DEDUP_COORD_KM) continue;

    let candidateTitleEmb = parseEmbedding(row.title_embedding);
    let candidateDescEmb = parseEmbedding(row.description_embedding);

    // If the row lacks embeddings, compute and persist them for future runs.
    if (!candidateTitleEmb || (row.description && !candidateDescEmb)) {
      const { titleEmbedding: computedTitleEmb, descriptionEmbedding: computedDescEmb } =
        await computeEmbeddingsForProject({ title: row.title, description: row.description || "" });
      candidateTitleEmb = candidateTitleEmb || computedTitleEmb;
      candidateDescEmb = candidateDescEmb || computedDescEmb;
      try {
        updateEmbeddingsStmt.run(serializeEmbedding(candidateTitleEmb), serializeEmbedding(candidateDescEmb), row.id);
      } catch (e) {
        // ignore write failures
      }
    }

    const titleSim = cosineSimilarity(titleEmbedding, candidateTitleEmb);
    const hasDescriptionComparison = descriptionEmbedding && candidateDescEmb;
    const descSim = hasDescriptionComparison ? cosineSimilarity(descriptionEmbedding, candidateDescEmb) : 1;

    if (titleSim >= DEDUP_TITLE_SIMILARITY && descSim >= DEDUP_DESC_SIMILARITY) {
      return { id: row.id, url: row.url };
    }

    // Fallback to token-level similarity when embeddings are unavailable
    if ((!titleEmbedding.length || !candidateTitleEmb?.length) && (titleSim >= DEDUP_TITLE_SIMILARITY)) {
      const fallbackTitleSim = textSimilarity(title, row.title);
      const fallbackDescSim = textSimilarity(description, row.description || "", 200);
      if (fallbackTitleSim >= DEDUP_TITLE_SIMILARITY && fallbackDescSim >= DEDUP_DESC_SIMILARITY) {
        return { id: row.id, url: row.url };
      }
    }
  }

  return null;
}

function haversineDistanceKm(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLng / 2) * Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

const selectCandidatesByCoords = db.prepare(`
  SELECT id, title, url, description, lat, lng, title_embedding, description_embedding FROM projects
  WHERE lat BETWEEN ? AND ? AND lng BETWEEN ? AND ?
`);

const updateProjectByIdStmt = db.prepare(`
  UPDATE projects SET
    title = ?, description = ?, funder = ?, lat = ?, lng = ?,
    category = ?, status = ?, image_url = ?, start_date = ?, end_date = ?,
    relevance_score = ?, s_ocean_score = ?,
    title_embedding = ?, description_embedding = ?
  WHERE id = ?
`);

const insertProjectStmt = db.prepare(`
  INSERT INTO projects (title, url, description, funder, lat, lng, category, status, relevance_score, image_url, start_date, end_date, s_ocean_score, title_embedding, description_embedding)
  VALUES (@title, @url, @description, @funder, @lat, @lng, @category, @status, @relevance_score, @image_url, @start_date, @end_date, @s_ocean_score, @title_embedding, @description_embedding)
  ON CONFLICT(url) DO UPDATE SET
    title = excluded.title,
    description = excluded.description,
    funder = excluded.funder,
    lat = excluded.lat,
    lng = excluded.lng,
    category = excluded.category,
    status = excluded.status,
    image_url = excluded.image_url,
    start_date = excluded.start_date,
    end_date = excluded.end_date,
    relevance_score = excluded.relevance_score,
    s_ocean_score = excluded.s_ocean_score,
    title_embedding = excluded.title_embedding,
    description_embedding = excluded.description_embedding
`);

type ProjectRow = {
  title: string;
  url?: string;
  description?: string;
  funder?: string | string[];
  lat: number;
  lng: number;
  category?: string;
  status?: string;
  image_url?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  relevance_score?: number;
  s_ocean_score?: number;
  title_embedding?: number[];
  description_embedding?: number[] | null;
};

function chooseBestValue(existing: any, incoming: any): any {
  if (incoming === undefined || incoming === null || incoming === "") return existing;
  if (existing === undefined || existing === null || existing === "") return incoming;
  if (typeof existing === "string" && typeof incoming === "string") {
    // Prefer longer text (more complete) when there is a difference
    return incoming.length > existing.length ? incoming : existing;
  }
  return incoming;
}

async function upsertProject(p: ProjectRow): Promise<"inserted" | "updated" | "skipped"> {
  if (swarmStopped) return "skipped"; // Stop all inserts when swarm is stopped
  if (!p || !p.title || p.lat === undefined || p.lng === undefined) return "skipped";
  const lat = parseFloat(String(p.lat));
  const lng = parseFloat(String(p.lng));
  if (isNaN(lat) || isNaN(lng)) return "skipped";

  const title = p.title;
  const description = p.description || "";
  const funder = Array.isArray(p.funder) ? p.funder.join(", ") : (p.funder || "");
  const projectUrl = p.url || `internal://${title.replace(/[^\w]/g, "-").toLowerCase()}-${lat}-${lng}`;

  const { titleEmbedding, descriptionEmbedding } = await computeEmbeddingsForProject({ title, description });

  const existingByUrl = db.prepare("SELECT id, title, description, funder FROM projects WHERE url = ?").get(projectUrl) as
    | { id: number; title: string; description: string; funder: string }
    | undefined;

  const mergedFunder = (existingByUrl?.funder || "")
    ? [...new Set([...(existingByUrl?.funder || "").split(",").map((f) => f.trim()).filter(Boolean), ...funder.split(",").map((f) => f.trim()).filter(Boolean)])].join(", ")
    : funder;

  const applyUpdate = (id: number, existing: any) => {
    updateProjectByIdStmt.run(
      chooseBestValue(existing.title, title),
      chooseBestValue(existing.description, description),
      mergedFunder,
      lat,
      lng,
      p.category || "General",
      p.status || "Active",
      p.image_url ?? null,
      p.start_date ?? null,
      p.end_date ?? null,
      p.relevance_score ?? 0.95,
      p.s_ocean_score ?? 0.75,
      serializeEmbedding(titleEmbedding),
      serializeEmbedding(descriptionEmbedding),
      id
    );
  };

  if (existingByUrl) {
    applyUpdate(existingByUrl.id, existingByUrl);
    return "updated";
  }

  const dup = await findDuplicateProject({
    title,
    description,
    lat,
    lng,
    title_embedding: titleEmbedding,
    description_embedding: descriptionEmbedding,
  });
  if (dup) {
    const existing = db.prepare("SELECT title, description, funder FROM projects WHERE id = ?").get(dup.id) as
      | { title: string; description: string; funder: string }
      | undefined;
    applyUpdate(dup.id, existing || {});
    return "updated";
  }

  insertProjectStmt.run({
    title,
    url: projectUrl,
    description,
    funder,
    lat,
    lng,
    category: p.category || "General",
    status: p.status || "Active",
    image_url: p.image_url ?? null,
    start_date: p.start_date ?? null,
    end_date: p.end_date ?? null,
    relevance_score: p.relevance_score ?? 0.95,
    s_ocean_score: p.s_ocean_score ?? 0.75,
    title_embedding: serializeEmbedding(titleEmbedding),
    description_embedding: serializeEmbedding(descriptionEmbedding),
  });
  const row = db.prepare("SELECT id FROM projects WHERE url = ?").get(projectUrl) as { id: number } | undefined;
  const feature = {
    type: "Feature",
    geometry: { type: "Point", coordinates: [lng, lat] },
    properties: {
      id: row?.id,
      title,
      url: projectUrl,
      description,
      funder,
      relevance_score: p.relevance_score ?? 0.95,
      s_ocean_score: p.s_ocean_score ?? 0.75,
      category: p.category,
      status: p.status,
      image_url: p.image_url,
      start_date: p.start_date,
      end_date: p.end_date,
    },
  };
  broadcastNewProject(feature);
  return "inserted";
}

async function insertManyProjects(projects: any[]): Promise<number> {
  let count = 0;
  for (const p of projects) {
    const result = await upsertProject(p);
    if (result !== "skipped") count++;
  }
  return count;
}

async function saveProjects(projects: any[]): Promise<number> {
  console.log(`[Database] saveProjects called with ${projects.length} items`);
  const savedCount = await insertManyProjects(projects);
  console.log(`[Database] Saved ${savedCount} projects to database`);
  return savedCount;
}

function recordTelemetry(engine: string, targetUrl: string, status: string, projectsFound: number, durationMs: number, errorMessage: string | null = null, rawResponse: string | null = null) {
  db.prepare(`
    INSERT INTO telemetry (engine, target_url, status, projects_found, duration_ms, error_message, raw_response)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `).run(engine, targetUrl, status, projectsFound, durationMs, errorMessage, rawResponse);
}

function parseProjectsData(projectsData: any) {
  console.log(`[Parser] Parsing data of type: ${typeof projectsData}`);
  let projects = projectsData;

  // If it's a string, try to parse it as JSON first
  if (typeof projects === 'string') {
    try {
      const parsed = JSON.parse(projects);
      if (typeof parsed === 'object' && parsed !== null) {
        projects = parsed;
      }
    } catch (e) {
      // Not a valid JSON object string, continue
    }
  }

  // Extract string from result/output wrapper if present
  if (typeof projects === 'object' && projects !== null && !Array.isArray(projects)) {
    if (typeof projects.result === 'string') {
      projects = projects.result;
    } else if (typeof projects.output === 'string') {
      projects = projects.output;
    }
  }

  if (typeof projects === 'string') {
    try {
      // Try to find JSON block
      const jsonMatch = projects.match(/```json\s*([\s\S]*?)\s*```/) || projects.match(/```\s*([\s\S]*?)\s*```/);
      const cleaned = jsonMatch ? jsonMatch[1] : projects.trim();

      // Remove any leading/trailing non-JSON characters if no block found
      let jsonStr = cleaned;
      if (!jsonMatch) {
        const start = cleaned.indexOf('[');
        const end = cleaned.lastIndexOf(']');
        const startObj = cleaned.indexOf('{');
        const endObj = cleaned.lastIndexOf('}');

        if (start !== -1 && end !== -1 && (start < startObj || startObj === -1)) {
          jsonStr = cleaned.substring(start, end + 1);
        } else if (startObj !== -1 && endObj !== -1) {
          jsonStr = cleaned.substring(startObj, endObj + 1);
        }
      }

      console.log(`[Parser] Attempting to parse JSON string: ${jsonStr.substring(0, 100)}...`);
      projects = JSON.parse(jsonStr);
    } catch (e) {
      console.error("Failed to parse string result:", typeof projects === 'string' ? projects.substring(0, 200) : projects);
      projects = [];
    }
  }

  if (!Array.isArray(projects) && projects !== null && typeof projects === 'object') {
    console.log(`[Parser] Data is object, extracting array. Keys: ${Object.keys(projects).join(', ')}`);
    if (projects.type === "FeatureCollection" && Array.isArray(projects.features)) {
      projects = projects.features;
    } else if (projects.projects && Array.isArray(projects.projects)) {
      projects = projects.projects;
    } else if (projects.result && Array.isArray(projects.result)) {
      projects = projects.result;
    } else if (projects.features && Array.isArray(projects.features)) {
      projects = projects.features;
    } else {
      projects = [projects];
    }
  }

  if (!Array.isArray(projects)) {
    console.warn("[Parser] Could not find array in data");
    return [];
  }

  console.log(`[Parser] Normalizing ${projects.length} items`);
  // Map GeoJSON features or normalize flat objects
  return projects.map((p: any) => {
    let normalized = p;
    if (p.type === "Feature" && p.properties && p.geometry) {
      normalized = {
        ...p.properties,
        lat: p.geometry.coordinates?.[1],
        lng: p.geometry.coordinates?.[0]
      };
    }

    // Normalize lat/lng keys
    if (normalized.latitude !== undefined && normalized.lat === undefined) normalized.lat = normalized.latitude;
    if (normalized.longitude !== undefined && normalized.lng === undefined) normalized.lng = normalized.longitude;

    return normalized;
  });
}

import { htmlToMarkdown } from "mdream";
import FirecrawlApp from "@mendable/firecrawl-js";

const turndownService = new TurndownService();

/** Extract image URLs from HTML (og:image, twitter:image, first content img). Used as fallback when Claude returns empty. */
function extractImageUrlsFromHtml(html: string, baseUrl: string): string[] {
  const urls: string[] = [];
  try {
    const doc = new JSDOM(html, { url: baseUrl });
    const document = doc.window.document;
    const metaOg = document.querySelector('meta[property="og:image"]');
    const metaTwitter = document.querySelector('meta[name="twitter:image"], meta[property="twitter:image"]');
    const firstImg = document.querySelector('article img, main img, [role="main"] img, .content img, .project img, .hero img, img[src*="project"], img[src*="hero"], img');
    if (metaOg?.getAttribute("content")) {
      const href = metaOg.getAttribute("content")!.trim();
      if (href.startsWith("http")) urls.push(href);
      else urls.push(new URL(href, baseUrl).href);
    }
    if (metaTwitter?.getAttribute("content")) {
      const href = metaTwitter.getAttribute("content")!.trim();
      if (href.startsWith("http")) urls.push(href);
      else urls.push(new URL(href, baseUrl).href);
    }
    if (firstImg?.getAttribute("src")) {
      const href = firstImg.getAttribute("src")!.trim();
      if (href.startsWith("http")) urls.push(href);
      else urls.push(new URL(href, baseUrl).href);
    }
    return [...new Set(urls)];
  } catch {
    return [];
  }
}

async function fetchMarkdown(url: string): Promise<{ markdown: string, method: string, imageUrls?: string[] }> {
  let html = "";
  try {
    // Niveau 1: Local & Gratuit (Readability)
    const res = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    html = await res.text();

    const doc = new JSDOM(html, { url });
    const reader = new Readability(doc.window.document);
    const article = reader.parse();

    if (article && article.textContent.length > 500) {
      const markdown = turndownService.turndown(article.content);
      const imageUrls = extractImageUrlsFromHtml(html, url);
      return { markdown, method: 'Readability', imageUrls };
    } else {
      throw new Error('Content too short or empty (JS-heavy)');
    }
  } catch (err: any) {
    console.log(`[Web Reading] Readability failed for ${url} (${err.message}), switching to Mdream...`);

    try {
      // Niveau 2: Mdream
      if (html) {
        const markdown = await htmlToMarkdown(html);
        if (markdown && markdown.length > 500) {
          const imageUrls = extractImageUrlsFromHtml(html, url);
          return { markdown, method: 'Mdream', imageUrls };
        }
      }
      throw new Error('Mdream result too short or empty');
    } catch (mdreamErr: any) {
      console.log(`[Web Reading] Mdream failed for ${url} (${mdreamErr.message}), switching to fallbacks...`);

      // Niveau 3: Firecrawl
      if (process.env.FIRECRAWL_API_KEY) {
        try {
          console.log(`[Web Reading] Trying Firecrawl for ${url}...`);
          const firecrawl = new FirecrawlApp({ apiKey: process.env.FIRECRAWL_API_KEY });
          const scrapeResult = await firecrawl.scrape(url, { formats: ['markdown'] }) as any;
          if (!scrapeResult.success) {
            throw new Error(scrapeResult.error || "Firecrawl scrape failed");
          }
          return { markdown: scrapeResult.markdown || "", method: 'Firecrawl' };
        } catch (firecrawlErr: any) {
          console.log(`[Web Reading] Firecrawl failed for ${url} (${firecrawlErr.message})`);
        }
      }

      // Niveau 4: Scrape.do
      const scrapeDoKey = process.env['SCRAPE.DO_API_KEY'] || process.env.SCRAPE_DO_API_KEY;
      if (scrapeDoKey) {
        try {
          console.log(`[Web Reading] Trying Scrape.do for ${url}...`);
          const scrapeRes = await fetch(`http://api.scrape.do?token=${scrapeDoKey}&url=${encodeURIComponent(url)}`);
          if (!scrapeRes.ok) throw new Error(`Scrape.do HTTP ${scrapeRes.status}`);
          const scrapeHtml = await scrapeRes.text();
          const markdown = await htmlToMarkdown(scrapeHtml);
          if (markdown && markdown.length > 500) {
            const imageUrls = extractImageUrlsFromHtml(scrapeHtml, url);
            return { markdown, method: 'Scrape.do', imageUrls };
          }
          throw new Error('Scrape.do result too short or empty');
        } catch (scrapeErr: any) {
          console.log(`[Web Reading] Scrape.do failed for ${url} (${scrapeErr.message})`);
        }
      }

      // Niveau 5: Jina Reader
      if (process.env.JINA_READER_API_KEY) {
        try {
          console.log(`[Web Reading] Trying Jina Reader for ${url}...`);
          const jinaRes = await fetch(`https://r.jina.ai/${url}`, {
            headers: { 'Authorization': `Bearer ${process.env.JINA_READER_API_KEY}` }
          });
          if (!jinaRes.ok) throw new Error(`Jina HTTP ${jinaRes.status}`);
          const markdown = await jinaRes.text();
          if (markdown && markdown.length > 500) {
            return { markdown, method: 'Jina Reader' };
          }
          throw new Error('Jina Reader result too short or empty');
        } catch (jinaErr: any) {
          console.log(`[Web Reading] Jina Reader failed for ${url} (${jinaErr.message})`);
          throw new Error(`All extraction methods failed. Last error: ${jinaErr.message}`);
        }
      }
      throw new Error(`All extraction methods failed. No API keys available for fallback.`);
    }
  }
}

const DEFAULT_GATEKEEPER = {
  marine_threshold: 0.75,
  inland_threshold: 0.9,
  coast_distance_km: 0, // 0 = comportement original (GSHHG point-in-polygon). >0 = tolérer projets à X km de la côte
};

/** Gatekeeper: reject projects not genuinely related to marine/ocean conservation. Stricter when inland or far from coast. */
function passesMarineGatekeeper(
  projectData: any,
  config?: { marine_threshold?: number; inland_threshold?: number; coast_distance_km?: number }
): { pass: boolean; reason?: string } {
  const marineRelevance = typeof projectData.marine_relevance === "number" ? projectData.marine_relevance : 0;
  const lat = parseFloat(projectData.lat) || 0;
  const lng = parseFloat(projectData.lng) || 0;
  const locationType = projectData.location_type || "unknown";
  const cfg = { ...DEFAULT_GATEKEEPER, ...config };
  const marineThreshold = cfg.marine_threshold ?? DEFAULT_GATEKEEPER.marine_threshold;
  const inlandThreshold = cfg.inland_threshold ?? DEFAULT_GATEKEEPER.inland_threshold;
  const coastKm = cfg.coast_distance_km ?? 0;

  let inland: boolean;
  if (coastKm > 0) {
    const dist = distanceToCoastKm(lat, lng);
    inland = dist > coastKm;
  } else {
    const coordsInland = isInlandGSHHG(lat, lng);
    inland = coordsInland || locationType === "inland";
  }

  const threshold = inland ? inlandThreshold : marineThreshold;
  if (marineRelevance < threshold) {
    return {
      pass: false,
      reason: inland
        ? (coastKm > 0
          ? `Gatekeeper: projet à >${coastKm} km de la côte (${lat.toFixed(2)}, ${lng.toFixed(2)}), marine_relevance=${marineRelevance.toFixed(2)} < ${threshold}`
          : `Gatekeeper: projet situé en terres (${lat.toFixed(2)}, ${lng.toFixed(2)}), marine_relevance=${marineRelevance.toFixed(2)} < ${threshold}`)
        : `Gatekeeper: marine_relevance=${marineRelevance.toFixed(2)} < ${threshold}`,
    };
  }
  return { pass: true };
}

/** Stage 1: Claude Haiku - Quick gatekeeper filter */
async function gatekeeperOnly(markdown: string, url: string, modelOverride?: string): Promise<{ pass: boolean; marine_relevance: number; location_type: string }> {
  const apiKey = process.env.CLAUDE_API_KEY || process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("CLAUDE_API_KEY (or ANTHROPIC_API_KEY) is not configured");
  const client = new Anthropic({ apiKey });
  const model = modelOverride || CLAUDE_GATEKEEPER_MODEL;
  const excerpt = markdown.slice(0, 4000);
  const response = await client.messages.create({
    model,
    max_tokens: 128,
    temperature: 0,
    messages: [{
      role: "user",
      content: `URL: ${url}\n\nIs this page about OCEAN/MARINE conservation? Return ONLY valid JSON: { "marine_relevance": 0-1, "location_type": "coastal"|"inland"|"unknown" }. 0=not marine, 1=clearly marine. No markdown, no explanation.\n\nExcerpt:\n${excerpt}`
    }],
  });
  const textBlock = (response.content as any[]).find((b: any) => b.type === "text");
  const raw = (textBlock?.text || "{}").trim();
  const jsonMatch = raw.match(/\{[\s\S]*\}/);
  const data = JSON.parse(jsonMatch ? jsonMatch[0] : "{}");
  const mr = typeof data.marine_relevance === "number" ? data.marine_relevance : 0.5;
  const lt = data.location_type || "unknown";
  const gk = passesMarineGatekeeper({ marine_relevance: mr, location_type: lt, lat: 0, lng: 0 });
  return { pass: gk.pass, marine_relevance: mr, location_type: lt };
}

const CLAUDE_EXTRACT_MODEL = process.env.CLAUDE_EXTRACT_MODEL || "claude-sonnet-4-5-20250929";
const CLAUDE_GATEKEEPER_MODEL = process.env.CLAUDE_GATEKEEPER_MODEL || "claude-haiku-4-5-20251001";
const EXTRACT_CONCURRENCY = Math.max(1, Math.min(20, parseInt(process.env.EXTRACT_CONCURRENCY || "2", 10)));

/** Run async tasks with limited concurrency (worker pool). */
async function runWithConcurrency<T>(
  items: T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<number>,
  shouldAbort?: () => boolean
): Promise<number> {
  if (items.length === 0) return 0;
  let nextIndex = 0;
  const results: number[] = new Array(items.length);
  const workers = Array(Math.min(concurrency, items.length))
    .fill(null)
    .map(async () => {
      while (true) {
        if (shouldAbort?.()) return;
        const i = nextIndex++;
        if (i >= items.length) return;
        try {
          results[i] = await fn(items[i], i);
        } catch {
          results[i] = 0;
        }
      }
    });
  await Promise.all(workers);
  return results.reduce((a, b) => a + (b || 0), 0);
}

/** Stage 2: Claude Sonnet - Analysis, geocoding, coastal snapping. Description < 250 chars */
async function extractAndGeocode(markdown: string, url: string, modelOverride?: string, imageUrls?: string[]): Promise<any> {
  const apiKey = process.env.CLAUDE_API_KEY || process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("CLAUDE_API_KEY (or ANTHROPIC_API_KEY) is not configured");
  const client = new Anthropic({ apiKey });
  const model = modelOverride || CLAUDE_EXTRACT_MODEL;
  const response = await client.messages.create({
    model,
    max_tokens: 1024,
    temperature: 0,
    messages: [{
      role: "user",
      content: `Extract marine conservation project from this page. URL: ${url}

RULES:
- description: answer why, what, how, when, where in < 250 characters.
- If coordinates (lat,lng) are INLAND (Paris, Geneva, US Midwest), apply COASTAL SNAPPING: recalculate to nearest maritime/coastal point.
- marine_relevance 0-1, location_type coastal|inland|unknown
- image_url: extract ANY project-related image. Prefer: og:image meta, first img/figure in content, hero/banner images, logos, diagrams. Use absolute URLs. If the page links to PDFs with images, use the PDF URL or an image URL from the same domain. Empty string only if no image found.
${imageUrls && imageUrls.length > 0 ? `\nAvailable images from page metadata (use the most relevant): ${imageUrls.slice(0, 5).join(", ")}` : ""}

Return ONLY valid JSON (no markdown, no explanation):
{"title":"string","url":"string","description":"string","funder":"string","lat":number,"lng":number,"category":"string","status":"string","image_url":"string","start_date":"string","end_date":"string","marine_relevance":number,"location_type":"coastal|inland|unknown"}

Markdown:
${markdown.slice(0, 12000)}`
    }],
  });
  const textBlock = (response.content as any[]).find((b: any) => b.type === "text");
  const raw = (textBlock?.text || "{}").trim();
  const jsonMatch = raw.match(/\{[\s\S]*\}/);
  const data = JSON.parse(jsonMatch ? jsonMatch[0] : "{}");
  if (data.marine_relevance === undefined) data.marine_relevance = 0.95;
  if (data.location_type === undefined) data.location_type = "unknown";
  if ((!data.image_url || data.image_url.trim() === "") && imageUrls && imageUrls.length > 0) {
    data.image_url = imageUrls[0];
  }
  return data;
}

/** Stage 3: Claude Sonnet - S_ocean scoring */
async function scoreSOcean(projectData: any, modelOverride?: string): Promise<number> {
  const apiKey = process.env.CLAUDE_API_KEY || process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("CLAUDE_API_KEY (or ANTHROPIC_API_KEY) is not configured");
  const client = new Anthropic({ apiKey });
  const model = modelOverride || CLAUDE_EXTRACT_MODEL;
  const response = await client.messages.create({
    model,
    max_tokens: 64,
    temperature: 0,
    messages: [{
      role: "user",
      content: `Score this marine project 0-1 (S_ocean): technicality, source reliability, oceanic localization. Return ONLY valid JSON: { "s_ocean_score": number }. No markdown, no explanation.\n\nProject: ${projectData.title}\n${projectData.description?.slice(0, 200) || ""}\nURL: ${projectData.url || ""}\nCoords: ${projectData.lat}, ${projectData.lng}`
    }],
  });
  const textBlock = (response.content as any[]).find((b: any) => b.type === "text");
  const raw = (textBlock?.text || "{}").trim();
  const jsonMatch = raw.match(/\{[\s\S]*\}/);
  const data = JSON.parse(jsonMatch ? jsonMatch[0] : "{}");
  return typeof data.s_ocean_score === "number" ? Math.max(0, Math.min(1, data.s_ocean_score)) : 0.75;
}

/** Full 3-stage pipeline: Haiku gatekeeper → Sonnet extract+coastal snapping → Sonnet S_ocean */
async function extractProjectData(markdown: string, url: string, extractionConfig?: ExtractionConfig, imageUrls?: string[]) {
  const gatekeeperModel = extractionConfig?.claudeGatekeeperModel || CLAUDE_GATEKEEPER_MODEL;
  const extractModel = extractionConfig?.claudeExtractModel || CLAUDE_EXTRACT_MODEL;
  const gatekeeper = await gatekeeperOnly(markdown, url, gatekeeperModel);
  if (!gatekeeper.pass) {
    throw new Error(`Gatekeeper rejected: marine_relevance=${gatekeeper.marine_relevance}`);
  }
  const projectData = await extractAndGeocode(markdown, url, extractModel, imageUrls);
  projectData.marine_relevance = gatekeeper.marine_relevance;
  projectData.location_type = gatekeeper.location_type;
  const scoringModel = extractionConfig?.claudeScoringModel || extractionConfig?.claudeExtractModel || CLAUDE_EXTRACT_MODEL;
  projectData.s_ocean_score = await scoreSOcean(projectData, scoringModel);
  if (!projectData.url || projectData.url.trim() === "") projectData.url = url;
  return projectData;
}

function parseUrlsData(projectsData: any): string[] {
  console.log(`[Parser] Parsing URLs data of type: ${typeof projectsData}`);
  let projects = projectsData;

  if (typeof projects === 'string') {
    try {
      const parsed = JSON.parse(projects);
      if (typeof parsed === 'object' && parsed !== null) {
        projects = parsed;
      }
    } catch (e) { }
  }

  if (typeof projects === 'object' && projects !== null && !Array.isArray(projects)) {
    if (typeof projects.result === 'string') projects = projects.result;
    else if (typeof projects.output === 'string') projects = projects.output;
  }

  if (typeof projects === 'string') {
    try {
      const jsonMatch = projects.match(/```json\s*([\s\S]*?)\s*```/) || projects.match(/```\s*([\s\S]*?)\s*```/);
      const cleaned = jsonMatch ? jsonMatch[1] : projects.trim();

      let jsonStr = cleaned;
      if (!jsonMatch) {
        const start = cleaned.indexOf('[');
        const end = cleaned.lastIndexOf(']');
        if (start !== -1 && end !== -1) {
          jsonStr = cleaned.substring(start, end + 1);
        }
      }

      projects = JSON.parse(jsonStr);
    } catch (e) {
      console.error("Failed to parse string result:", typeof projects === 'string' ? projects.substring(0, 200) : projects);
      projects = [];
    }
  }

  if (!Array.isArray(projects) && projects !== null && typeof projects === 'object') {
    if (projects.urls && Array.isArray(projects.urls)) projects = projects.urls;
    else if (projects.result && Array.isArray(projects.result)) projects = projects.result;
    else projects = [projects];
  }

  if (!Array.isArray(projects)) return [];

  return projects.filter(p => typeof p === 'string');
}

async function runTinyFishAgent(targetUrl: string, proxy?: string, retryCount = 0, mode: 'discover' | 'extract' = 'discover', taskConfig?: TaskConfig) {
  if (swarmStopped) return 0;
  const startTime = performance.now();
  console.log(`[TinyFish] Starting agent for: ${targetUrl} (Attempt ${retryCount + 1}, Mode: ${mode})`);
  let runId = "";
  try {
    const apiKey = process.env.TINYFISH_API_KEY;
    if (!apiKey) throw new Error("TINYFISH_API_KEY is not configured");

    // 1. Launch TinyFish Run
    const response = await fetch("https://agent.tinyfish.ai/v1/automation/run-async", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey,
      },
      body: JSON.stringify({
        url: targetUrl,
        goal: mode === 'extract' ? EXTRACT_PROMPT : GOAL_PROMPT,
        max_steps: 60
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[TinyFish] API Error Response (${response.status}):`, errorText);
      throw new Error(`TinyFish API Error: ${errorText}`);
    }

    const runData = await response.json();
    const safeRunData = { ...runData };
    if (safeRunData.error === null) delete safeRunData.error;
    if (process.env.DEBUG_TINYFISH) {
      console.log(`[TinyFish] Initial Run Data:`, JSON.stringify(safeRunData));
    }

    if ((!runData.id && !runData.run_id) || (runData.error && runData.error !== null)) {
      console.error(`[TinyFish] Failed to start run for ${targetUrl}. Full Response:`, JSON.stringify(runData));
      const errorMsg = typeof runData.error === 'object' ? JSON.stringify(runData.error) : runData.error;
      throw new Error(errorMsg || `TinyFish failed to start run. Response: ${JSON.stringify(runData)}`);
    }
    runId = runData.id || runData.run_id;
    const streamingUrl = runData.streamingUrl || runData.streaming_url;

    console.log(`[TinyFish] Run started: ${runId} for ${targetUrl}. Status: ${runData.status || 'UNKNOWN'}`);
    if (runData.error) console.log(`[TinyFish] Note: Run started with non-fatal error key: ${runData.error}`);

    if (swarmStopped) return 0; // Don't increment or register if we stopped
    agentCounter++;
    const label = `TinyFish ${agentCounter}`;
    activeRuns.set(runId, { streamingUrl, logs: [], status: runData.status || "PENDING", targetUrl, mode, agentLabel: label });
    const shortUrl = targetUrl.length > 50 ? targetUrl.slice(0, 47) + "…" : targetUrl;
    broadcastLog({ key: "swarm_dispatched", label, url: shortUrl });
    broadcastLog({ key: "swarm_browsing", label, modeKey: mode === "extract" ? "extract" : "discovery", url: shortUrl });

    // 3. Poll for completion (or we could use SSE, but for the backend poll is safer for DB update)
    let status = runData.status || "PENDING";
    let result = null;
    let pendingStartTime = Date.now();
    const PENDING_TIMEOUT_MS = 120000; // 2 minutes: free slot if TinyFish never starts
    let lastPendingLog = 0;

    while (status === "RUNNING" || status === "PENDING") {
      // Check if run was aborted
      const currentRun = activeRuns.get(runId);
      if (!currentRun || currentRun.aborted) {
        console.log(`[TinyFish] Run ${runId} was aborted or removed. Stopping poll.`);
        break;
      }

      // Safety timeout for PENDING state
      const timeInPending = Date.now() - pendingStartTime;
      if (status === "PENDING" && (timeInPending > PENDING_TIMEOUT_MS)) {
        console.error(`[TinyFish] Run ${runId} timed out in PENDING state after ${Math.round(timeInPending / 1000)}s. Freeing slot.`);
        activeRuns.delete(runId);
        throw new Error("Agent timed out while waiting to start (PENDING state too long). Slot freed for next task.");
      }
      // Log PENDING at most every 30s to avoid spam
      if (status === "PENDING" && (Date.now() - lastPendingLog > 30000)) {
        lastPendingLog = Date.now();
        console.log(`[TinyFish] Run ${runId} still PENDING (${Math.round(timeInPending / 1000)}s). Timeout in ${Math.round((PENDING_TIMEOUT_MS - timeInPending) / 1000)}s.`);
      }

      await new Promise(resolve => setTimeout(resolve, 5000));
      try {
        const statusRes = await fetch(`https://agent.tinyfish.ai/v1/runs/${runId}`, {
          headers: { "X-API-Key": apiKey }
        });

        if (!statusRes.ok) {
          console.error(`[TinyFish] Failed to fetch status for run ${runId}. Status: ${statusRes.status}`);
          continue; // Try again next loop
        }

        const statusData = await statusRes.json();
        status = statusData.status;

        // Update streaming URL if it was missing and is now available
        if (currentRun && !currentRun.streamingUrl && (statusData.streamingUrl || statusData.streaming_url)) {
          activeRuns.set(runId, {
            ...currentRun,
            streamingUrl: statusData.streamingUrl || statusData.streaming_url,
            status: status
          });
        } else if (currentRun) {
          activeRuns.set(runId, { ...currentRun, status: status });
        }

        if (status !== "RUNNING" && status !== "PENDING") {
          console.log(`[TinyFish] Run ${runId} status: ${status}`);
        }

        if (status === "COMPLETED") {
          broadcastLog({ key: "swarm_complete", label, modeKey: mode === "extract" ? "extract" : "discovery" });
          const safeStatusData = { ...statusData };
          if (safeStatusData.error === null) delete safeStatusData.error;
          if (safeStatusData.steps) delete safeStatusData.steps;
          if (safeStatusData.goal) delete safeStatusData.goal;
          if (safeStatusData.result) delete safeStatusData.result;
          if (safeStatusData.output) delete safeStatusData.output;
          console.log(`[TinyFish] Run ${runId} COMPLETED. Data:`, JSON.stringify(safeStatusData));
          result = statusData.result || statusData.output;
          // Retirer immédiatement de l'UI : les anciens agents ne s'affichent plus
          activeRuns.delete(runId);
        } else if (status === "FAILED") {
          const safeStatusData = { ...statusData };
          if (safeStatusData.steps) delete safeStatusData.steps;
          if (safeStatusData.goal) delete safeStatusData.goal;
          if (safeStatusData.result) delete safeStatusData.result;
          if (safeStatusData.output) delete safeStatusData.output;
          console.error(`[TinyFish] Run ${runId} FAILED. Data:`, JSON.stringify(safeStatusData));
          const errorMsg = typeof statusData.error === 'object' && statusData.error !== null
            ? (statusData.error.message || JSON.stringify(statusData.error))
            : statusData.error;
          throw new Error(errorMsg || "TinyFish run failed");
        }
      } catch (pollError: any) {
        console.error(`[TinyFish] Error polling status for run ${runId}:`, pollError.message);
        // Don't throw here, let the loop continue and potentially timeout if it's a transient network error
      }
    }

    let projectsFound = 0;
    let rawResponse = "";
    if (result) {
      rawResponse = typeof result === 'string' ? result : JSON.stringify(result);
      console.log(`[TinyFish] Received result for ${targetUrl}`);

      if (mode === 'extract') {
        if (swarmStopped) return 0;
        try {
          // Parse the JSON result directly
          let projectData = result;
          if (typeof result === 'string') {
            const jsonMatch = result.match(/```json\s*([\s\S]*?)\s*```/) || result.match(/```\s*([\s\S]*?)\s*```/);
            const cleaned = jsonMatch ? jsonMatch[1] : result.trim();
            projectData = JSON.parse(cleaned);
          }
          if (projectData.marine_relevance === undefined) projectData.marine_relevance = 0.5;
          if (projectData.location_type === undefined) projectData.location_type = "unknown";

          if (swarmStopped) return 0;
          const gatekeeper = passesMarineGatekeeper(projectData, taskConfig?.gatekeeper);
          if (!gatekeeper.pass) {
            console.log(`[Gatekeeper] REJECTED (extract mode) ${targetUrl}: ${gatekeeper.reason}`);
            try {
              db.prepare(`
                INSERT INTO failed_extractions (target_url, project_url, error_message)
                VALUES (?, ?, ?)
                ON CONFLICT(project_url) DO UPDATE SET
                  error_message = excluded.error_message,
                  created_at = CURRENT_TIMESTAMP
              `).run(targetUrl, targetUrl, gatekeeper.reason!);
            } catch (dbErr) { }
            throw new Error(gatekeeper.reason);
          }

          const relevanceScore = typeof projectData.marine_relevance === "number" ? projectData.marine_relevance : 0.95;
          const upsertResult = await upsertProject({
            title: projectData.title || 'Unknown',
            url: projectData.url || targetUrl,
            description: projectData.description || '',
            funder: projectData.funder || 'Unknown',
            lat: projectData.lat || 0,
            lng: projectData.lng || 0,
            category: projectData.category || 'Marine Conservation',
            status: projectData.status || 'Active',
            image_url: projectData.image_url || '',
            start_date: projectData.start_date || null,
            end_date: projectData.end_date || null,
            relevance_score: relevanceScore,
            s_ocean_score: projectData.s_ocean_score ?? 0.75,
          });

          try {
            db.prepare(`DELETE FROM failed_extractions WHERE project_url = ?`).run(targetUrl);
          } catch (e) { }

          projectsFound = upsertResult !== "skipped" ? 1 : 0;
          if (upsertResult !== "skipped") {
            broadcastLog({ key: "etl_saved", title: (projectData.title || "?").slice(0, 35) });
          }
          broadcastLog({ key: projectsFound === 1 ? "etl_extract_one" : "etl_extract_many", n: projectsFound });
          console.log(`[TinyFish] Extraction mode finished successfully for ${targetUrl}`);
        } catch (err: any) {
          console.error(`[TinyFish] Error parsing extraction result for ${targetUrl}:`, err.message);
          throw new Error(`Failed to parse extraction result: ${err.message}`);
        }
      } else {
        const urls = parseUrlsData(result);
        const currentRunAfterDiscover = activeRuns.get(runId);
        if (swarmStopped || currentRunAfterDiscover?.aborted) {
          console.log(`[TinyFish] Run ${runId} stopped before extraction. Skipping hybrid extraction.`);
          activeRuns.delete(runId);
          return;
        }
        const extractConcurrency = taskConfig?.extraction?.concurrency ?? EXTRACT_CONCURRENCY;
        broadcastLog({ key: "etl_cache_updated", n: urls.length });
        broadcastLog({ key: "etl_pipeline", n: urls.length, c: extractConcurrency });
        console.log(`[TinyFish] Discovered ${urls.length} URLs for ${targetUrl}. Starting hybrid extraction in background (slot freed for next agent)...`);
        appendToDeepLinkCache("DeepLinkCacheProjectsLists.json", [targetUrl]);
        if (urls.length > 0) appendToDeepLinkCache("DeepLinkCacheProjectsPages.json", urls);

        activeRuns.delete(runId);
        extractingCount++;
        hybridExtractCounter++;
        const hybridId = `hybrid-${hybridExtractCounter}`;
        const hybridLabel = `Readability.js ${hybridExtractCounter}`;
        activeHybridExtractions.set(hybridId, { targetUrl, agentLabel: hybridLabel, totalUrls: urls.length });
        (async () => {
          const discoverStartTime = startTime;
          try {
            const successCount = await runWithConcurrency(urls, extractConcurrency, async (projectUrl, i) => {
              if (swarmStopped) return 0;
              try {
                const host = (() => { try { return new URL(projectUrl).hostname; } catch { return projectUrl.slice(0, 30); } })();
                broadcastLog({ key: "etl_fetch_n", i: i + 1, total: urls.length, host });
                const { markdown, method, imageUrls } = await fetchMarkdown(projectUrl);
                const projectData = await extractProjectData(markdown, projectUrl, taskConfig?.extraction, imageUrls);

                const gatekeeper = passesMarineGatekeeper(projectData, taskConfig?.gatekeeper);
                if (!gatekeeper.pass) {
                  console.log(`[Gatekeeper] REJECTED ${projectUrl}: ${gatekeeper.reason}`);
                  try {
                    db.prepare(`
                      INSERT INTO failed_extractions (target_url, project_url, error_message)
                      VALUES (?, ?, ?)
                      ON CONFLICT(project_url) DO UPDATE SET
                        error_message = excluded.error_message,
                        created_at = CURRENT_TIMESTAMP
                    `).run(targetUrl, projectUrl, gatekeeper.reason!);
                  } catch (dbErr) { }
                  return 0;
                }

                if (swarmStopped) return 0;
                const relevanceScore = typeof projectData.marine_relevance === "number" ? projectData.marine_relevance : 0.95;
                broadcastLog({ key: "etl_claude_extract", title: (projectData.title || "?").slice(0, 40) });
                const upsertResult = await upsertProject({
                  title: projectData.title || 'Unknown',
                  url: projectData.url || projectUrl,
                  description: projectData.description || '',
                  funder: projectData.funder || 'Unknown',
                  lat: projectData.lat || 0,
                  lng: projectData.lng || 0,
                  category: projectData.category || 'Marine Conservation',
                  status: projectData.status || 'Active',
                  image_url: projectData.image_url || '',
                  start_date: projectData.start_date || null,
                  end_date: projectData.end_date || null,
                  relevance_score: relevanceScore,
                  s_ocean_score: projectData.s_ocean_score ?? 0.75,
                });

                try {
                  db.prepare(`DELETE FROM failed_extractions WHERE project_url = ?`).run(projectUrl);
                } catch (e) { }

                if (upsertResult !== "skipped") {
                  broadcastLog({ key: "etl_saved", title: (projectData.title || "?").slice(0, 35) });
                }
                console.log(`[Extraction] ${i + 1}/${urls.length} URL traitée via (${method}) [marine=${relevanceScore.toFixed(2)}]${upsertResult === "updated" ? " [dédoublonné]" : ""}: ${projectUrl}`);
                return upsertResult !== "skipped" ? 1 : 0;
              } catch (err: any) {
                console.error(`[Extraction] Error processing ${projectUrl}:`, err.message);
                try {
                  db.prepare(`
                    INSERT INTO failed_extractions (target_url, project_url, error_message)
                    VALUES (?, ?, ?)
                    ON CONFLICT(project_url) DO UPDATE SET
                      error_message = excluded.error_message,
                      created_at = CURRENT_TIMESTAMP
                  `).run(targetUrl, projectUrl, err.message);
                } catch (dbErr) { }
                return 0;
              }
            }, () => swarmStopped);

            const duration = Math.round(performance.now() - discoverStartTime);
            recordTelemetry('tinyfish', targetUrl, 'SUCCESS', successCount, duration, null, rawResponse);
            broadcastLog({ key: successCount === 1 ? "etl_extract_one" : "etl_extract_many", n: successCount });
            console.log(`[TinyFish] Hybrid extraction finished. Saved ${successCount} projects for ${targetUrl}`);
          } catch (err: any) {
            console.error(`[TinyFish] Hybrid extraction error for ${targetUrl}:`, err.message);
            recordTelemetry('tinyfish', targetUrl, 'ERROR', 0, Math.round(performance.now() - discoverStartTime), err.message);
          } finally {
            activeHybridExtractions.delete(hybridId);
            extractingCount--;
            checkSwarmComplete();
          }
        })();
        return;
      }
    }

    const duration = Math.round(performance.now() - startTime);
    recordTelemetry('tinyfish', targetUrl, 'SUCCESS', projectsFound, duration, null, rawResponse);
    activeRuns.delete(runId);
  } catch (error: any) {
    console.error(`[TinyFish] Error for ${targetUrl}:`, error.message);
    const duration = Math.round(performance.now() - startTime);

    // Auto-retry once with standard profile if it failed early or timed out
    if (retryCount < 1 && !error.message.includes("aborted")) {
      console.log(`[TinyFish] Retrying ${targetUrl} due to error...`);
      if (runId) activeRuns.delete(runId);
      return runTinyFishAgent(targetUrl, proxy, retryCount + 1, mode, taskConfig);
    }

    recordTelemetry('tinyfish', targetUrl, 'ERROR', 0, duration, error.message);
    if (runId) activeRuns.delete(runId);
    throw error;
  }
}

// Log stream for frontend (informative process logs)
// Supports: broadcastLog("raw") or broadcastLog({ key: "etl_fetch", host: "x.com" })
const logStreamSubscribers: { res: express.Response }[] = [];
function broadcastLog(msg: string | Record<string, unknown>) {
  const payload = typeof msg === "string" ? { message: msg } : { ...msg };
  const line = JSON.stringify(payload) + "\n";
  for (let i = logStreamSubscribers.length - 1; i >= 0; i--) {
    try {
      logStreamSubscribers[i].res.write(`data: ${line}`);
    } catch (_) {
      logStreamSubscribers.splice(i, 1);
    }
  }
}

// NDJSON live feed: subscribers receive new projects as they're inserted
const projectStreamSubscribers: { res: express.Response }[] = [];
function broadcastNewProject(feature: object) {
  if (swarmStopped) return;
  const line = JSON.stringify(feature) + "\n";
  for (let i = projectStreamSubscribers.length - 1; i >= 0; i--) {
    try {
      projectStreamSubscribers[i].res.write(`data: ${line}`);
    } catch (_) {
      projectStreamSubscribers.splice(i, 1);
    }
  }
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // CORS + frame-ancestors: permet au flux TinyFish (inspector) d'afficher l'app sans blocage
  app.use((_req, res, next) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Content-Security-Policy", "frame-ancestors 'self' https://tetra-streaming.tinyfish.io https://*.tinyfish.io");
    next();
  });
  app.options("*", (_req, res) => res.sendStatus(204));

  app.get("/favicon.ico", (_req, res) => {
    const svgPath = path.join(__dirname, "dist", "logo-blue-intelligence.svg");
    if (fs.existsSync(svgPath)) {
      res.setHeader("Content-Type", "image/svg+xml");
      res.sendFile(svgPath);
    } else {
      res.status(404).end();
    }
  });

  const DEBUG_API = process.env.DEBUG_API === "1" || process.env.DEBUG_API === "true";
  app.use((req, res, next) => {
    if (DEBUG_API && req.path.startsWith("/api/")) {
      const start = Date.now();
      res.on("finish", () => {
        const duration = Date.now() - start;
        if (res.statusCode >= 400 || duration > 500) {
          console.log(`[API] ${req.method} ${req.path} ${res.statusCode} ${duration}ms`);
        }
      });
    }
    next();
  });

  // API Routes
  app.get("/api/debug/db", (req, res) => {
    try {
      const projects = db.prepare("SELECT * FROM projects").all();
      const telemetry = db.prepare("SELECT * FROM telemetry").all();
      res.json({ projects, telemetry });
    } catch (error) {
      res.status(500).json({ error: String(error) });
    }
  });

  app.get("/api/config-check", (req, res) => {
    res.json({
      tinyfishKeySet: !!process.env.TINYFISH_API_KEY,
      claudeKeySet: !!(process.env.CLAUDE_API_KEY || process.env.ANTHROPIC_API_KEY),
      envKeys: Object.keys(process.env).filter(k => k.includes('FISH') || k.includes('API') || k.includes('KEY') || k.includes('TINY') || k.includes('CLAUDE') || k.includes('ANTHROPIC'))
    });
  });

  app.get("/api/projects", (req, res) => {
    try {
      const projects = db.prepare("SELECT * FROM projects").all();

      const geojson = {
        type: "FeatureCollection",
        features: projects.map((p: any) => ({
          type: "Feature",
          geometry: {
            type: "Point",
            coordinates: [p.lng, p.lat],
          },
          properties: {
            id: p.id,
            title: p.title,
            url: p.url,
            description: p.description,
            funder: p.funder,
            relevance_score: p.relevance_score,
            s_ocean_score: p.s_ocean_score,
            category: p.category,
            status: p.status,
            image_url: p.image_url,
            start_date: p.start_date,
            end_date: p.end_date,
          },
        })),
      };

      res.json(geojson);
    } catch (error) {
      console.error("Error fetching projects:", error);
      res.status(500).json({ error: "Failed to fetch projects" });
    }
  });

  function hasValidImageUrl(v: string | null | undefined): boolean {
    if (v == null) return false;
    const t = String(v).trim();
    if (t === "" || t === "null" || t === "undefined") return false;
    if (!t.startsWith("http://") && !t.startsWith("https://")) return false;
    if (t.length < 12) return false; // "https://x.co" minimum
    return true;
  }

  app.get("/api/projects/without-image", (req, res) => {
    try {
      const all = db.prepare("SELECT id, title, url, funder, image_url FROM projects ORDER BY id DESC").all() as { id: number; title: string; url: string; funder: string; image_url: string | null }[];
      const rows = all.filter((p) => !hasValidImageUrl(p.image_url)).map(({ id, title, url, funder }) => ({ id, title, url, funder }));
      res.json(rows);
    } catch (error) {
      console.error("Error fetching projects without image:", error);
      res.status(500).json({ error: "Failed to fetch" });
    }
  });

  app.get("/api/projects/image-stats", (req, res) => {
    try {
      const all = db.prepare("SELECT id, image_url FROM projects").all() as { id: number; image_url: string | null }[];
      const without = all.filter((p) => !hasValidImageUrl(p.image_url));
      const samples = without.slice(0, 5).map((p) => ({ id: p.id, image_url: p.image_url, repr: JSON.stringify(p.image_url) }));
      res.json({ total: all.length, withoutImage: without.length, samples });
    } catch (error) {
      console.error("Error fetching image stats:", error);
      res.status(500).json({ error: "Failed to fetch" });
    }
  });

  app.get("/api/projects/stream", (req, res) => {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    projectStreamSubscribers.push({ res });
    req.on("close", () => {
      const idx = projectStreamSubscribers.findIndex((s) => s.res === res);
      if (idx >= 0) projectStreamSubscribers.splice(idx, 1);
    });
  });

  app.get("/api/logs/stream", (req, res) => {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    logStreamSubscribers.push({ res });
    req.on("close", () => {
      const idx = logStreamSubscribers.findIndex((s) => s.res === res);
      if (idx >= 0) logStreamSubscribers.splice(idx, 1);
    });
  });

  app.get("/api/test-project", async (req, res) => {
    try {
      // Create a fake project
      const project = {
        title: "Coral Reef Restoration Maldives",
        description: "Project restoring coral reefs in Maldives waters",
        lat: 3.2,
        lng: 73.0,
      };

      // Force insertion even if the swarm is stopped (test mode)
      const oldSwarmStopped = swarmStopped;
      console.log("[Test] /api/test-project called (swarmStopped=", swarmStopped, ")");
      swarmStopped = false;
      const result = await upsertProject(project);
      swarmStopped = oldSwarmStopped;
      console.log("[Test] /api/test-project result=", result, "(swarmStopped restored to", swarmStopped, ")");

      res.setHeader("Content-Type", "application/json");
      res.send(JSON.stringify({ success: true, project, result, swarmStopped: oldSwarmStopped }));
    } catch (err: any) {
      console.error("[Test] /api/test-project error:", err);
      res.setHeader("Content-Type", "application/json");
      res.status(500).send(JSON.stringify({ success: false, error: String(err), stack: err?.stack }));
    }
  });

  app.get("/api/test-project-2", async (req, res) => {
    try {
      // Create a semantically similar project (same location)
      const project = {
        title: "Maldives Coral Reef Recovery",
        description: "Local coral reef recovery program in the Maldives",
        lat: 3.2,
        lng: 73.0,
        url: "https://example.com/project2",
        funder: "Ocean Protectors",
        category: "Marine",
        status: "Active",
      };

      // Force insertion even if the swarm is stopped (test mode)
      const oldSwarmStopped = swarmStopped;
      console.log("[Test] /api/test-project-2 called (swarmStopped=", swarmStopped, ")");
      swarmStopped = false;
      const result = await upsertProject(project);
      swarmStopped = oldSwarmStopped;
      console.log("[Test] /api/test-project-2 result=", result, "(swarmStopped restored to", swarmStopped, ")");

      // Query nearby records and compute similarity stats for debugging
      const candidates = db
        .prepare("SELECT id, title, url, description, lat, lng, title_embedding, description_embedding FROM projects WHERE abs(lat - ?) < 0.01 AND abs(lng - ?) < 0.01")
        .all(project.lat, project.lng) as Array<{
          id: number;
          title: string;
          url: string;
          description: string;
          lat: number;
          lng: number;
          title_embedding?: string;
          description_embedding?: string;
        }>;

      const { titleEmbedding: queryTitleEmb, descriptionEmbedding: queryDescEmb } = await computeEmbeddingsForProject({
        title: project.title,
        description: project.description,
      });

      const similarities = await Promise.all(
        candidates.map(async (c) => {
          let candidateTitleEmb = parseEmbedding(c.title_embedding);
          let candidateDescEmb = parseEmbedding(c.description_embedding);

          // If embeddings are missing, compute them for debug insight
          if (!candidateTitleEmb || (c.description && !candidateDescEmb)) {
            const computed = await computeEmbeddingsForProject({ title: c.title, description: c.description });
            candidateTitleEmb = candidateTitleEmb || computed.titleEmbedding;
            candidateDescEmb = candidateDescEmb || computed.descriptionEmbedding;
          }

          const titleSim = cosineSimilarity(queryTitleEmb, candidateTitleEmb);
          const hasDesc = queryDescEmb && candidateDescEmb;
          const descSim = hasDesc ? cosineSimilarity(queryDescEmb, candidateDescEmb) : null;
          return {
            id: c.id,
            url: c.url,
            title: c.title,
            description: c.description,
            titleSim,
            descSim,
          };
        })
      );

      res.setHeader("Content-Type", "application/json");
      res.send(JSON.stringify({ success: true, project, result, nearby: similarities }));
    } catch (err: any) {
      console.error("[Test] /api/test-project-2 error:", err);
      res.setHeader("Content-Type", "application/json");
      res.status(500).send(JSON.stringify({ success: false, error: String(err), stack: err?.stack }));
    }
  });

  app.get("/api/projects/ndjson", (req, res) => {
    res.setHeader("Content-Type", "application/x-ndjson");
    const projects = db.prepare("SELECT * FROM projects").all() as any[];
    for (const p of projects) {
      res.write(JSON.stringify({
        type: "Feature",
        geometry: { type: "Point", coordinates: [p.lng, p.lat] },
        properties: { id: p.id, title: p.title, url: p.url, description: p.description, funder: p.funder, relevance_score: p.relevance_score, s_ocean_score: p.s_ocean_score, category: p.category, status: p.status, image_url: p.image_url, start_date: p.start_date, end_date: p.end_date }
      }) + "\n");
    }
    res.end();
  });

  app.post("/api/projects/clear", (req, res) => {
    try {
      swarmStopped = true;
      deployStartTime = null;
      agentQueue.length = 0;
      activeAgents = 0;
      agentCounter = 0;
      activeRuns.clear();
      activeExtractRuns.clear();
      activeHybridExtractions.clear();
      db.prepare("DELETE FROM projects").run();
      broadcastLog({ key: "etl_db_cleared" });
      res.json({ status: "ok", message: "All projects cleared" });
    } catch (error) {
      console.error("Error clearing projects:", error);
      res.status(500).json({ error: "Failed to clear projects" });
    }
  });

  app.post("/api/audit/clear", (req, res) => {
    try {
      db.prepare("DELETE FROM telemetry").run();
      db.prepare("DELETE FROM failed_extractions").run();
      broadcastLog({ key: "etl_audit_cleared" });
      res.json({ status: "ok", message: "Audit data cleared (telemetry + failed extractions)" });
    } catch (error) {
      console.error("Error clearing audit:", error);
      res.status(500).json({ error: "Failed to clear audit data" });
    }
  });

  app.get("/api/telemetry", (req, res) => {
    try {
      const telemetry = db.prepare("SELECT * FROM telemetry ORDER BY created_at DESC LIMIT 100").all();
      res.json(telemetry);
    } catch (error) {
      console.error("Error fetching telemetry:", error);
      res.status(500).json({ error: "Failed to fetch telemetry" });
    }
  });

  app.get("/api/failed-extractions", (req, res) => {
    try {
      const failed = db.prepare("SELECT * FROM failed_extractions ORDER BY created_at DESC").all();
      res.json(failed);
    } catch (error) {
      console.error("Error fetching failed extractions:", error);
      res.status(500).json({ error: "Failed to fetch failed extractions" });
    }
  });

  app.get("/api/config/defaults", (_req, res) => {
    res.json({
      gatekeeper: {
        marine_threshold: DEFAULT_GATEKEEPER.marine_threshold,
        inland_threshold: DEFAULT_GATEKEEPER.inland_threshold,
        coast_distance_km: DEFAULT_GATEKEEPER.coast_distance_km,
      },
    });
  });

  app.post("/api/agent/force-extract", async (req, res) => {
    const { projectUrls, proxy, config } = req.body;

    if (!projectUrls || !Array.isArray(projectUrls) || projectUrls.length === 0) {
      return res.status(400).json({ error: "projectUrls array is required" });
    }
    const tinyfishKey = process.env.TINYFISH_API_KEY;
    if (!tinyfishKey) {
      return res.status(500).json({ error: "TINYFISH_API_KEY is required for force-extract (re-extraction uses TinyFish to handle PDFs and pages that Readability cannot parse)" });
    }
    // Re-extraction via TinyFish : gère PDFs, 404, pages que Readability ne peut pas parser
    swarmStopped = false;
    const taskConfig = config as TaskConfig | undefined;
    maxConcurrentAgents = Math.max(1, Math.min(10, taskConfig?.agent?.maxConcurrentAgents ?? 2));
    for (const url of projectUrls) {
      if (!agentQueue.find(t => t.url === url)) {
        agentQueue.push({ url, proxy, mode: 'extract', useTinyFish: true, config: taskConfig });
      }
    }
    processQueue();

    res.json({ status: "QUEUED", message: `${projectUrls.length} agents deployed in background for forced extraction (TinyFish)` });
  });

  app.get("/api/etl/seeds", (req, res) => {
    try {
      const seeds = loadMasterSeeds();
      res.json(seeds);
    } catch (e) {
      res.status(500).json({ error: "Failed to load MasterSeeds" });
    }
  });

  app.get("/api/etl/caches", (req, res) => {
    try {
      const lists = loadDeepLinkCache("DeepLinkCacheProjectsLists.json");
      const pages = loadDeepLinkCache("DeepLinkCacheProjectsPages.json");
      res.json({ lists: lists.urls, pages: pages.urls });
    } catch (e) {
      res.status(500).json({ error: "Failed to load caches" });
    }
  });

  app.post("/api/etl/swarm-deploy", async (req, res) => {
    const { clearBeforeStart, testMode, proxy, config } = req.body || {};
    const tinyfishKey = process.env.TINYFISH_API_KEY;
    if (!tinyfishKey) {
      return res.status(500).json({ error: "TINYFISH_API_KEY is not configured" });
    }
    swarmStopped = false;
    deployStartTime = Date.now();
    maxConcurrentAgents = Math.max(1, Math.min(10, config?.agent?.maxConcurrentAgents ?? 2));
    broadcastLog({ key: "etl_deploy_start" });
    agentQueue.length = 0;
    activeAgents = 0; // Reset so processQueue can start new tasks immediately
    agentCounter = 0;
    extractCounter = 0;
    hybridExtractCounter = 0;
    activeRuns.clear();
    activeExtractRuns.clear();
    activeHybridExtractions.clear();
    if (clearBeforeStart) {
      db.prepare("DELETE FROM projects").run();
      broadcastLog({ key: "etl_db_cleared" });
    }
    const seeds = loadMasterSeeds();
    const lists = loadDeepLinkCache("DeepLinkCacheProjectsLists.json");
    const pages = loadDeepLinkCache("DeepLinkCacheProjectsPages.json");
    broadcastLog({ key: "etl_master_seeds", n: seeds.length });
    broadcastLog({ key: "etl_deeplink", lists: lists.urls.length, pages: pages.urls.length });
    const existingUrls = new Set(
      (db.prepare("SELECT url FROM projects WHERE url IS NOT NULL").all() as { url: string }[]).map(r => r.url)
    );
    const discoverUrlsFull = [...new Set([
      ...seeds.map((s: { url: string }) => s.url),
      ...lists.urls
    ])];
    const discoverUrls = testMode
      ? seeds.slice(0, Math.max(2, maxConcurrentAgents)).map((s: { url: string }) => s.url)
      : discoverUrlsFull;
    const extractUrls = pages.urls.filter((u: string) => !existingUrls.has(u));
    const taskConfig = config as TaskConfig | undefined;
    const toEnqueue: { url: string; proxy?: string; mode: "discover" | "extract"; config?: TaskConfig }[] = [];
    const discoverLimit = testMode ? Math.max(2, maxConcurrentAgents) : discoverUrls.length;
    const extractLimit = testMode ? 0 : extractUrls.length;
    for (let i = 0; i < Math.min(discoverUrls.length, discoverLimit); i++) {
      toEnqueue.push({ url: discoverUrls[i], proxy, mode: "discover", config: taskConfig });
    }
    for (let i = 0; i < Math.min(extractUrls.length, extractLimit); i++) {
      toEnqueue.push({ url: extractUrls[i], proxy, mode: "extract", config: taskConfig });
    }
    const nDiscover = toEnqueue.filter(t => t.mode === "discover").length;
    const nExtract = toEnqueue.filter(t => t.mode === "extract").length;
    broadcastLog({ key: "etl_queue", discover: nDiscover, extract: nExtract, total: toEnqueue.length });
    for (const t of toEnqueue) {
      agentQueue.push(t);
    }
    broadcastLog({ key: "etl_enqueued", n: toEnqueue.length });
    console.log(`[Swarm] Deploy: ${toEnqueue.length} tasks enqueued (${nDiscover} discover, ${nExtract} extract)`);
    processQueue();
    res.json({
      status: "QUEUED",
      message: `ETL Swarm deployed. ${toEnqueue.length} tasks enqueued (${discoverUrls.length} discover, ${extractUrls.length} extract available).`,
      enqueued: toEnqueue.length,
      discoverAvailable: discoverUrls.length,
      extractAvailable: extractUrls.length
    });
  });

  app.post("/api/agent/start", async (req, res) => {
    const { targetUrl, proxy, mode, config } = req.body;

    if (!targetUrl) {
      return res.status(400).json({ error: "targetUrl is required" });
    }
    if (swarmStopped) {
      return res.status(409).json({ error: "Swarm is stopped. Deploy first." });
    }

    const tinyfishKey = process.env.TINYFISH_API_KEY;
    if (!tinyfishKey) {
      return res.status(500).json({ error: "TINYFISH_API_KEY is not configured" });
    }

    const taskConfig = config as TaskConfig | undefined;
    const taskMode = mode === "extract" ? "extract" : "discover";
    if (!agentQueue.find(t => t.url === targetUrl)) {
      agentQueue.push({ url: targetUrl, proxy, mode: taskMode, config: taskConfig });
      processQueue();
    }

    res.json({ status: "QUEUED", message: `Agent deployed (${taskMode})` });
  });

  app.get("/api/agent/active-runs", (req, res) => {
    const tinyFishRuns = swarmStopped ? [] : Array.from(activeRuns.entries())
      .filter(([, data]) => !data.aborted && (data.status === "RUNNING" || data.status === "PENDING"))
      .map(([id, data]) => ({
        id,
        agentLabel: data.agentLabel || "TinyFish",
        streamingUrl: data.streamingUrl,
        status: data.status,
        targetUrl: data.targetUrl || null,
        mode: data.mode || "discover"
      }));
    const extractRuns = Array.from(activeExtractRuns.entries()).map(([id, data]) => ({
      id,
      agentLabel: data.agentLabel,
      streamingUrl: null,
      status: data.status,
      targetUrl: data.targetUrl,
      mode: "extract"
    }));
    const hybridRuns = Array.from(activeHybridExtractions.entries()).map(([id, data]) => ({
      id,
      agentLabel: data.agentLabel,
      streamingUrl: null,
      status: "RUNNING",
      targetUrl: data.targetUrl,
      mode: "extract",
      totalUrls: data.totalUrls
    }));
    res.json([...tinyFishRuns, ...extractRuns, ...hybridRuns]);
  });

  app.get("/api/agent/stream/:runId", async (req, res) => {
    const { runId } = req.params;
    const run = activeRuns.get(runId);
    if (!run || !run.streamingUrl) {
      return res.status(404).json({ error: "Run or streaming URL not found" });
    }

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const apiKey = process.env.TINYFISH_API_KEY;
    const sseResponse = await fetch(run.streamingUrl, {
      headers: { "X-API-Key": apiKey! }
    });

    if (!sseResponse.body) return res.end();

    const reader = sseResponse.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      res.write(chunk);
    }
    res.end();
  });

  app.post("/api/agent/stop", (req, res) => {
    broadcastLog({ key: "swarm_stop" });
    swarmStopped = true;
    deployStartTime = null;
    agentQueue.length = 0;
    agentCounter = 0;
    activeAgents = 0;
    activeRuns.clear();
    activeExtractRuns.clear();
    activeHybridExtractions.clear();
    res.json({ status: "STOPPED", message: "Agent queue cleared and active runs signaled to stop" });
  });

  app.get("/api/agent/status", (req, res) => {
    res.json({ activeAgents, queuedAgents: agentQueue.length });
  });

  app.get("/api/proxy-image", async (req, res) => {
    let imageUrl = req.query.url as string;
    if (!imageUrl) return res.status(400).send("No URL provided");

    // Handle relative URLs just in case
    if (imageUrl.startsWith('/')) {
      return res.status(400).send("Invalid relative URL. Must be absolute.");
    }

    try {
      const response = await fetch(imageUrl, {
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
          "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
          "Accept-Language": "en-US,en;q=0.9",
          "Referer": new URL(imageUrl).origin
        }
      });

      if (!response.ok) {
        console.error(`[Proxy Image] Failed to fetch ${imageUrl}: ${response.status} ${response.statusText}`);
        throw new Error("Failed to fetch image");
      }

      const contentType = response.headers.get("content-type");
      if (contentType) res.setHeader("Content-Type", contentType);

      // Cache for 24 hours
      res.setHeader("Cache-Control", "public, max-age=86400");

      const arrayBuffer = await response.arrayBuffer();
      res.end(Buffer.from(arrayBuffer));
    } catch (error) {
      console.error(`[Proxy Image] Error fetching ${imageUrl}:`, error);
      res.status(404).send("Image not found");
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, "dist")));
  }


  // --- DEDUPLICATION TEST FUNCTION ---
  async function runDedupTest() {
    console.log("[TEST] Running deduplication test...");

    const testProjects = [
      {
        title: "Coral Reef Restoration Maldives",
        description: "Project restoring coral reefs in Maldives waters",
        lat: 3.2,
        lng: 73.0,
        url: "https://example.com/project1",
        funder: "Blue Ocean Foundation",
        category: "Marine",
        status: "Active"
      },
      {
        title: "Coral Reef Restoration Maldives",
        description: "Project restoring coral reefs in Maldives waters",
        lat: 3.2001,
        lng: 73.0002,
        url: "https://example.com/project2",
        funder: "Ocean Protectors",
        category: "Marine",
        status: "Active"
      }
    ];

    const savedCount = await saveProjects(testProjects);
    console.log(`[TEST] Saved ${savedCount} projects (deduplicated)`);
  }
  // Uncomment to test manually
  // await runDedupTest();

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
