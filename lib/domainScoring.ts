import * as fs from 'fs';
import * as path from 'path';

export interface DomainMetrics {
  domain: string;
  totalProjects: number;
  validProjects: number;
  reliabilityScore: number;
  lastUpdated: number;
}

interface DomainMetricsStore {
  version: number;
  lastUpdated: number;
  metrics: Record<string, DomainMetrics>;
}

export class DomainScoringEngine {
  private readonly storePath: string;
  private metricsCache: Map<string, DomainMetrics> = new Map();

  constructor(storePath: string = 'data/domain-metrics.json') {
    this.storePath = storePath;
    this.loadMetrics();
  }

  private loadMetrics(): void {
    try {
      if (fs.existsSync(this.storePath)) {
        const data = fs.readFileSync(this.storePath, 'utf-8');
        const store: DomainMetricsStore = JSON.parse(data);
        this.metricsCache.clear();
        Object.values(store.metrics).forEach((metric) => {
          this.metricsCache.set(metric.domain, metric);
        });
      }
    } catch (error) {
      console.warn(`[DomainScoring] Failed to load metrics: ${error}`);
      this.metricsCache.clear();
    }
  }

  private saveMetrics(): void {
    try {
      const dir = path.dirname(this.storePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }

      const store: DomainMetricsStore = {
        version: 1,
        lastUpdated: Date.now(),
        metrics: Object.fromEntries(this.metricsCache),
      };

      const tempPath = `${this.storePath}.tmp`;
      fs.writeFileSync(tempPath, JSON.stringify(store, null, 2), 'utf-8');
      fs.renameSync(tempPath, this.storePath);
    } catch (error) {
      console.error(`[DomainScoring] Failed to save metrics: ${error}`);
    }
  }

  extractDomain(url: string): string | null {
    try {
      const urlObj = new URL(url);
      return urlObj.hostname.toLowerCase();
    } catch {
      return null;
    }
  }

  recordProject(url: string, isValid: boolean): void {
    const domain = this.extractDomain(url);
    if (!domain) return;

    const existing = this.metricsCache.get(domain) || {
      domain,
      totalProjects: 0,
      validProjects: 0,
      reliabilityScore: 0,
      lastUpdated: Date.now(),
    };

    existing.totalProjects += 1;
    if (isValid) existing.validProjects += 1;
    existing.reliabilityScore =
      existing.totalProjects > 0
        ? existing.validProjects / existing.totalProjects
        : 0;
    existing.lastUpdated = Date.now();

    this.metricsCache.set(domain, existing);
    if (this.metricsCache.size > 50000) {
      console.warn('[DomainScoring] Cache size exceeding 50k domains');
    }
    this.saveMetrics();
  }

  getSortedDomains(minProjects: number = 1): DomainMetrics[] {
    return Array.from(this.metricsCache.values())
      .filter((m) => m.totalProjects >= minProjects)
      .sort((a, b) => {
        if (b.reliabilityScore !== a.reliabilityScore) {
          return b.reliabilityScore - a.reliabilityScore;
        }
        return b.totalProjects - a.totalProjects;
      });
  }

  getDomainMetrics(domain: string): DomainMetrics | null {
    return this.metricsCache.get(domain.toLowerCase()) || null;
  }

  getAllMetrics(): DomainMetrics[] {
    return Array.from(this.metricsCache.values());
  }

  getWeightedScore(domain: string): number {
    const metrics = this.getDomainMetrics(domain);
    if (!metrics) return 0;

    const n = metrics.totalProjects;
    const p = metrics.reliabilityScore;
    const z = 1.96;

    if (n === 0) return 0;

    const denominator = 1 + (z * z) / n;
    const center = (p + (z * z) / (2 * n)) / denominator;
    const margin = (z * Math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))) / denominator;

    return Math.max(0, center - margin);
  }

  getTopDomains(limit: number = 10): DomainMetrics[] {
    return this.getAllMetrics()
      .map((m) => ({
        ...m,
        weightedScore: this.getWeightedScore(m.domain),
      }))
      .sort((a, b) => (b as any).weightedScore - (a as any).weightedScore)
      .slice(0, limit)
      .map(({ weightedScore, ...m }) => m);
  }

  resetMetrics(): void {
    this.metricsCache.clear();
    if (fs.existsSync(this.storePath)) {
      fs.unlinkSync(this.storePath);
    }
  }
}

let engineInstance: DomainScoringEngine | null = null;

export function initializeDomainScoring(storePath?: string): DomainScoringEngine {
  if (!engineInstance) {
    engineInstance = new DomainScoringEngine(storePath);
  }
  return engineInstance;
}

export function getDomainScoringEngine(): DomainScoringEngine {
  if (!engineInstance) {
    throw new Error('DomainScoringEngine not initialized. Call initializeDomainScoring() first.');
  }
  return engineInstance;
}