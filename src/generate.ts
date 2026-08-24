/**
 * Generator entrypoint. Read a toolkit catalog, infer its dependencies, write a graph.
 *
 * How we run it:
 *   - The path to a toolkit's catalog JSON is passed as a CLI ARGUMENT, e.g.
 *     `node --import tsx src/generate.ts path/to/catalog.json`. We append it as the last
 *     argument, so reading the final argv entry works whatever else your command carries.
 *   - Write your graph to `dependency_graph.json` in the working directory.
 *   - For LLM access, the OpenAI SDK reads OPENAI_API_KEY / OPENAI_BASE_URL from the
 *     environment (set from your assessment page's AI credentials; the same are provided
 *     when we run your generator). Use an OpenRouter model id such as `openai/gpt-4o`.
 *
 * This is a SKELETON. Replace the inference in generate() with your own approach. Do not
 * hardcode a toolkit's relations: your node ids must be slugs from the catalog you are
 * handed, and your output must change when the input changes.
 */
import { readFileSync, writeFileSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import OpenAI from "openai";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load .env locally if present (grader injects vars via environment directly).
if (existsSync(".env")) {
  for (const line of readFileSync(".env", "utf-8").split("\n")) {
    const m = line.match(/^([^#=]+)=(.*)$/);
    if (m) process.env[m[1].trim()] = m[2].trim();
  }
}

type Tool = Record<string, any>;
interface Node {
  id: string;
  service?: string;
}
interface Edge {
  from: string;
  to: string;
  label?: string;
}
interface Graph {
  nodes: Node[];
  edges: Edge[];
}
interface SlimField {
  name: string;
  description: string;
}
interface SlimTool {
  slug: string;
  requiredInputs: SlimField[];
  outputFields: SlimField[];
}

// The catalog path is the last CLI argument (we append it after your run command).
const CATALOG_PATH = process.argv.length > 2 ? process.argv[process.argv.length - 1] : undefined;
const OUT_PATH = "dependency_graph.json";

// Generic wrapper fields present on every tool's output — not meaningful for dependencies.
const SKIP_OUTPUT_FIELDS = new Set(["data", "error", "successful", "errors", "extensions"]);

function loadCatalog(): Tool[] {
  if (!CATALOG_PATH) {
    throw new Error("pass the toolkit catalog path as the first argument");
  }
  const data = JSON.parse(readFileSync(CATALOG_PATH, "utf-8"));
  return Array.isArray(data) ? data : (data.tools ?? data.items ?? []);
}

function slugOf(tool: Tool): string | undefined {
  return tool.slug ?? tool.name ?? tool.function?.name;
}

/** Follow a "#/$defs/Name" pointer into the local $defs map. */
function resolveRef(ref: string, defs: Record<string, any>): Record<string, any> | null {
  const match = ref.match(/^#\/\$defs\/(.+)$/);
  if (!match) return null;
  return defs[match[1]] ?? null;
}

/**
 * Recursively walk an output schema, resolving $refs, and collect leaf fields
 * (string / integer / number / boolean) with their names and descriptions.
 * Skips generic wrapper fields that every tool shares.
 */
function collectOutputFields(
  schema: Record<string, any>,
  defs: Record<string, any>,
  visited = new Set<string>(),
): SlimField[] {
  const fields: SlimField[] = [];
  const props: Record<string, any> = schema.properties ?? {};

  for (const [name, prop] of Object.entries(props)) {
    if (SKIP_OUTPUT_FIELDS.has(name)) {
      // Still follow $ref on skipped wrappers so we reach the real payload.
      if (prop.$ref && !visited.has(prop.$ref)) {
        visited.add(prop.$ref);
        const resolved = resolveRef(prop.$ref, defs);
        if (resolved) fields.push(...collectOutputFields(resolved, defs, visited));
      }
      continue;
    }

    if (prop.$ref && !visited.has(prop.$ref)) {
      visited.add(prop.$ref);
      const resolved = resolveRef(prop.$ref, defs);
      if (resolved) fields.push(...collectOutputFields(resolved, defs, visited));
    } else if (prop.type === "object") {
      fields.push(...collectOutputFields(prop, defs, visited));
    } else if (prop.type === "array" && prop.items) {
      const items = prop.items.$ref
        ? (() => {
            const r = resolveRef(prop.items.$ref, defs);
            return r ?? {};
          })()
        : prop.items;
      fields.push(...collectOutputFields(items, defs, visited));
    } else {
      // Leaf field — keep it.
      fields.push({ name, description: prop.description ?? "" });
    }
  }

  return fields;
}

/** Build a slim, dependency-relevant view of one tool. */
function extractSlimTool(tool: Tool): SlimTool | null {
  const slug = slugOf(tool);
  if (!slug) return null;

  const inputProps: Record<string, any> = tool.inputParameters?.properties ?? {};
  const required: string[] = tool.inputParameters?.required ?? [];
  const requiredInputs: SlimField[] = required.map((name) => ({
    name,
    description: inputProps[name]?.description ?? "",
  }));

  const outputSchema: Record<string, any> = tool.outputParameters ?? {};
  const defs: Record<string, any> = outputSchema.$defs ?? {};
  const outputFields = collectOutputFields(outputSchema, defs);

  return { slug, requiredInputs, outputFields };
}

function inferService(slug: string): string {
  const s = slug.replace(/^GITHUB_/, "");
  // More specific checks first to avoid wrong matches.
  if (s.includes("PULL_REQUEST") || s.includes("PULL")) return "pulls";
  if (s.includes("ISSUE"))        return "issues";
  if (s.includes("WORKFLOW") || s.includes("ACTION")) return "actions";
  if (s.includes("CODE_SCANNING") || s.includes("ADVISORY") || s.includes("SARIF")) return "security";
  if (s.includes("CHECK_RUN") || s.includes("CHECK_SUITE") || s.includes("STATUS_CHECK")) return "checks";
  if (s.includes("DEPLOY_KEY"))   return "deploy_keys";
  if (s.includes("DEPLOYMENT"))   return "deployments";
  if (s.includes("ENVIRONMENT"))  return "environments";
  if (s.includes("CODESPACE"))    return "codespaces";
  if (s.includes("CLASSROOM") || s.includes("ASSIGNMENT")) return "classroom";
  if (s.includes("SPONSOR"))      return "sponsors";
  if (s.includes("NOTIFICATION") || s.includes("THREAD")) return "notifications";
  if (s.includes("PAGES"))        return "pages";
  if (s.includes("ARTIFACT"))     return "artifacts";
  if (s.includes("PROJECT"))      return "projects";
  if (s.includes("RELEASE"))      return "releases";
  if (s.includes("BRANCH"))       return "branches";
  if (s.includes("COMMIT"))       return "commits";
  if (s.includes("REVIEW"))       return "reviews";
  if (s.includes("COMMENT"))      return "comments";
  if (s.includes("LABEL"))        return "labels";
  if (s.includes("MILESTONE"))    return "milestones";
  if (s.includes("DISCUSSION"))   return "discussions";
  if (s.includes("TEAM"))         return "teams";
  if (s.includes("ORG"))          return "orgs";
  if (s.includes("MIGRATION"))    return "migrations";
  if (s.includes("PACKAGE"))      return "packages";
  if (s.includes("WEBHOOK"))      return "webhooks";
  if (s.includes("REPOSITORY") || s.includes("REPO")) return "repos";
  if (s.includes("USER"))         return "users";
  if (s.includes("GIST"))         return "gists";
  if (s.includes("FORK"))         return "forks";
  if (s.includes("STAR"))         return "stars";
  return "other";
}

// Field names that are too generic to draw a reliable heuristic edge for.
// These will be handled by the LLM pass instead.
const SKIP_MATCH_FIELDS = new Set([
  "description", "visibility", "package_type", "content_type",
  "is_enabled", "permissions", "data_type",
]);

/** Returns true for field names specific enough to trust as heuristic matches. */
function isSpecificFieldName(name: string): boolean {
  if (SKIP_MATCH_FIELDS.has(name)) return false;
  if (name.length > 8) return true;
  const specific = ["_number", "_id", "_name", "_key", "_token", "_sha", "_ref"];
  return specific.some((suffix) => name.endsWith(suffix));
}

/** Heuristic pass: match required input names against output field names exactly,
 *  but only for field names specific enough to avoid false positives. */
function heuristicEdges(slim: SlimTool[]): Edge[] {
  // Build a map: field name -> list of tools that produce it
  const producers = new Map<string, string[]>();
  for (const tool of slim) {
    for (const field of tool.outputFields) {
      if (!isSpecificFieldName(field.name)) continue;
      const list = producers.get(field.name) ?? [];
      list.push(tool.slug);
      producers.set(field.name, list);
    }
  }

  const edges: Edge[] = [];
  for (const consumer of slim) {
    for (const input of consumer.requiredInputs) {
      if (!isSpecificFieldName(input.name)) continue;
      const producerSlugs = producers.get(input.name) ?? [];
      for (const producerSlug of producerSlugs) {
        if (producerSlug === consumer.slug) continue; // skip self-edges
        edges.push({ from: producerSlug, to: consumer.slug, label: input.name });
      }
    }
  }

  return edges;
}

// --- LLM cache ---
const CACHE_PATH = "llm_cache.json";
const USE_CACHE = !process.argv.includes("--no-cache");

function loadCache(): Edge[] | null {
  if (!USE_CACHE) return null;
  if (!existsSync(CACHE_PATH)) return null;
  try {
    return JSON.parse(readFileSync(CACHE_PATH, "utf-8"));
  } catch {
    return null;
  }
}

function saveCache(edges: Edge[]): void {
  writeFileSync(CACHE_PATH, JSON.stringify(edges, null, 2), "utf-8");
  console.error(`LLM cache saved to ${CACHE_PATH}`);
}

// --- Prompt building ---

const SYSTEM_PROMPT = `\
You are a dependency analyst for a GitHub API toolkit.
Your job is to identify which tools must be called before other tools in order to supply required input values.

A dependency edge means: run tool FROM first to obtain a value, then pass it as a required input to tool TO.

Rules:
1. Only emit an edge when TO genuinely cannot run without first calling FROM to get a specific value.
2. Do NOT emit edges for values a user already knows upfront: owner, repo, username, org name, or similar user-provided context.
3. The "label" must be the required input field name on the consumer tool (TO), not the producer's output field name.
4. Only use slugs from the tools listed in the batch.
5. Return ONLY a valid JSON array of edges. No explanation, no markdown fences.`;

/**
 * Keep only output fields that look like resource identifiers —
 * values another tool might need as input. Drop timestamps, URLs,
 * booleans, and free-text content fields.
 */
function isIdentifierField(field: SlimField): boolean {
  const n = field.name;
  if (n.endsWith("_at")) return false;                          // timestamps
  if (n.endsWith("_url") || n === "url") return false;         // URLs
  if (n.startsWith("is_") || n.startsWith("has_") ||
      n.startsWith("can_") || n.startsWith("allow_")) return false; // booleans
  const dropExact = new Set([
    "title", "body", "description", "message", "content", "text",
    "total_count", "size", "draft", "locked", "merged", "private",
    "fork", "archived", "disabled",
  ]);
  if (dropExact.has(n)) return false;
  // Keep names that are clearly identifiers
  const keepSuffixes = ["_id", "_number", "_sha", "_key", "_slug", "_token", "_ref", "_name", "_login"];
  if (keepSuffixes.some((s) => n.endsWith(s))) return true;
  const keepExact = new Set(["id", "number", "sha", "ref", "login", "name", "slug", "tag", "email", "type", "state"]);
  if (keepExact.has(n)) return true;
  return false;
}

function formatToolForPrompt(tool: SlimTool): string {
  const inputs = tool.requiredInputs.length
    ? tool.requiredInputs.map((f) => `    - ${f.name}: ${f.description}`).join("\n")
    : "    (none)";
  const identifierOutputs = tool.outputFields.filter(isIdentifierField);
  const outputs = identifierOutputs.length
    ? identifierOutputs.map((f) => `    - ${f.name}: ${f.description}`).join("\n")
    : "    (none)";
  return `TOOL: ${tool.slug}\n  REQUIRED INPUTS:\n${inputs}\n  OUTPUTS:\n${outputs}`;
}

function buildPrompt(batch: SlimTool[]): string {
  const toolList = batch.map(formatToolForPrompt).join("\n\n");
  return `\
Identify all dependency edges among the tools below.
Return a JSON array of objects: { "from": string, "to": string, "label": string }

Example:
  GITHUB_LIST_REPOSITORY_ISSUES outputs "number" (the issue number).
  GITHUB_CREATE_AN_ISSUE_COMMENT requires "issue_number" as input.
  Edge: {"from":"GITHUB_LIST_REPOSITORY_ISSUES","to":"GITHUB_CREATE_AN_ISSUE_COMMENT","label":"issue_number"}

Tools:

${toolList}

Return only the JSON array. If no dependencies exist return [].`;
}

// Services grouped by relatedness — keeps cross-service dependencies visible in one batch.
const SERVICE_GROUPS: string[][] = [
  ["issues", "comments", "labels", "milestones"],
  ["pulls", "reviews"],
  ["repos", "branches", "forks"],
  ["commits", "deployments", "releases"],
  ["actions"],
  ["orgs", "teams"],
  ["users"],
  ["projects"],
  ["environments", "checks", "security"],
  ["packages", "webhooks", "gists"],
  ["codespaces", "notifications", "pages"],
  ["sponsors", "classroom", "deploy_keys", "artifacts", "discussions", "stars", "migrations", "other"],
];

function groupIntoServiceBatches(slim: SlimTool[], maxBatchSize = 150): SlimTool[][] {
  const byService = new Map<string, SlimTool[]>();
  for (const tool of slim) {
    const svc = inferService(tool.slug);
    const list = byService.get(svc) ?? [];
    list.push(tool);
    byService.set(svc, list);
  }

  const batches: SlimTool[][] = [];
  for (const group of SERVICE_GROUPS) {
    const groupTools: SlimTool[] = group.flatMap((svc) => byService.get(svc) ?? []);
    // Split oversized groups into chunks of maxBatchSize.
    for (let i = 0; i < groupTools.length; i += maxBatchSize) {
      const chunk = groupTools.slice(i, i + maxBatchSize);
      if (chunk.length > 0) batches.push(chunk);
    }
  }
  return batches;
}

async function callLLMBatch(client: OpenAI, batch: SlimTool[], batchIndex: number): Promise<Edge[]> {
  const prompt = buildPrompt(batch);
  console.error(`  Batch ${batchIndex}: ${batch.length} tools → calling LLM...`);

  const response = await client.chat.completions.create({
    model: "openai/gpt-4o",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: prompt },
    ],
    temperature: 0,
  });

  const text = response.choices[0]?.message?.content ?? "[]";

  // Strip markdown fences if the model wraps its response.
  const json = text.replace(/^```(?:json)?\n?/i, "").replace(/\n?```$/i, "").trim();

  try {
    const edges: Edge[] = JSON.parse(json);
    console.error(`  Batch ${batchIndex}: got ${edges.length} edges`);
    return edges.filter(
      (e) => e.from && e.to && e.label && e.from !== e.to,
    );
  } catch {
    console.error(`  Batch ${batchIndex}: failed to parse response, skipping`);
    return [];
  }
}

async function llmEdges(slim: SlimTool[]): Promise<Edge[]> {
  const cached = loadCache();
  if (cached) {
    console.error(`LLM pass: loaded ${cached.length} edges from cache (pass --no-cache to regenerate)`);
    return cached;
  }

  const client = new OpenAI();
  const batches = groupIntoServiceBatches(slim);
  console.error(`LLM pass: ${batches.length} batches`);

  const allEdges: Edge[] = [];
  for (let i = 0; i < batches.length; i++) {
    const edges = await callLLMBatch(client, batches[i], i + 1);
    allEdges.push(...edges);
  }

  saveCache(allEdges);
  return allEdges;
}

async function generate(tools: Tool[]): Promise<Graph> {
  const slim = tools.map(extractSlimTool).filter((t): t is SlimTool => t !== null);

  const nodes: Node[] = slim.map((t) => ({ id: t.slug, service: inferService(t.slug) }));

  const hEdges = heuristicEdges(slim);
  console.error(`Heuristic pass: ${hEdges.length} edges found`);

  const lEdges = await llmEdges(slim);
  console.error(`LLM pass: ${lEdges.length} edges found`);

  // Merge and deduplicate by from+to+label
  const seen = new Set<string>();
  const allEdges: Edge[] = [];
  for (const e of [...hEdges, ...lEdges]) {
    const key = `${e.from}|${e.to}|${e.label ?? ""}`;
    if (!seen.has(key)) {
      seen.add(key);
      allEdges.push(e);
    }
  }

  return { nodes, edges: allEdges };
}

function writeViz(graph: Graph): void {
  const template = readFileSync(join(__dirname, "viz-template.html"), "utf-8");
  const html = template.replace("__GRAPH_DATA__", JSON.stringify(graph));
  writeFileSync("viz.html", html, "utf-8");
  console.error("wrote viz.html");
}

async function main() {
  const graph = await generate(loadCatalog());
  writeFileSync(OUT_PATH, JSON.stringify(graph, null, 2), "utf-8");
  console.error(
    `wrote ${graph.nodes.length} nodes, ${graph.edges.length} edges to ${OUT_PATH}`,
  );
  writeViz(graph);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
