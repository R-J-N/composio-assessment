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
import { readFileSync, writeFileSync } from "fs";

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
  if (s.includes("PULL_REQUEST") || s.includes("PULL")) return "pulls";
  if (s.includes("ISSUE"))        return "issues";
  if (s.includes("WORKFLOW") || s.includes("ACTION")) return "actions";
  if (s.includes("RELEASE"))      return "releases";
  if (s.includes("BRANCH"))       return "branches";
  if (s.includes("COMMIT"))       return "commits";
  if (s.includes("REVIEW"))       return "reviews";
  if (s.includes("COMMENT"))      return "comments";
  if (s.includes("LABEL"))        return "labels";
  if (s.includes("MILESTONE"))    return "milestones";
  if (s.includes("DEPLOYMENT"))   return "deployments";
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

async function generate(tools: Tool[]): Promise<Graph> {
  const slim = tools.map(extractSlimTool).filter((t): t is SlimTool => t !== null);

  const nodes: Node[] = slim.map((t) => ({ id: t.slug, service: inferService(t.slug) }));
  const edges: Edge[] = [];
  return { nodes, edges };
}

async function main() {
  const graph = await generate(loadCatalog());
  writeFileSync(OUT_PATH, JSON.stringify(graph, null, 2), "utf-8");
  console.error(
    `wrote ${graph.nodes.length} nodes, ${graph.edges.length} edges to ${OUT_PATH}`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
