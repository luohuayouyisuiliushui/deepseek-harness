/**
 * Threshold-free measurement of the persisted Session listing over a real corpus.
 *
 * The listing used to traverse the corpus root twice — once to resolve the
 * selected generations and once to fingerprint them for the historical
 * revisions — and stat every artifact twice. It now traverses once and reuses
 * the identities it already collected. This diagnostic prices that difference
 * on the corpus the reader actually has, instead of on a synthetic one.
 *
 * It reports three numbers:
 *
 * - `currentListing`: the real `ctx.sessionPersistence.list()` call, timed.
 * - `extraTraversalMs` / `extraStatMs`: what the previous algorithm added —
 *   one more root traversal and one more stat pass over the same files.
 * - `previousListingEstimatedMs`: the current listing plus that extra work.
 *
 * The traversal and stat passes are measured directly against the corpus
 * layout, so the operation counts are exact for the model and the timings are
 * observations, not a claim about any other machine.
 *
 * Run from the repository root:
 *
 * ```sh
 * pnpm exec tsx packages/session/session-persistence-jsonl/tests/session-list.perf.ts [corpus-root]
 * ```
 *
 * The corpus root defaults to the `sessions` directory under the dsh home. The
 * measurement is a warm-filesystem observation: it claims no disk-cold latency
 * and establishes no threshold. It reads the corpus; the backend's own listing
 * opens the root it is given and lists it.
 */

import { Context } from '@deepseek-ai/cordis'
import * as JsonlSessionPersistence from '../src/index.ts'
import { readdir, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { performance } from 'node:perf_hooks'
import pLimit from 'p-limit'

/** Concurrency the backend's listing limiter uses by default. */
const LIST_CONCURRENCY = 3
/** Repetitions per measurement; the minimum is reported. */
const REPEATS = 3

/** The generation file names a Session directory may hold. */
const GENERATION_PATTERN = /^session(?:\.v(\d+))?\.jsonl(?:\.zstd)?$/

/** One traversal of the corpus: the directories read and the selected generations. */
interface Traversal {
  /** Directory reads the traversal performed. */
  readonly readdirs: number
  /** The selected generation file of every Session directory. */
  readonly selected: readonly string[]
}

/**
 * The generation version a file name carries.
 * @param name - the file name inside a Session directory.
 * @returns the generation version, or undefined when the file is not a generation.
 */
function generationVersion(name: string): number | undefined {
  const match = GENERATION_PATTERN.exec(name)
  if (match === null) return undefined
  return match[1] === undefined ? 0 : Number(match[1])
}

/**
 * Walk the corpus the way the backend walks it: every directory is read once,
 * and a Session directory contributes its highest-version generation file.
 * @param root - the persistence root.
 * @returns the directory reads and the selected generation files.
 */
async function traverse(root: string): Promise<Traversal> {
  let readdirs = 0
  const selected: string[] = []
  const visit = async (dir: string): Promise<void> => {
    const entries = await readdir(dir, { withFileTypes: true })
    readdirs += 1
    let best: { readonly path: string; readonly version: number } | undefined
    for (const entry of entries) {
      if (entry.isDirectory()) {
        await visit(join(dir, entry.name))
        continue
      }
      const version = generationVersion(entry.name)
      if (version === undefined) continue
      if (best === undefined || version > best.version) {
        best = { path: join(dir, entry.name), version }
      }
    }
    if (best !== undefined) selected.push(best.path)
  }
  await visit(root)
  return { readdirs, selected }
}

/**
 * Stat every listed file through a limiter of the backend's default width.
 * @param paths - the files to stat.
 * @returns the elapsed milliseconds.
 */
async function statPass(paths: readonly string[]): Promise<number> {
  const limit = pLimit(LIST_CONCURRENCY)
  const start = performance.now()
  await Promise.all(paths.map(path => limit(async () => { await stat(path, { bigint: true }) })))
  return performance.now() - start
}

/**
 * Time the real listing against the corpus.
 * @param root - the persistence root.
 * @returns the elapsed milliseconds and the number of listed snapshots.
 */
async function timeListing(root: string): Promise<{ readonly ms: number; readonly snapshots: number }> {
  const ctx = new Context()
  try {
    await ctx.plugin(JsonlSessionPersistence, { root })
    const start = performance.now()
    const snapshots = await ctx.sessionPersistence.list()
    return { ms: performance.now() - start, snapshots: snapshots.length }
  } finally {
    await ctx.fiber.dispose()
  }
}

/**
 * Measure the listing and the work the previous algorithm added.
 * @param root - the persistence root.
 * @returns the measured report.
 */
async function measure(root: string): Promise<Record<string, unknown>> {
  const first = await traverse(root)
  const traversalMs = await (async () => {
    const start = performance.now()
    await traverse(root)
    return performance.now() - start
  })()
  const statMs = await statPass(first.selected)
  const listings: number[] = []
  let snapshots = 0
  for (let repeat = 0; repeat < REPEATS; repeat++) {
    const listing = await timeListing(root)
    listings.push(listing.ms)
    snapshots = listing.snapshots
  }
  const currentMs = Math.min(...listings)
  return {
    root,
    sessions: snapshots,
    sessionDirectories: first.selected.length,
    traversal: { readdirs: first.readdirs, ms: traversalMs },
    statPass: { files: first.selected.length, ms: statMs },
    currentListing: { ms: currentMs, samples: listings },
    previousListingEstimated: { ms: currentMs + traversalMs + statMs },
    operationCounts: {
      // The previous algorithm stat'ed every selected generation for the
      // fingerprint and every readable artifact again for its own revision.
      previous: { readdirs: first.readdirs * 2, stats: first.selected.length + snapshots },
      // The current one stats each selected generation once and reuses that
      // identity for both the fingerprint and the artifact revision.
      current: { readdirs: first.readdirs, stats: first.selected.length },
    },
    node: process.version,
    platform: `${process.platform} ${process.arch}`,
  }
}

const root = process.argv[2] ?? join(homedir(), '.dsh', 'sessions')
try {
  const report = await measure(root)
  process.stdout.write(JSON.stringify(report, null, 2) + '\n')
} catch (error: unknown) {
  process.stdout.write(JSON.stringify({ root, error: String(error) }, null, 2) + '\n')
  process.exitCode = 1
}
