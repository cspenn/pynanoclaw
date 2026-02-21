# RULES.MD - Chrome Extension (MV3) Development Standards

**Target**: Chromium Manifest V3 (2026), TypeScript 5.8+, Node 24+ (LTS).
**Scope**: This document governs all code generation, architecture, and quality assurance for the extension.

---

## PART 1: FIRST PRINCIPLES

These principles take precedence over all other rules.

### P1: Fix Over Create
- **Refactor First**: Always update/fix existing modules rather than creating new files.
- **Complexity Cap**: When `eslint complexity` or `sonar-scanner` indicates cognitive complexity > 10, refactor into smaller, pure functions or hooks.
- **Hook Extraction**: If a React component or Background listener exceeds 100 lines, extract logic into a custom hook (`useLogic.ts`) or utility.

### P2: Reusable Testing Infrastructure
- **NO Console Scripting**: Do not write one-off JS scripts in the root.
- **NPM Scripts as Source of Truth**: All diagnostic logic lives in `scripts/` and is exposed via `package.json` scripts.
- **Unit vs E2E**: Pure logic tests belong in `src/**/*.test.ts` (Vitest). Browser interaction tests belong in `e2e/` (Playwright).

### P3: Documentation Location
- ALL documentation goes in `docs/`.
- **Manifest is Documentation**: `manifest.json` must be heavily commented (using JSONC format if supported by build tool, or standard JSON with rigorous strictness).
- **Self-Documenting Types**: TypeScript interfaces serve as the primary documentation for data structures.

### P4: Never Defer Necessary Work
- **Strict Mode Always**: `tsconfig.json` must have `"strict": true` and `"noImplicitAny": true`. Never use `@ts-ignore` to silence errors; fix the type definition.
- **Async Hygiene**: Never leave a Promise floating (dangling). All Promises must be `await`ed or explicitly `.catch()`ed.

### P5: Component/Module Isolation
- **Service Worker Purity**: The Background Service Worker must be event-driven and ephemeral. Never rely on global variables in `background.ts` persisting (they die when the worker sleeps).
- **Storage as State**: `chrome.storage` is the only source of truth for persistent state.

---

## PART 2: CORE STANDARDS

### 2.1 Project Structure

| Requirement | Rule |
|-------------|------|
| **File Markers** | Start files with purpose comment: `// Service Worker Entry Point` |
| **Imports** | Absolute imports via aliases only (e.g., `@/components/...`). No `../../`. |
| **Entry Points** | `background.ts` (Logic), `popup.tsx` (UI), `content.ts` (DOM). |
| **Complexity** | No file exceeds 250 LOC. Components must satisfy Single Responsibility. |

**Required Files:**
- `manifest.json` (The heart of the extension)
- `vite.config.ts` (Build orchestration)
- `.npmrc` (Dependency strictness)

### 2.2 Configuration

- **NO Runtime Env Vars**: Browsers cannot read process.env at runtime.
- **Build-Time Config**: Use `import.meta.env` (Vite) for constants injected at build time.
- **Secrets Management**: `credentials.json` or `.env` must be `.gitignore`d. Secrets are injected during the build process, NEVER committed.
- **Runtime Validation**: All configuration loaded from `chrome.storage` or APIs MUST be validated with **Zod** schemas immediately upon retrieval.

### 2.3 Build & Task Interface (CLI equivalent)

- Use **Vite** for all building and bundling.
- `package.json` is the command center.
- **Strictness**: The build command `pnpm build` must fail if there is a single TypeScript error or unused import.

### 2.4 Code Quality Principles

| Principle | Meaning |
|-----------|---------|
| **DRY** | Don't Repeat Yourself (Extract to `utils/`) |
| **Immutability** | Prefer `const` and spread operators over mutation. |
| **SOLID** | Apply specifically to TypeScript Classes/Interfaces. |
| **Hooks Pattern** | Logic goes in Hooks, UI goes in Components. |

**Rules:**
- **TSDoc**: All exported functions must have TSDoc comments explaining `@param` and `@returns`.
- **Pure Functions**: Prefer pure functions for all business logic (easier to test with Vitest).

### 2.5 Environment & Tooling

| Tool | Purpose |
|------|---------|
| **Node.js 24+** | Runtime environment. |
| **pnpm** | Package management (Enforces strict dependency tree, faster than npm). |
| **Vite** | Build tool / Bundler (Replaces Webpack). |
| **CRXJS** | Vite plugin specifically for Chrome Extensions. |

**pnpm commands:**
```bash
pnpm install            # Install dependencies
pnpm add <package>      # Add dependency
pnpm run dev            # HMR Dev Server
pnpm run build          # Production Build
```

---

## PART 3: QUALITY GATE CHECKLIST

### Tier 1 - Gate Checks (Must Pass Before Commit)

| Tool | Command | Purpose |
|------|---------|---------|
| **Biome** | `pnpm biome check src/` | Ultra-fast Linting & Formatting (Replaces Prettier/ESLint). |
| **TypeScript** | `pnpm tsc --noEmit` | Strict Type Checking. |
| **Vitest** | `pnpm vitest run` | Unit Tests. |
| **Knip** | `pnpm knip` | Detect unused files, exports, and dependencies. |

### Tier 2 - Quality Analysis

| Tool | Command | Purpose |
|------|---------|---------|
| **ESLint (Compat)** | `pnpm eslint src/` | Specific browser compatibility checks (`eslint-plugin-compat`). |
| **Audit** | `pnpm audit` | Security vulnerability scanning. |
| **Size Limit** | `pnpm size-limit` | Ensure bundle size doesn't bloat. |

### Tier 3 - Advanced

| Tool | Command | Purpose |
|------|---------|---------|
| **Playwright** | `pnpm playwright test` | End-to-End browser testing (loads extension in real Chrome). |
| **SonarQube** | (CI Only) | Deep static analysis and code smells. |

---

## PART 4: 8-DIMENSION QA FRAMEWORK

Evaluate code on these 8 dimensions (The "Extension Context"):

| Dimension | Question |
|-----------|----------|
| **Async/Sync** | Are we blocking the main thread? (Critical in UI). |
| **CSP Compliance** | Does this violate Content Security Policy (e.g., `eval`, inline scripts)? |
| **Memory Leak** | Are listeners (`chrome.runtime.onMessage`) cleaned up in useEffect/Hooks? |
| **Permission** | Is the permission strictly necessary in `manifest.json`? |
| **Quota** | Does this hit `chrome.storage.sync` write limits? |
| **Service Worker** | Will this fail when the Service Worker goes to sleep? |
| **DOM Isolation** | Does this CSS bleed into the host page? (Use Shadow DOM). |
| **Offline** | Does it handle network failure gracefully? |

---

## PART 5: IMPLEMENTATION STANDARDS

### 5.1 Type Hints (TypeScript 5.8+)

**Modern Syntax (USE THIS):**
```typescript
// Explicit return types always
const processItems = (items: string[]): Record<string, number> => {
  const result: Record<string, number> = {};
  return result;
};

// Zod for Runtime Validation (Equivalent to Pydantic)
import { z } from 'zod';
const UserSchema = z.object({
  id: z.string().uuid(),
  name: z.string(),
});
type User = z.infer<typeof UserSchema>;

// Generics for Storage
const save = <T>(key: string, data: T): Promise<void> => { ... }
```

**Anti-Patterns (AVOID):**
```typescript
// DON'T USE:
const data: any = ...          // "any" is strictly forbidden
const data: Object = ...       // use Record<string, unknown>
interface IProps { ... }       // Don't prefix interfaces with "I"
```

### 5.2 Core Libraries

| Domain | Library | Rule |
|--------|---------|------|
| **State/Storage** | `chrome.storage` + Hooks | Single source of truth. Wrapper required for type safety. |
| **HTTP** | `fetch` (Native) | Wrapper required for error handling. No Axios (bloat). |
| **Message Passing** | `webext-bridge` | Strongly typed message passing between Content Script and Background. |
| **Validation** | **Zod** | All external data (API, Storage, DOM) must be parsed via Zod. |
| **DOM** | **React** | Use React for Popup/Options/Sidepanel. Use Shadow DOM for Content Scripts. |

### 5.3 Logging & Error Handling

```typescript
import { logger } from '@/utils/logger';

// Loggers must be stripped in production builds via Vite/Terser
logger.info("Sync complete");
logger.warn("Rate limit approaching");
logger.error("Auth failed", error);
```

**Rules:**
- **No `console.log` in production**: Build pipeline must strip these.
- **Error Boundaries**: React components must be wrapped in Error Boundaries.
- **Telemetry**: If allowed, log errors to a service (e.g., Sentry), but respect privacy (PII stripping).

### 5.4 Testing Strategy

**Vitest (Unit):**
- Test utility functions.
- Test Reducers/State logic.
- Mock `chrome` API using `vitest-chrome` or `sinon-chrome`.

**Playwright (E2E):**
- Launch Chrome with the extension loaded.
- Test Popup interactions.
- Test Content Script injection on real websites.

**Critical Rule**: If a test fails, the build fails. No "flaky" tests allowed—fix them or delete them.

### 5.5 Progress & Reporting

- **UI**: Use Skeleton loaders (React Suspense) for async data.
- **Background**: Use `chrome.action.setBadgeText` to communicate status to the user without opening the popup.

---

## PART 6: VUW METHODOLOGY (Adapted)

**Verifiable Units of Work** - Micro-plans for disciplined engineering.

### VUW Template

```markdown
**VUW_ID:** [e.g., EXT-FIX-001]

**Objective:** [Why this matters]

**Files:** [List of files: background.ts, manifest.json, etc.]

**Pre-Work Checkpoint:** git commit

**Steps:**
1. [Literal instruction: "Update manifest.json host_permissions..."]
2. [Code snippet for Zod schema update]

**Verification:**
- [ ] `pnpm build` succeeds (No TS errors)
- [ ] `pnpm vitest` passes
- [ ] `pnpm biome check` passes
- [ ] Load unpacked extension -> No console errors in Background or Content Script

**Post-Work Checkpoint:** git commit
```

---

## Manifest V3 & TypeScript Reference

| Feature | Usage |
|---------|-------|
| `chrome.scripting` | Inject CSS/JS programmatically (replaces code injection). |
| `declarativeNetRequest` | Block/Modify network requests (replaces `webRequest` blocking). |
| `Offscreen Documents` | DOM parsing typically done in background (now requires offscreen). |
| `Storage Access API` | Handling third-party cookie restrictions. |

---

## Anti-Patterns (NEVER DO)

- **Eval()**: Strictly forbidden by CSP.
- **Remote Code**: Loading JS from a CDN (analytics, libraries). All code must be bundled locally.
- **Unbounded Storage**: Saving unlimited data to `local` storage without cleanup strategies.
- **Sync/Blocking XHR**: Deprecated and will crash. Use `fetch` and `await`.
- **Mixing Paradigms**: Don't mix `then().catch()` chains with `async/await`. Stick to `async/await`.
- **Ignoring cleanup**: `useEffect` must always return a cleanup function if it sets a listener.