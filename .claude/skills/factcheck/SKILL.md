---
name: factcheck
description: "Comprehensive fact-checking skill that validates outputs against source data and external references. Invoked with /factcheck. Use this skill to audit any analysis, report, or strategic document for accuracy. The skill dispatches explore agents to validate claims against source materials (data/, docs/, input/), uses web searches to corroborate external facts and assumptions, and generates a detailed audit report."
---

# Fact-Check Skill

You are executing a comprehensive fact-checking audit. This is a three-phase process that validates outputs against source materials and external references.

## PHASE 1: Source Data Validation (Explore Agents)

Dispatch multiple explore agents in parallel to examine the source materials and validate claims found in output documents.

### Step 1.1: Identify Target Document
First, identify what document(s) need fact-checking:
- If the user specified a file, use that file
- If no file specified, examine the most recent files in `output/` folder
- Read the target document(s) to extract all factual claims

### Step 1.2: Extract Claims
From the target document(s), extract and categorize all factual claims:

**Quantitative Claims** (numbers, percentages, statistics):
- Survey statistics (e.g., "178 responses", "23.6% mentioned X")
- Financial projections (e.g., "$150K investment", "320-450% ROI")
- Time estimates (e.g., "5-7 months payback")
- Counts and rankings

**Qualitative Claims** (statements of fact):
- Market conditions
- Company capabilities and offerings
- Strategic assertions
- Historical facts

**Source References**:
- Claims citing specific documents
- References to survey data
- References to playbook or other docs

### Step 1.3: Dispatch Explore Agents
Launch parallel explore agents to search for supporting evidence:

**Agent 1: Survey Data Validator**
- Explore `data/` folder (1q.json, 1q.csv)
- Validate all survey-related statistics
- Check topic counts, percentages, response totals

**Agent 2: Documentation Validator**
- Explore `docs/` folder (salesplaybook.md, etc.)
- Verify claims about company offerings, ICPs, methodologies
- Check service descriptions and pricing references

**Agent 3: Input/Context Validator**
- Explore `input/` folder if present
- Check any additional source materials
- Validate background information

### Step 1.4: Compile Source Validation Results
For each claim, record:
- **GROUNDED**: Claim has direct supporting evidence in source materials
- **PARTIALLY GROUNDED**: Claim has some support but numbers/details differ
- **INFERRED**: Claim is a reasonable inference from source data but not explicitly stated
- **UNGROUNDED**: No supporting evidence found in source materials

---

## PHASE 2: External Corroboration (Web Searches)

For claims that require external validation or involve assumptions about market conditions, industry trends, or external facts:

### Step 2.1: Identify Claims Needing External Validation
Select claims that:
- Reference market conditions or industry trends
- Make assumptions about external factors
- Cite statistics that should be publicly verifiable
- Reference tools, platforms, or technologies
- Make competitive or market positioning claims

### Step 2.2: Execute Web Searches
For each identified claim:
1. Formulate appropriate search queries
2. Execute web searches using WebSearch tool
3. Compare search results to claimed facts
4. Note any discrepancies or confirmations

### Step 2.3: Compile External Validation Results
For each externally-validated claim, record:
- **CORROBORATED**: External sources confirm the claim
- **PARTIALLY CORROBORATED**: External sources support some aspects
- **CONTRADICTED**: External sources contradict the claim
- **UNABLE TO VERIFY**: No reliable external sources found

---

## PHASE 3: Audit Report Generation

Generate a comprehensive markdown audit report.

### Step 3.1: Create Report Structure

```markdown
# Fact-Check Audit Report

**Generated:** [YYYY-MM-DD-HH-MM]
**Audited Document(s):** [list files checked]
**Audit Scope:** [describe what was checked]

---

## Executive Summary

[Brief overview of findings: X claims verified, Y partially verified, Z unverified]

---

## Section 1: Claims Grounded in Source Materials

### Verified Claims (Full Match)
| Claim | Source | Location | Status |
|-------|--------|----------|--------|
| [claim text] | [source file] | [line/section] | GROUNDED |

### Partially Verified Claims (Partial Match)
| Claim | Source | Discrepancy | Status |
|-------|--------|-------------|--------|
| [claim text] | [source file] | [what differs] | PARTIALLY GROUNDED |

---

## Section 2: Claims Corroborated by External Sources

### Externally Verified Claims
| Claim | Search Query | Sources Found | Status |
|-------|--------------|---------------|--------|
| [claim text] | [query used] | [source URLs] | CORROBORATED |

### Claims with External Discrepancies
| Claim | Search Query | Finding | Status |
|-------|--------------|---------|--------|
| [claim text] | [query used] | [what was found] | CONTRADICTED |

---

## Section 3: Unverified or Potentially Hallucinated Content

### Claims Without Source Evidence
| Claim | Category | Risk Level | Recommendation |
|-------|----------|------------|----------------|
| [claim text] | [quantitative/qualitative] | [High/Medium/Low] | [action needed] |

### Inferred Claims (Reasonable but Unverified)
| Claim | Basis for Inference | Confidence | Recommendation |
|-------|---------------------|------------|----------------|
| [claim text] | [why it seems reasonable] | [High/Medium/Low] | [action needed] |

---

## Section 4: Recommendations for Revision

### High Priority (Immediate Action)
1. [Specific revision recommendation]
2. [Specific revision recommendation]

### Medium Priority (Should Address)
1. [Specific revision recommendation]

### Low Priority (Consider for Accuracy)
1. [Specific revision recommendation]

---

## Audit Methodology

- **Source Materials Checked:** [list all files examined]
- **Web Searches Conducted:** [number]
- **Total Claims Analyzed:** [number]
- **Verification Rate:** [X% grounded in sources, Y% externally corroborated]

---

*This audit report was generated by the /factcheck skill to ensure document accuracy and grounding in source materials.*
```

### Step 3.2: Save Report
Save the audit report to `output/` with filename format:
`YYYY-MM-DD-HH-MM-factcheck-audit.md`

---

## Execution Instructions

When this skill is invoked:

1. **Parse the invocation context** - determine if a specific file was mentioned or if auditing recent outputs

2. **Read target document(s)** - gather all content to be fact-checked

3. **Phase 1 - Source Validation**:
   - Use Task tool with `subagent_type: Explore` to dispatch parallel agents
   - Each agent searches specific source folders
   - Collect and compile results

4. **Phase 2 - External Validation**:
   - Use WebSearch tool for claims requiring external corroboration
   - Focus on market claims, industry statistics, technology facts
   - Note: Be selective - don't search for every claim, only those that benefit from external validation

5. **Phase 3 - Report Generation**:
   - Compile all findings into structured markdown report
   - Generate datetime-stamped filename
   - Write to `output/` folder

6. **Provide Summary** - After generating the report, provide the user with:
   - Location of the audit report
   - Key findings summary
   - Highest priority items needing attention

---

## Usage Examples

**User invokes:** `/factcheck`
- Audits the most recent file(s) in `output/`

**User invokes:** `/factcheck output/2026-01-13-11-00-2026-strategic-plan.md`
- Audits the specific file mentioned

**Agent invokes internally:** After generating analysis, an agent can invoke `/factcheck` to validate its own outputs

---

## Important Notes

- This skill is designed to be thorough but efficient - not every claim needs external validation
- Focus web searches on claims that SHOULD have external evidence (market data, public statistics)
- For internal projections/calculations, source validation is sufficient
- Flag hallucinations clearly but distinguish from reasonable inferences
- The audit report should be actionable - provide specific revision guidance
