---
name: qa-text
description: Run text quality audit when user asks to check written text for quality assurance QA. This includes reviewing text and critical thinking for text.
---

You will be performing a comprehensive quality assessment of written text. The text to analyze is provided below:

<text>
{{TEXT}}
</text>

Your task is to evaluate this text across 4 key dimensions and generate a detailed quality assessment report. This analysis should focus on the content, structure, clarity, and effectiveness of the written text.

## Four-Dimension Analysis Framework

Analyze the text according to these four dimensions:

### 1. What's Good
Identify strengths in the text, including:
- Clear, effective communication
- Well-structured arguments or narratives
- Strong evidence or examples
- Appropriate tone and style for the intended audience
- Good grammar, spelling, and punctuation
- Effective use of formatting or organization
- Compelling or engaging content
- Logical flow and coherence

### 2. What's Bad
Identify problems and issues, including:
- **Critical**: Factual errors, misleading information, offensive content, major logical fallacies
- **High**: Significant clarity issues, confusing structure, inappropriate tone, major grammatical errors
- **Medium**: Awkward phrasing, minor logical inconsistencies, weak arguments, repetitive content
- **Low**: Minor grammatical issues, stylistic inconsistencies, minor formatting issues

### 3. What's Missing
Identify gaps and omissions, including:
- Missing context or background information
- Lack of supporting evidence or examples
- Missing transitions between ideas
- Incomplete arguments or explanations
- Absent conclusions or summaries
- Missing citations or references (if applicable)
- Lack of clear structure (headings, paragraphs, etc.)
- Undefined terms or jargon

### 4. What's Unnecessary
Identify excess or distracting elements, including:
- Redundant information or repetitive points
- Irrelevant tangents or off-topic content
- Excessive jargon or overly complex language
- Filler words or phrases that don't add value
- Overly verbose explanations
- Unnecessary formatting or visual clutter

## Analysis Process

Before writing your final report, use the scratchpad below to organize your thoughts:

<scratchpad>
- Read through the entire text carefully
- Note specific examples for each dimension with references to specific sentences or sections
- Assess the overall quality and assign a health score (A/B/C/D/F)
- Prioritize findings by severity and importance
- Formulate concrete, actionable recommendations
</scratchpad>

## Report Format

Generate your assessment report in the following markdown format:

```markdown
# Text Quality Assessment Report

**Date:** [Current date]
**Text Type:** [Identify the type: article, documentation, email, report, etc.]

---

## Executive Summary

[Provide 2-3 sentences summarizing the overall quality and main findings]

**Overall Quality Score:** [A/B/C/D/F] - [Brief justification]

| Dimension | Status | Summary |
|-----------|--------|---------|
| What's Good | [✅/⚠️/❌] | [Brief summary] |
| What's Bad | [✅/⚠️/❌] | [Count and severity] |
| What's Missing | [✅/⚠️/❌] | [Count of gaps] |
| What's Unnecessary | [✅/⚠️/❌] | [Count of excess] |

---

## 1. What's Good

[List specific strengths with examples from the text. Quote relevant passages when helpful.]

---

## 2. What's Bad

### Critical Issues
[List any critical problems that must be addressed immediately]

### High Severity Issues
[List significant problems that seriously impact quality]

### Medium Severity Issues
[List moderate problems that should be addressed]

### Low Severity Issues
[List minor issues for polish and refinement]

---

## 3. What's Missing

[List gaps, omissions, and missing elements that would improve the text. Be specific about what should be added and where.]

---

## 4. What's Unnecessary

[List redundant, irrelevant, or excessive elements that should be removed or simplified. Quote specific examples.]

---

## Recommendations

### High Priority
1. [Most critical improvements needed]
2. [Additional high-priority items]

### Medium Priority
1. [Important improvements]
2. [Additional medium-priority items]

### Low Priority
1. [Nice-to-have enhancements]
2. [Additional low-priority items]

---

## Conclusion

[Provide a final 1-2 sentence assessment and overall recommendation]

---
```

Your final output should be the complete quality assessment report in the markdown format specified above. Do not include the scratchpad in your final output - only the formatted report itself.
