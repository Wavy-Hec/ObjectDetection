# Design Phase Command

**Purpose:** Create a comprehensive design document for a feature  
**Input:** Feature requirements (PRD.md)  
**Output:** design.md document  
**When to use:** After PRD is approved, before implementation planning

---

## Prompt for AI Assistant

```
You are a software architect working on the Real-Time Object Tracking project.

CONTEXT:
- Review the PAD.md (Project Architecture Document) for overall system architecture
- Review the feature PRD at feature-[X]/PRD.md for requirements
- Understand existing design patterns from previous features

TASK:
Create a detailed design document (design.md) that includes:

1. **Objective**: Clear statement of what this feature accomplishes

2. **Scope and Assumptions**: 
   - What is included/excluded
   - Key assumptions about inputs, environment, constraints
   - Dependencies on existing components

3. **Design Details**:
   - Input and output specifications
   - Data structures and formats
   - Processing algorithms
   
4. **System Components**:
   - Module descriptions
   - Class and function interfaces
   - Component interactions (with diagrams)
   
5. **Success Metrics**:
   - Performance targets
   - Quality criteria
   - How to measure success
   
6. **Risks and Mitigations**:
   - Technical risks
   - Performance concerns
   - Fallback strategies

REQUIREMENTS:
- Design must align with PAD.md architecture
- Use existing modules/patterns where possible
- Keep design modular and testable
- Consider performance implications
- Document all assumptions

OUTPUT FORMAT:
- Clear headings and sections
- ASCII diagrams for architecture
- Code examples for key interfaces
- Concrete, measurable metrics

Please create the design document now.
```

---

## AI Assistant Instructions

1. **Read PAD.md first** to understand the overall architecture
2. **Read feature PRD** to understand specific requirements
3. **Review existing code** to understand patterns and conventions
4. **Create design.md** with all required sections
5. **Ensure alignment** between design and PAD
6. **Be specific** - avoid vague descriptions
7. **Include examples** - show data structures and interfaces

---

## Checklist

Before moving to the planning phase, ensure:

- [ ] Design addresses all requirements from PRD
- [ ] Design aligns with PAD architecture
- [ ] All components and interfaces are specified
- [ ] Data structures are clearly defined
- [ ] Success metrics are measurable
- [ ] Risks are identified with mitigations
- [ ] Design is reviewed and approved
- [ ] Assumptions are documented

---

## Common Mistakes to Avoid

1. **Too vague**: "The system will process data efficiently"
   - ✅ Better: "The system will process frames at > 30 FPS using GPU acceleration"

2. **Missing interfaces**: Not specifying function signatures
   - ✅ Better: Include exact function signatures with parameters and return types

3. **Ignoring PAD**: Creating design that conflicts with architecture
   - ✅ Better: Reference PAD sections and align with existing patterns

4. **No metrics**: Not defining how to measure success
   - ✅ Better: Specific, measurable performance and quality targets

5. **Skipping risks**: Not identifying potential problems
   - ✅ Better: List technical and performance risks with mitigations

---

## Example Usage

```bash
# Step 1: Ensure you have context
cat PAD.md
cat feature-2/PRD.md

# Step 2: Use the prompt with AI assistant
# [Copy prompt from above]

# Step 3: Save output to design.md
# [AI generates design.md]

# Step 4: Review and refine
# Check alignment with PAD
# Verify all requirements covered
# Validate metrics are measurable

# Step 5: Get approval and move to planning
```

---

## Integration with Workflow

```
PRD.md (Requirements)
    ↓
[USE THIS COMMAND]
    ↓
design.md (Architecture & Design)
    ↓
plan.md (Task Breakdown)
    ↓
Implementation
```

---

**Last Updated:** 2025-01-27  
**Version:** 1.0
